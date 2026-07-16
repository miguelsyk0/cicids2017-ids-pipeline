"""
preprocessing.py
----------------
Reusable preprocessing module for the ITE 17 IDS project.

Covers the steps that sit between "flat data out of SQL (cic_typed)" and
"model.fit()": timestamp engineering, splitting, categorical encoding,
scaling, class balancing, and label encoding. Built to be shared by
XGBoost and any baseline (e.g. Random Forest) so both use IDENTICAL
preprocessing -- if they don't, comparing their scores is not valid.

Recommended pipeline order (each step's function is below):
    1. engineer_timestamp(df)               -- BEFORE split_data()
    2. split_data(df, target_col="label")    -- drop "binary_label" from df
                                                 first unless it's an
                                                 intentional secondary target
    3. encode_categoricals(X_train, X_test)  -- one-hot, fit-shape from train
    4. fit_scaler(X_train_encoded, exclude=[...dummy cols, "hour_of_day"])
       transform_scaler(scaler, X_test_encoded)
    5. apply_smote(X_train_scaled, y_train)  -- y_train is still TEXT here;
                                                 sampling_strategy dict keys
                                                 use human-readable labels
    6. encode_labels(y_train_resampled, y_test) -- text -> int, LAST, right
                                                 before the labels reach
                                                 XGBoost (which requires
                                                 integer-encoded classes)

Design decisions this file enforces (do not skip these when using it):
    1. Split happens BEFORE scaling, encoding, and SMOTE. Fitting any of
       these on the full dataset before splitting leaks test-set
       information into training -- your reported metrics would be
       inflated and wouldn't survive a panel member asking "did you split
       first?"
    2. The scaler and the one-hot column shape are both fit ONLY on
       X_train, then applied via .transform() (not .fit_transform()) to
       X_test. Same leakage reasoning as above.
    3. SMOTE is applied ONLY to the training set, AFTER scaling/encoding,
       and is never applied to X_test / y_test. Test data must reflect
       real-world class imbalance to mean anything.
    4. Isolation Forest and the rule-based layer should NOT use the SMOTE
       output of this file -- only XGBoost (and baseline comparisons like
       Random Forest) should. Isolation Forest wants scaled-but-unbalanced
       data; the rule-based layer wants raw, unscaled, named features.
    5. Categorical columns (protocol, port_group, source_day) are one-hot
       encoded rather than left as pandas 'category' dtype. This is a
       deliberate choice: BorderlineSMOTE/SMOTEENN (confirmed by testing)
       cannot consume 'category' dtype -- that requires SMOTENC, which is
       more machinery than three low-cardinality columns justify. One-hot
       keeps everything numeric and compatible with the SMOTE step below.
    6. Label encoding happens LAST, after SMOTE, right before the model.
       SMOTE itself works fine with text labels (verified), so keeping
       "label" as text through the SMOTE step preserves the human-readable
       sampling_strategy dicts (e.g. {"Infiltration": 2000}) documented in
       apply_smote() below.

Libraries used:
    - pandas               -> DataFrame handling, get_dummies() for one-hot
    - sklearn.model_selection.train_test_split -> stratified train/test split
    - sklearn.preprocessing.RobustScaler -> scaling resistant to outliers
                              (chosen over StandardScaler because network
                              flow features like Flow Bytes/s have extreme
                              outliers even after cleaning Infinity/NaN)
    - sklearn.preprocessing.LabelEncoder -> converts text class labels to
                              the integer format XGBoost's sklearn API
                              requires (confirmed: raw text labels raise
                              ValueError otherwise)
    - imblearn              -> BorderlineSMOTE / SMOTEENN for class balancing,
                              chosen over vanilla SMOTE because CICIDS2017
                              has attack classes as small as ~30 samples
                              (e.g. Infiltration); vanilla SMOTE would
                              interpolate near-duplicate synthetic points
                              from such a small pool.
    - joblib               -> saves/loads fitted scaler/encoder objects so
                              the exact same transform can be reused later
                              by the Streamlit app on live pcap-derived
                              features.
"""

import pandas as pd
import joblib
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN
from typing import cast, Union
import numpy as np


def engineer_timestamp(df: pd.DataFrame, timestamp_col: str = "timestamp", drop_original: bool = True) -> pd.DataFrame:
    """
    Converts a raw datetime column into an "hour_of_day" numeric feature.
    Run this BEFORE split_data() -- it's a row-wise transform, not something
    that needs train/test fitting.

    Why hour-of-day instead of dropping the column outright: certain attack
    types in CICIDS2017 cluster in specific time windows within each capture
    day, so time-of-day carries real signal worth keeping. Neither
    RobustScaler nor XGBoost can consume a raw datetime object directly, so
    it has to be transformed into something numeric before it reaches
    fit_scaler().

    Parameters:
        df (pd.DataFrame): must contain `timestamp_col` as a datetime (or
                        datetime-parseable) column, e.g. the "timestamp"
                        column from cic_typed.
        timestamp_col (str): name of the column to engineer. Defaults to
                        "timestamp" to match this project's schema.
        drop_original (bool): if True, removes the raw datetime column
                        after extracting hour_of_day.

    Returns:
        pd.DataFrame with a new "hour_of_day" integer column (0-23).

    Example:
        df = engineer_timestamp(df)   # run before split_data()
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["hour_of_day"] = df[timestamp_col].dt.hour
    if drop_original:
        df = df.drop(columns=[timestamp_col])
    return df


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2,
               stratify: bool = True, random_state: int = 42):
    """
    Splits a flat feature DataFrame into train/test sets.

    Parameters:
        df (pd.DataFrame): full dataset, e.g. output of fetch_training_data()
                            concatenated into one DataFrame.
        target_col (str): name of the label column to predict. On this
                            project's schema that's "label" (multiclass
                            attack type) for the XGBoost attack-vector
                            classifier. NOTE: if the source view also
                            includes "binary_label" (the other target
                            column from dim_label), it must be dropped from
                            X as well -- it's a second, coarser encoding of
                            the same information as "label" and will leak
                            into a multiclass model exactly like label_id
                            would. Drop it explicitly if you're not using
                            it as a secondary/auxiliary target.
        test_size (float): fraction reserved for testing. Default 0.2 (80/20).
        stratify (bool): if True, preserves class proportions in both splits.
                          Strongly recommended given CICIDS2017's imbalance --
                          without it, a random split could drop a rare attack
                          class (e.g. Infiltration) entirely from the test set.
        random_state (int): seed for reproducibility across team members.

    Returns:
        X_train, X_test, y_train, y_test (all pandas objects)

    Example:
        X_train, X_test, y_train, y_test = split_data(
            df.drop(columns=["binary_label"]), target_col="label"
        )
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    strat_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=strat_arg,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame, columns: list):
    """
    One-hot encodes low-cardinality categorical text columns (e.g.
    "protocol", "port_group", "source_day") using pd.get_dummies(). The
    resulting dummy column SHAPE is effectively "fit" on X_train -- X_test
    is reindexed to match those exact columns, so a category that appears
    only in test (or is missing from test) can't silently create a
    train/test schema mismatch.

    Why one-hot instead of native pandas 'category' dtype + XGBoost's
    enable_categorical: confirmed by direct testing that BorderlineSMOTE/
    SMOTEENN cannot consume 'category' dtype columns (they require numeric
    input; that capability lives in SMOTENC instead). Given cic_typed's
    categorical columns are all low-cardinality (protocol: a handful of
    values, port_group: 4, source_day: 8), one-hot keeps everything numeric
    and fully compatible with apply_smote() below, without introducing a
    different SMOTE variant just for three columns.

    Parameters:
        X_train, X_test (pd.DataFrame): outputs of split_data().
        columns (list[str]): categorical column names to encode, e.g.
                        ["protocol", "port_group", "source_day"].

    Returns:
        X_train_encoded, X_test_encoded, dummy_columns:
            X_train_encoded, X_test_encoded (pd.DataFrame): same row counts
                as input, with `columns` replaced by one-hot dummy columns.
            dummy_columns (list[str]): names of the newly created dummy
                columns -- pass this straight into fit_scaler(exclude=...)
                so the scaler doesn't treat 0/1 dummy columns as continuous.

    Example:
        X_train_enc, X_test_enc, dummy_cols = encode_categoricals(
            X_train, X_test, columns=["protocol", "port_group", "source_day"]
        )
        scaler, X_train_scaled = fit_scaler(
            X_train_enc, exclude=dummy_cols + ["hour_of_day"]
        )
    """
    X_train_encoded = pd.get_dummies(X_train, columns=columns)
    X_test_encoded = pd.get_dummies(X_test, columns=columns)

    # Align test to train's exact dummy column set: fills 0 for any
    # category seen in train but absent from test, and drops any column
    # that appeared only in test (which the model was never trained on).
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    dummy_columns = [c for c in X_train_encoded.columns if c not in X_train.columns]

    return X_train_encoded, X_test_encoded, dummy_columns


def fit_scaler(X_train: pd.DataFrame, columns: Optional[list] = None, exclude: Optional[list] = None):
    """
    Fits a RobustScaler on the TRAINING set only.

    Parameters:
        X_train (pd.DataFrame): training features.
        columns (list[str], optional): specific numeric columns to scale.
                          Defaults to all columns in X_train EXCEPT those
                          listed in `exclude`.
        exclude (list[str], optional): columns to leave untouched -- e.g.
                          categorical/text dimension columns ("protocol",
                          "port_group", "source_day") or datetime columns
                          ("timestamp"). These should NEVER be passed through
                          RobustScaler: it will happily "scale" a categorical
                          code or a datetime as if it were a continuous
                          magnitude, which is silently wrong rather than an
                          error. If `columns` is not given, defaults to
                          excluding these from "all columns" automatically.

    Returns:
        (scaler, X_train_scaled) tuple:
            scaler: the fitted RobustScaler object (reuse via transform_scaler)
            X_train_scaled (pd.DataFrame): scaled training features

    Example (matches the actual vw_ml_training_data schema):
        scaler, X_train_scaled = fit_scaler(
            X_train,
            exclude=["timestamp", "protocol", "port_group", "source_day"]
        )
    """
    if columns:
        cols = columns
    else:
        exclude = exclude or []
        cols = [c for c in X_train.columns if c not in exclude]

    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[cols] = scaler.fit_transform(X_train[cols])

    return scaler, X_train_scaled


def transform_scaler(scaler: RobustScaler, X: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Applies an ALREADY-FITTED scaler to new data (test set, or later, live
    pcap-derived features in Streamlit). Never re-fits.

    Parameters:
        scaler (RobustScaler): the scaler returned by fit_scaler().
        X (pd.DataFrame): features to transform (e.g. X_test).
        columns (list[str], optional): must match the columns used in
                          fit_scaler(). Defaults to all columns in X.

    Returns:
        pd.DataFrame: scaled features, same shape as input.

    Example:
        X_test_scaled = transform_scaler(
            scaler, X_test,
            columns=[c for c in X_test.columns if c not in
                     ["timestamp", "protocol", "port_group", "source_day"]]
        )
    """
    cols = columns if columns else X.columns.tolist()

    X_scaled = X.copy()
    X_scaled[cols] = scaler.transform(X[cols])

    return X_scaled

def impute_missing(X_train: pd.DataFrame, X_test: pd.DataFrame, columns: Optional[list] = None):    
    """
    Fills remaining NaN values with the TRAINING set's median, per column.
    Run this AFTER fit_scaler()/transform_scaler(), BEFORE apply_smote() --
    confirmed by testing against real data that RobustScaler silently
    tolerates NaN (scales around it, leaves it in place), but
    BorderlineSMOTE/SMOTEENN reject it outright with "Input X contains
    NaN". This means NaN can survive scaling undetected and only surface
    as a crash at the SMOTE step, which is exactly what happened here.
 
    Why this exists even though cic_typed's SQL cleaning already handles
    Infinity/NaN in flow_bytes_s/flow_packets_s: that specific fix only
    covers the known division-by-zero case in those two rate columns.
    Other columns can still have NULLs from TRY_CAST failures during
    staging that never got the same treatment. Median (not 0) is used
    here since it's a more defensible default for arbitrary numeric flow
    features than an arbitrary constant -- document whichever choice you
    make in the BRD.
 
    Parameters:
        X_train, X_test (pd.DataFrame): outputs of fit_scaler()/
                        transform_scaler(). Should NOT include one-hot
                        dummy columns unless you're comfortable "median
                        imputing" a 0/1 column (usually fine since dummies
                        rarely have NaN, but worth being aware of).
        columns (list[str], optional): specific columns to impute.
                        Defaults to all numeric columns in X_train.
 
    Returns:
        X_train_imputed, X_test_imputed, fill_values:
            X_train_imputed, X_test_imputed (pd.DataFrame)
            fill_values (dict): {column: median_used} -- log this in your
                        BRD/notebook so the imputation is documented, not
                        silent.
 
    Example:
        X_train_scaled, X_test_scaled, fill_values = impute_missing(
            X_train_scaled, X_test_scaled
        )
        print("Columns imputed:", fill_values)
    """
    cols = columns if columns else X_train.select_dtypes(include="number").columns.tolist()
 
    fill_values = {}
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
 
    for col in cols:
        if X_train_imputed[col].isna().any() or X_test_imputed[col].isna().any():
            median = X_train_imputed[col].median()
            fill_values[col] = median
            X_train_imputed[col] = X_train_imputed[col].fillna(median)
            X_test_imputed[col] = X_test_imputed[col].fillna(median)
 
    return X_train_imputed, X_test_imputed, fill_values



def build_smote_strategy(y_train, min_samples: int = 2000, max_ratio_to_majority: float = 0.5):
    """
    Builds a sampling_strategy dict for apply_smote() from the ACTUAL class
    counts in y_train, rather than hand-picking numbers ahead of time.
    Useful the first time you run against real data, when you don't yet
    know exact class sizes.

    Rule: any class with fewer than `min_samples` gets boosted up to
    `min_samples` -- but never above `max_ratio_to_majority` of the
    majority class's count, so a truly tiny class (e.g. Heartbleed with
    ~11 real-world samples) doesn't get inflated to a target that implies
    far more synthetic diversity than the original data can support.
    Classes already at or above `min_samples` are left untouched (not
    included in the returned dict at all -- BorderlineSMOTE only needs
    entries for classes you want to change).

    Parameters:
        y_train: training labels (text), pre-SMOTE.
        min_samples (int): target floor for small classes. Default 2000.
        max_ratio_to_majority (float): safety cap as a fraction of the
                        majority class's count. Default 0.5 (never
                        oversample a class past half the majority size).

    Returns:
        dict: {class_name: target_count}, ready to pass directly as
        apply_smote()'s sampling_strategy argument.

    Example:
        strategy = build_smote_strategy(y_train, min_samples=2000)
        print(strategy)  # inspect before committing -- see what got boosted
        X_bal, y_bal = apply_smote(X_train_scaled, y_train, sampling_strategy=strategy)
    """
    counts = pd.Series(y_train).value_counts()
    majority_count = counts.max()
    cap = int(majority_count * max_ratio_to_majority)

    strategy = {}
    for cls, count in counts.items():
        if count < min_samples:
            target = min(min_samples, cap)
            if target > count:  # never request fewer samples than already exist
                strategy[cls] = target

    return strategy


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, variant: str = "borderline",
                 sampling_strategy: Union[str, dict, float] = "auto",
                 k_neighbors: int = 5, random_state: int = 42):
    """
    Balances the TRAINING set only. Must be called AFTER fit_scaler(), and
    must NEVER be called on X_test/y_test.

    Parameters:
        X_train (pd.DataFrame): scaled training features.
        y_train (pd.Series): training labels.
        variant (str): "borderline" for BorderlineSMOTE (focuses synthetic
                       samples near the decision boundary -- good default),
                       or "smoteenn" for SMOTEENN (oversamples AND cleans
                       noisy/overlapping points -- more aggressive, slower
                       on 2.8M rows, but often cleaner class separation).
        sampling_strategy: passed directly to the underlying imblearn class.
                       "auto" balances all classes to the majority count --
                       almost certainly NOT what you want given Infiltration's
                       ~30 samples. Prefer passing an explicit dict, e.g.
                       {"Infiltration": 2000, "Heartbleed": 2000}, to cap
                       oversampling of the tiniest classes instead of forcing
                       them to match DDoS/BENIGN counts. Document whatever
                       dict you use in the BRD as a stated limitation.
        k_neighbors (int): neighbors used to generate synthetic samples.
                       Must be less than the smallest class's sample count
                       BEFORE oversampling, or this will error out -- lower
                       it (e.g. to 2 or 3) for extremely small classes.
        random_state (int): seed for reproducibility.

    Returns:
        X_resampled, y_resampled

    Example:
        X_train_bal, y_train_bal = apply_smote(
            X_train_scaled, y_train,
            variant="borderline",
            sampling_strategy={"Infiltration": 2000, "Heartbleed": 2000},
            k_neighbors=3
        )
    """
    if variant == "borderline":
        sampler = BorderlineSMOTE(
            sampling_strategy=cast(str, sampling_strategy),
            k_neighbors=k_neighbors,
            random_state=random_state
        )
    elif variant == "smoteenn":
        sampler = SMOTEENN(
            sampling_strategy=cast(str, sampling_strategy),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown variant '{variant}'. Use 'borderline' or 'smoteenn'.")

    X_resampled, y_resampled = cast(
        "tuple[pd.DataFrame, pd.Series]",
        sampler.fit_resample(X_train, y_train)
    )
    # KNOWN GOTCHA (confirmed via manual testing on this project):
    # BorderlineSMOTE only synthesizes points for minority samples it flags
    # as "in danger" (near the boundary with the majority class). If a class
    # has no danger samples, it silently generates ZERO synthetic points for
    # that class -- no error, no warning. Requesting {"Infiltration": 20} on
    # a 4-sample class produced no change in count with BorderlineSMOTE,
    # while plain SMOTE honored it exactly. Always check post-SMOTE counts
    # against what you requested, especially for tiny classes (Infiltration,
    # Heartbleed, etc.) -- that's what this check below does.
    if isinstance(sampling_strategy, dict):
        actual_counts = pd.Series(y_resampled).value_counts().to_dict()
        for cls, target in sampling_strategy.items():
            actual = actual_counts.get(cls, 0)
            if actual < target:
                print(
                    f"WARNING: requested {target} samples for class '{cls}' "
                    f"but only got {actual} after '{variant}' resampling. "
                    f"If using 'borderline', this likely means no 'danger' "
                    f"samples were found for this class -- consider "
                    f"variant='smoteenn' or plain SMOTE for that class instead."
                )

    return X_resampled, y_resampled


def encode_labels(y_train, y_test):
    """
    Encodes text class labels (e.g. "BENIGN", "DDoS") into the integer
    format XGBoost's sklearn API requires -- confirmed by direct testing
    that passing raw text labels raises:
        ValueError: Invalid classes inferred from unique values of `y`.
    Fit ONLY on y_train, same train/test discipline as the scaler. Run
    this LAST in the pipeline, after apply_smote() -- SMOTE itself works
    fine with text labels, so keeping "label" as text up through the SMOTE
    step preserves human-readable sampling_strategy dicts like
    {"Infiltration": 2000}.

    Parameters:
        y_train, y_test: label Series/arrays. y_train should be the
                        POST-SMOTE resampled labels if SMOTE was applied.
                        Both must only contain classes seen in y_train --
                        guaranteed by split_data()'s stratify=True default,
                        since SMOTE only adds more of existing classes.

    Returns:
        encoder (LabelEncoder): fitted encoder. Save it via save_artifact()
                        -- encoder.classes_ maps predictions back to
                        human-readable labels for the confusion matrix,
                        classification report, and Streamlit output.
        y_train_encoded, y_test_encoded (np.ndarray of ints)

    Example:
        encoder, y_train_enc, y_test_enc = encode_labels(y_train_resampled, y_test)
        save_artifact(encoder, "artifacts/label_encoder.joblib")
        # later, to decode predictions back to attack names:
        predicted_labels = encoder.inverse_transform(y_pred)
    """
    encoder = LabelEncoder()
    y_train_encoded = np.asarray(encoder.fit_transform(y_train))
    y_test_encoded = np.asarray(encoder.transform(y_test))
    return encoder, y_train_encoded, y_test_encoded


def save_artifact(obj, path: str):
    """
    Saves a fitted object (scaler, encoder, etc.) to disk so the exact same
    transform can be reloaded later -- critical for the Streamlit app, which
    must apply the SAME scaler used at training time to live pcap features.

    Parameters:
        obj: any fitted Python object (e.g. a RobustScaler instance).
        path (str): file path to save to, e.g. "artifacts/scaler.joblib".

    Example:
        save_artifact(scaler, "artifacts/scaler.joblib")
    """
    joblib.dump(obj, path)


def load_artifact(path: str):
    """
    Loads a previously saved artifact (e.g. a fitted scaler) back into memory.

    Parameters:
        path (str): file path to load from.

    Returns:
        the deserialized object.

    Example:
        scaler = load_artifact("artifacts/scaler.joblib")
    """
    return joblib.load(path)


if __name__ == "__main__":
    # Full smoke test mimicking the real cic_typed schema shape: numeric
    # flow features + categorical text columns + a raw timestamp + a text
    # label with a small minority class.
    import numpy as np

    df_test = pd.DataFrame({
        "flow_duration": np.random.rand(200) * 1000,
        "flow_bytes_s": np.random.rand(200) * 5000,
        "protocol": np.random.choice(["TCP", "UDP"], 200),
        "port_group": np.random.choice(["web", "db", "mail", "other"], 200),
        "source_day": np.random.choice(["Mon", "Tue", "Wed"], 200),
        "timestamp": pd.date_range("2017-07-03", periods=200, freq="min"),
        "label": ["BENIGN"] * 180 + ["DDoS"] * 15 + ["Infiltration"] * 5
    })

    df_test = engineer_timestamp(df_test)
    X_train, X_test, y_train, y_test = split_data(df_test, target_col="label")

    X_train_enc, X_test_enc, dummy_cols = encode_categoricals(
        X_train, X_test, columns=["protocol", "port_group", "source_day"]
    )

    scaler, X_train_scaled = fit_scaler(X_train_enc, exclude=dummy_cols + ["hour_of_day"])
    X_test_scaled = transform_scaler(scaler, X_test_enc, columns=[c for c in X_test_enc.columns if c not in dummy_cols + ["hour_of_day"]])

    X_bal, y_bal = apply_smote(
        X_train_scaled, y_train,
        variant="borderline",
        sampling_strategy={"Infiltration": 20},
        k_neighbors=2
    )

    encoder, y_train_final, y_test_final = encode_labels(y_bal, y_test)

    print("Train shape before SMOTE:", X_train_scaled.shape)
    print("Train shape after SMOTE:", X_bal.shape)
    print("Encoded label classes:", encoder.classes_)
    print("y_train_final sample:", y_train_final[:10])
    print("Test shape:", X_test_scaled.shape)