"""
preprocessing.py
----------------
Reusable preprocessing module for the ITE 17 IDS project.

Covers the three steps that sit between "flat data out of SQL" and
"model.fit()": splitting, scaling, and class balancing. Built to be shared
by XGBoost and any baseline (e.g. Random Forest) so both use IDENTICAL
preprocessing -- if they don't, comparing their scores is not valid.

Design decisions this file enforces (do not skip these when using it):
  1. Split happens BEFORE scaling and BEFORE SMOTE. Fitting a scaler or
     SMOTE on the full dataset before splitting leaks test-set
     information into training -- your reported metrics would be
     inflated and wouldn't survive a panel member asking "did you split
     first?"
  2. The scaler is fit ONLY on X_train, then used to .transform() (not
     .fit_transform()) X_test. Same reasoning as above.
  3. SMOTE is applied ONLY to the training set, AFTER scaling, and is
     never applied to X_test / y_test. Test data must reflect real-world
     class imbalance to mean anything.
  4. Isolation Forest and the rule-based layer should NOT use the SMOTE
     output of this file -- only XGBoost (and baseline comparisons like
     Random Forest) should. Isolation Forest wants scaled-but-unbalanced
     data; the rule-based layer wants raw, unscaled, named features.

Libraries used:
  - pandas               -> DataFrame handling
  - sklearn.model_selection.train_test_split -> stratified train/test split
  - sklearn.preprocessing.RobustScaler -> scaling resistant to outliers
                            (chosen over StandardScaler because network
                            flow features like Flow Bytes/s have extreme
                            outliers even after cleaning Infinity/NaN)
  - imblearn              -> BorderlineSMOTE / SMOTEENN for class balancing,
                            chosen over vanilla SMOTE because CICIDS2017
                            has attack classes as small as ~30 samples
                            (e.g. Infiltration); vanilla SMOTE would
                            interpolate near-duplicate synthetic points
                            from such a small pool.
  - joblib               -> saves/loads fitted scaler objects so the exact
                            same transform can be reused later by the
                            Streamlit app on live pcap-derived features.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN
from typing import Any, Optional, Union, cast


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
):
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
        X, y, test_size=test_size, stratify=strat_arg, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def fit_scaler(
    X_train: pd.DataFrame,
    columns: Optional[list] = None,
    exclude: Optional[list] = None,
):
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


def transform_scaler(
    scaler: RobustScaler, X: pd.DataFrame, columns: Optional[list] = None
) -> pd.DataFrame:
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


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    variant: str = "borderline",
    sampling_strategy: Union[str, dict] = "auto",
    k_neighbors: int = 5,
    random_state: int = 42,
):
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

             returns:        X_resampled, y_resampled

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
            sampling_strategy=cast(Any, sampling_strategy),
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
    elif variant == "smoteenn":
        sampler = SMOTEENN(
            sampling_strategy=cast(Any, sampling_strategy), random_state=random_state
        )
    else:
        raise ValueError(f"Unknown variant '{variant}'. Use 'borderline' or 'smoteenn'")

    # Some samplers may return more than two objects (e.g., additional
    # indices). To avoid tuple size mismatch errors, capture the result
    # and explicitly take the first two elements as X and y.
    _res = sampler.fit_resample(X_train, y_train)
    if isinstance(_res, tuple) or isinstance(_res, list):
        if len(_res) >= 2:
            X_resampled, y_resampled = _res[0], _res[1]
        else:
            raise ValueError("resampler returned unexpected number of values")
    else:
        # Fallback: assume single object with attributes
        try:
            X_resampled, y_resampled = _res.X_resampled, _res.y_resampled
        except Exception:
            raise ValueError("Unable to unpack resampler result")

    if isinstance(sampling_strategy, dict):
        if isinstance(y_resampled, pd.DataFrame):
            y_series = y_resampled.iloc[:, 0]
        else:
            y_series = pd.Series(y_resampled, dtype="object")
        actual_counts = y_series.value_counts().to_dict()
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
    import numpy as np

    df_test = pd.DataFrame(
        {
            "feat_1": np.random.rand(200),
            "feat_2": np.random.rand(200) * 100,
            "Label": ["BENIGN"] * 180 + ["DDoS"] * 15 + ["Infiltration"] * 5,
        }
    )

    X_train, X_test, y_train, y_test = split_data(df_test, target_col="Label")
    scaler, X_train_scaled = fit_scaler(X_train)
    X_test_scaled = transform_scaler(scaler, X_test)

    X_bal, y_bal = apply_smote(
        X_train_scaled,
        y_train,
        variant="borderline",
        sampling_strategy={"Infiltration": 20},
        k_neighbors=2,
    )

    print("Train shape before SMOTE:", X_train_scaled.shape)
    print("Train shape after SMOTE:", X_bal.shape)
    print(y_bal.value_counts())
