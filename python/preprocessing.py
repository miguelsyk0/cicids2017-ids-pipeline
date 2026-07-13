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

def split_data(
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42
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


def fit_scaler(
        X_train: pd.DataFrame,
        columns: list = None,
        exclude: list = None
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

def transform_scaler(
        scaler: RobustScaler,
        X: pd.DataFrame,
        columns: list = None
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

def apply_smote(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        variant: str = "borderline",
        sampling_strategy = "auto",
        k_neighbors: int = 5,
        random_state: int = 42
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

def save_artifact(obj, path: str):

def load_artifact(path: str):

if __name__ == "__main__":
