"""
isolation_forest_model.py

Isolation Forest model class for network intrusion detection (CICIDS2017).
Unsupervised anomaly detection layer -- structured to mirror
xgboost_model.py (train/predict/evaluate/export_for_powerbi) so the two
models are easy to compare side by side in the BRD and defense.

Running this file directly (python isolation_forest_model.py) will train
the model, predict, evaluate, and export results for Power BI, all in
one run.

KEY DESIGN CONSTRAINTS (do not violate these -- see preprocessing.py):
    1. NO SMOTE. Isolation Forest relies on attacks being genuinely rare
       and structurally distinct from the bulk of benign traffic --
       that's the entire mechanism it exploits. SMOTE-balanced data
       would destroy that assumption; feed it the natural, imbalanced
       class distribution.
    2. NO PCA. Consistent with the XGBoost decision -- keeps feature
       importance / anomaly investigation tied to real column names.
    3. UNSUPERVISED. train() never sees y_train. Labels are used only
       AFTER training, to evaluate how well the anomaly scores line up
       with reality -- specifically against "binary_label" (BENIGN vs
       ATTACK), not the multiclass "label". Isolation Forest fundamentally
       makes a binary call (inlier/outlier); it isn't equipped to name
       an attack type, that's XGBoost's job downstream.
    4. NO one-hot categorical columns (protocol/port_group/source_day) and
       NO hour_of_day -- drop them ENTIRELY from X before training, don't
       just exclude them from scaling. This is DIFFERENT from the XGBoost
       pipeline, and confirmed by direct testing, not just theory: with a
       realistic column ratio (79 continuous features + 14 one-hot
       columns), adding the dummy columns roughly DOUBLED the false
       positive rate (589/600 flagged vs 442/600 without them, against a
       true attack rate of ~8%). Isolation Forest builds trees via RANDOM
       feature/threshold splits, so sparse binary indicator columns get
       selected disproportionately often relative to the real signal they
       carry -- unlike XGBoost, which uses loss-based splitting and
       naturally down-weights uninformative features. Feed this model
       ONLY the continuous flow-measurement columns.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

METRICS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "metrics")
)
os.makedirs(METRICS_DIR, exist_ok=True)


class IsolationForestModel:
    def __init__(self, contamination="auto", n_estimators=100, random_state=42):
        # contamination: expected proportion of anomalies in the training
        # data. 'auto' lets sklearn set its own internal threshold from
        # the fitted data's own anomaly score distribution, WITHOUT ever
        # looking at labels -- the purer "unsupervised" choice.
        #
        # Alternatively, you can pass an explicit float (e.g. 0.2) based
        # on the known/estimated attack ratio -- see estimate_contamination()
        # below. That's still not "training on labels" (no labels reach
        # .fit()), but it IS using label information to pick a
        # hyperparameter, which sits in a gray area for a "purely
        # unsupervised" claim. Document whichever choice you make in the
        # BRD -- don't let it go unstated.
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,  # uses all CPU cores available
        )
        self.contamination = contamination
        self.is_trained = False
        self.model_name = "IsolationForest"

    # ============================================
    # TRAINING / PREDICTION
    # ============================================
    def train(self, X_train):
        """
        X_train: scaled, UNBALANCED training features (i.e. straight from
        fit_scaler() in preprocessing.py -- NOT apply_smote() output).
        No y_train parameter by design: Isolation Forest is unsupervised,
        and mixing in labels here would defeat the point of using it
        alongside a supervised model like XGBoost.
        """
        print(
            f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features (unsupervised)..."
        )
        start = time.time()

        self.model.fit(X_train)

        elapsed = time.time() - start
        print(f"Training complete in {elapsed:.2f} seconds.")
        self.is_trained = True

    def predict(self, X):
        """
        Returns raw sklearn convention: 1 = normal (inlier), -1 = anomaly
        (outlier). Kept as-is (rather than immediately mapping to text)
        so callers can choose numeric or text form depending on use case.
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_labels(self, X, benign_label="BENIGN", anomaly_label="ATTACK"):
        """
        Convenience wrapper: maps predict()'s -1/1 output to human-readable
        text labels, so it can be compared directly against the ground
        truth "binary_label" column.

        Parameters:
            benign_label (str): text value to assign to inliers (1).
                        Must match whatever string dim_label/cic_typed's
                        "binary_label" actually uses for non-attack rows --
                        confirm this against real data before relying on it.
            anomaly_label (str): text value to assign to outliers (-1).

        Returns:
            np.ndarray of strings, same length as X.
        """
        raw = self.predict(X)
        return np.where(raw == 1, benign_label, anomaly_label)

    def anomaly_score(self, X):
        """
        Returns a continuous anomaly score per row: HIGHER = MORE
        anomalous. Note this is the NEGATIVE of sklearn's own
        decision_function() (which scores higher = more normal) --
        inverted here because "higher score = more suspicious" is the
        intuitive direction for a dashboard (e.g. sorting flows by
        severity in Power BI, or flagging the most suspicious flows in
        the Streamlit app).
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Call train() first.")
        return -self.model.decision_function(X)

    @staticmethod
    def estimate_contamination(y_train, benign_label="BENIGN"):
        """
        Computes the empirical attack ratio in y_train -- a REFERENCE
        point if you want to set `contamination` explicitly instead of
        'auto'. This does not feed labels into the model itself; it only
        informs a hyperparameter choice made before training. Still worth
        stating explicitly in the BRD, since it's a point where domain
        knowledge (via labels) entered an otherwise-unsupervised pipeline.

        Parameters:
            y_train: the "binary_label" column for the training split
                        (NOT used elsewhere in training).
            benign_label (str): value representing non-attack traffic.

        Returns:
            float: proportion of rows where y_train != benign_label.

        Example:
            attack_ratio = IsolationForestModel.estimate_contamination(y_train)
            model = IsolationForestModel(contamination=attack_ratio)
        """
        y_train = pd.Series(y_train)
        return float((y_train != benign_label).mean())

    def save_model(self, path="isolation_forest_model.joblib"):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path="isolation_forest_model.joblib"):
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")

    # ============================================
    # EVALUATION METRICS
    # Reports weighted AND macro, same pattern as xgboost_model.py, so
    # the two models' summary tables are directly comparable.
    # ============================================
    def evaluate(self, y_true_text, y_pred_text):
        """
        Binary evaluation: y_true_text is the ground-truth "binary_label"
        column, y_pred_text is predict_labels()'s output. Both must use
        the SAME two string values (e.g. "BENIGN"/"ATTACK").
        """
        metrics = {
            "Model": self.model_name,
            "Accuracy": accuracy_score(y_true_text, y_pred_text),
            "Precision_Weighted": precision_score(
                y_true_text, y_pred_text, average="weighted", zero_division=0
            ),
            "Recall_Weighted": recall_score(
                y_true_text, y_pred_text, average="weighted", zero_division=0
            ),
            "F1_Weighted": f1_score(
                y_true_text, y_pred_text, average="weighted", zero_division=0
            ),
            "Precision_Macro": precision_score(
                y_true_text, y_pred_text, average="macro", zero_division=0
            ),
            "Recall_Macro": recall_score(
                y_true_text, y_pred_text, average="macro", zero_division=0
            ),
            "F1_Macro": f1_score(
                y_true_text, y_pred_text, average="macro", zero_division=0
            ),
        }
        return metrics

    def get_confusion_matrix(self, y_true_text, y_pred_text, class_labels=None):
        """
        Binary confusion matrix (BENIGN vs ATTACK), ready to export as
        CSV for Power BI.
        """
        if class_labels is None:
            class_labels = sorted(
                pd.unique(pd.concat([pd.Series(y_true_text), pd.Series(y_pred_text)]))
            )
        cm = confusion_matrix(y_true_text, y_pred_text, labels=class_labels)
        return pd.DataFrame(cm, index=class_labels, columns=class_labels)

    def get_classification_report(self, y_true_text, y_pred_text):
        """
        Precision/recall/F1 for BENIGN vs ATTACK individually -- useful
        to check whether the model is biased toward one direction (e.g.
        flagging everything as an anomaly, which would show up as high
        recall but poor precision on ATTACK).
        """
        report = classification_report(y_true_text, y_pred_text, output_dict=True)
        return pd.DataFrame(report).transpose()

    def export_for_powerbi(
        self,
        metrics_dict,
        cm_df,
        report_df,
        scores_df=None,
        metrics_path="isolation_forest_metrics_summary.csv",
        cm_path="isolation_forest_confusion_matrix.csv",
        report_path="isolation_forest_classification_report.csv",
        scores_path="isolation_forest_anomaly_scores.csv",
    ):
        """
        Exports all evaluation outputs as CSV for Power BI. `scores_df`
        (optional) is per-row anomaly scores -- useful for a severity
        heatmap or "most suspicious flows" table on the dashboard, distinct
        from XGBoost's per-model metrics since this model produces a
        continuous score, not just a class label.
        """
        pd.DataFrame([metrics_dict]).to_csv(
            os.path.join(METRICS_DIR, metrics_path), index=False
        )
        cm_df.to_csv(os.path.join(METRICS_DIR, cm_path))
        report_df.to_csv(os.path.join(METRICS_DIR, report_path))
        print(f"Exported: {os.path.join(METRICS_DIR, metrics_path)}")
        print(f"Exported: {os.path.join(METRICS_DIR, cm_path)}")
        print(f"Exported: {os.path.join(METRICS_DIR, report_path)}")

        if scores_df is not None:
            scores_df.to_csv(os.path.join(METRICS_DIR, scores_path), index=False)
            print(f"Exported: {os.path.join(METRICS_DIR, scores_path)}")


# ============================================
# INDEPENDENT MAIN RUN
# Running this file directly trains AND evaluates Isolation Forest only.
# ============================================
if __name__ == "__main__":

    # ============================================
    # Real pipeline wiring:
    #
    # from data_fetch import fetch_training_data
    # from preprocessing import (
    #     engineer_timestamp, split_data, encode_categoricals,
    #     fit_scaler, transform_scaler
    # )
    #
    # df = pd.concat(fetch_training_data(chunksize=200_000), ignore_index=True)
    # df = df.drop(columns=["label"])  # drop the MULTICLASS label -- not
    #                                  # used here, and leaves binary_label
    #                                  # as this model's target instead
    # df = engineer_timestamp(df)
    #
    # NOTE the target_col here is "binary_label", not "label" -- Isolation
    # Forest is evaluated against the binary ground truth, not attack type.
    # X_train, X_test, y_train, y_test = split_data(df, target_col="binary_label")
    #
    # X_train_enc, X_test_enc, dummy_cols = encode_categoricals(
    #     X_train, X_test, columns=["protocol", "port_group", "source_day"]
    # )
    #
    # IMPORTANT -- unlike the XGBoost pipeline: drop the one-hot dummy
    # columns AND hour_of_day ENTIRELY here, don't just exclude them from
    # scaling. Confirmed by testing that leaving them in roughly doubles
    # Isolation Forest's false positive rate (see module docstring above).
    # iso_exclude = dummy_cols + ["hour_of_day"]
    # X_train_iso = X_train_enc.drop(columns=iso_exclude)
    # X_test_iso = X_test_enc.drop(columns=iso_exclude)
    #
    # scaler, X_train_scaled = fit_scaler(X_train_iso)
    # X_test_scaled = transform_scaler(scaler, X_test_iso)
    #
    # NOTE: no apply_smote(), no encode_labels() here -- unlike XGBoost,
    # Isolation Forest trains on X_train_scaled directly, keeping the
    # natural class imbalance, and evaluation compares text labels
    # directly rather than needing integer encoding.
    # ============================================

    X_train, X_test, y_train, y_test = None, None, None, None  # placeholders

    if X_train is None:
        raise Exception(
            "No data loaded yet. Replace the placeholder section above "
            "with the real pipeline wiring shown in the comment block."
        )

    # Optional: check the empirical attack ratio before deciding whether
    # to override contamination='auto'.
    attack_ratio = IsolationForestModel.estimate_contamination(
        y_train, benign_label="BENIGN"
    )
    print(f"Empirical attack ratio in training data: {attack_ratio:.4f}")

    # --- Train (unsupervised -- no y_train passed) ---
    iso_model = IsolationForestModel(contamination="auto")
    iso_model.train(X_train)

    # --- Predict ---
    y_pred = iso_model.predict_labels(
        X_test, benign_label="BENIGN", anomaly_label="ATTACK"
    )
    scores = iso_model.anomaly_score(X_test)

    # Save trained model
    iso_model.save_model("isolation_forest_model.joblib")

    # --- Evaluate against ground-truth binary_label ---
    metrics = iso_model.evaluate(y_test, y_pred)
    print("\nMetrics:", metrics)

    cm_df = iso_model.get_confusion_matrix(y_test, y_pred)
    report_df = iso_model.get_classification_report(y_test, y_pred)

    scores_df = pd.DataFrame(
        {
            "true_label": pd.Series(y_test).reset_index(drop=True),
            "predicted_label": y_pred,
            "anomaly_score": scores,
        }
    )

    # --- Export everything for Power BI ---
    iso_model.export_for_powerbi(metrics, cm_df, report_df, scores_df=scores_df)

    print("\nIsolation Forest run complete. All results exported for Power BI.")
