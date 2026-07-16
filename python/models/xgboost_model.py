"""
xgboost_model.py

XGBoost Model class for network intrusion detection (CICIDS2017).
Evaluation metrics are now built directly into this same class.

Running this file directly (python xgboost_model.py) will train the
model, predict, evaluate, and export results for Power BI, all in
one run.
"""

import os
import time
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

METRICS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "metrics")
)
os.makedirs(METRICS_DIR, exist_ok=True)


class XGBoostModel:
    def __init__(self, num_class=None, random_state=42):
        # CHANGE HERE: tune these later based on your results
        # tree_method='hist' is important for a dataset this large,
        # it is much faster than the default method on 1M+ rows
        #
        # This is set up for MULTICLASS classification (15 labels:
        # Benign + 14 attack types), not multi-label. Each row gets
        # exactly ONE predicted label out of the 15.
        #
        # num_class is no longer hardcoded -- if not passed explicitly,
        # it's derived from the data at train() time via
        # len(np.unique(y_train)). This avoids a silent mismatch if the
        # team later groups or drops rare attack types during cleaning.
        # Passing it explicitly still works if you want to fix it ahead
        # of time (e.g. num_class=15 to match dim_label's 15 rows).
        self.num_class = num_class
        self.model = None  # built in train(), once num_class is known
        self.random_state = random_state
        self.is_trained = False
        self.model_name = "XGBoost"
        self.class_labels_ = None  # populated in train() if provided

    def _build_model(self, num_class):
        return XGBClassifier(
            # multi:softprob instead of multi:softmax: returns the same
            # .predict() class output, but also enables .predict_proba(),
            # giving a confidence score per prediction. For an IDS feeding
            # a Streamlit dashboard, "flagged as DDoS, 94% confidence" is
            # meaningfully more useful than a bare label, at no extra cost.
            objective='multi:softprob',
            num_class=num_class,
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            tree_method='hist',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1  # uses all CPU cores available
        )

    # ============================================
    # TRAINING / PREDICTION
    # ============================================
    def train(self, X_train, y_train, class_labels=None):
        """
        X_train, y_train: output from preprocessing.py's full pipeline --
        split -> encode_categoricals -> fit_scaler -> apply_smote ->
        encode_labels(). y_train must already be INTEGER-encoded (XGBoost's
        sklearn API rejects text labels -- confirmed via testing).

        class_labels (optional): pass encoder.classes_ from
        preprocessing.encode_labels() here to store the human-readable
        label order alongside the model, so get_confusion_matrix() and
        get_classification_report() can use it automatically without
        having to pass it again at evaluation time.
        """
        num_class = self.num_class or len(pd.unique(y_train))
        self.num_class = num_class
        self.model = self._build_model(num_class)
        if class_labels is not None:
            self.class_labels_ = list(class_labels)

        print(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features, {num_class} classes...")
        start = time.time()

        self.model.fit(X_train, y_train)

        elapsed = time.time() - start
        print(f"Training complete in {elapsed:.2f} seconds.")
        self.is_trained = True

    def predict(self, X_test):
        if self.model is None:
            raise Exception("Model is not trained yet. Call train() first.")
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Returns per-class confidence scores (rows sum to 1.0). Available
        now that the model uses 'multi:softprob'. Useful for the Streamlit
        app -- e.g. showing "94% confidence" alongside the predicted
        attack vector, or flagging low-confidence predictions for review.
        """
        if self.model is None:
            raise Exception("Model is not trained yet. Call train() first.")
        return self.model.predict_proba(X_test)

    def get_feature_importance(self, feature_names=None):
        """
        Returns feature importance as a DataFrame. Since PCA is NOT
        used here, this ties directly to the ORIGINAL feature names
        (e.g. Flow Duration, Packet Length), useful for answering
        which traffic characteristics matter most for detection.
        """
        if self.model is None:
            raise Exception("Model is not trained yet. Call train() first.")
        importances = self.model.feature_importances_

        # CHANGE HERE: pass in real column names,
        # e.g. X_train.columns.tolist()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        return importance_df

    def save_model(self, path="xgboost_model.joblib"):
        # Saves the trained model so you don't need to retrain
        # every time you rerun the notebook, useful since training
        # on 1M+ rows takes real time
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path="xgboost_model.joblib"):
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")

    # ============================================
    # EVALUATION METRICS (merged into this class)
    # ============================================
    def evaluate(self, y_true, y_pred):
        """
        Reports both 'weighted' and 'macro' averages. 'weighted' reflects
        overall performance proportional to class frequency; 'macro'
        weighs every class equally regardless of size. Both matter here:
        a model can post a great weighted F1 (dominated by BENIGN/DDoS)
        while quietly failing on Infiltration/Heartbleed -- macro surfaces
        that. get_classification_report() below already gives full
        per-class detail; these are the two top-line summaries worth
        reporting together rather than weighted alone.
        """
        metrics = {
            'Model': self.model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision_Weighted': precision_score(y_true, y_pred, average='weighted'),
            'Recall_Weighted': recall_score(y_true, y_pred, average='weighted'),
            'F1_Weighted': f1_score(y_true, y_pred, average='weighted'),
            'Precision_Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall_Macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'F1_Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        return metrics

    def get_confusion_matrix(self, y_true, y_pred, class_labels=None):
        """
        Returns confusion matrix as a labeled DataFrame, ready to
        export as CSV for Power BI (heatmap visual).
        """
        cm = confusion_matrix(y_true, y_pred)

        # Falls back to the labels stored at train() time (from
        # preprocessing.encode_labels()'s encoder.classes_), if any.
        if class_labels is None:
            class_labels = self.class_labels_
        if class_labels is None:
            class_labels = [f"Class_{i}" for i in range(cm.shape[0])]

        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        return cm_df

    def get_classification_report(self, y_true, y_pred, class_labels=None):
        """
        Per-class precision/recall/F1, useful for spotting weak
        performance on rare attack types (e.g. Infiltration) even
        if overall metrics look good.
        """
        if class_labels is None:
            class_labels = self.class_labels_
        report = classification_report(
            y_true, y_pred,
            target_names=class_labels,
            output_dict=True
        )
        return pd.DataFrame(report).transpose()

    def export_for_powerbi(self, metrics_dict, cm_df, report_df,
                            metrics_path="xgboost_metrics_summary.csv",
                            cm_path="xgboost_confusion_matrix.csv",
                            report_path="xgboost_classification_report.csv"):
        """
        Exports all evaluation outputs as CSV, ready to load into
        Power BI for the Model Performance dashboard page.
        """
        pd.DataFrame([metrics_dict]).to_csv(
            os.path.join(METRICS_DIR, metrics_path), index=False
        )
        cm_df.to_csv(os.path.join(METRICS_DIR, cm_path))
        report_df.to_csv(os.path.join(METRICS_DIR, report_path))

        print(f"Exported: {os.path.join(METRICS_DIR, metrics_path)}")
        print(f"Exported: {os.path.join(METRICS_DIR, cm_path)}")
        print(f"Exported: {os.path.join(METRICS_DIR, report_path)}")


# ============================================
# INDEPENDENT MAIN RUN
# Running this file directly trains AND evaluates XGBoost only.
# ============================================
if __name__ == "__main__":

    # ============================================
    # Real pipeline wiring (replaces the old placeholder section):
    #
    # from data_fetch import fetch_training_data
    # from preprocessing import (
    #     engineer_timestamp, split_data, encode_categoricals,
    #     fit_scaler, transform_scaler, apply_smote, encode_labels,
    #     save_artifact
    # )
    #
    # df = pd.concat(fetch_training_data(chunksize=200_000), ignore_index=True)
    # df = df.drop(columns=["binary_label"])  # drop unless used as a
    #                                          # secondary target -- keeping
    #                                          # it in X alongside "label"
    #                                          # would leak the target
    # df = engineer_timestamp(df)
    # X_train, X_test, y_train, y_test = split_data(df, target_col="label")
    # X_train_enc, X_test_enc, dummy_cols = encode_categoricals(
    #     X_train, X_test, columns=["protocol", "port_group", "source_day"]
    # )
    # scaler, X_train_scaled = fit_scaler(X_train_enc, exclude=dummy_cols + ["hour_of_day"])
    # X_test_scaled = transform_scaler(scaler, X_test_enc,
    #     columns=[c for c in X_test_enc.columns if c not in dummy_cols + ["hour_of_day"]])
    # X_train_bal, y_train_bal = apply_smote(
    #     X_train_scaled, y_train, variant="borderline",
    #     sampling_strategy={"Infiltration": 2000, "Heartbleed": 2000}, k_neighbors=3
    # )
    # encoder, y_train, y_test = encode_labels(y_train_bal, y_test)
    # save_artifact(encoder, "artifacts/label_encoder.joblib")
    # save_artifact(scaler, "artifacts/scaler.joblib")
    # X_train, X_test = X_train_bal, X_test_scaled
    # class_labels = encoder.classes_.tolist()
    # ============================================

    X_train, y_train, X_test, y_test = None, None, None, None  # placeholders, remove once real data is loaded
    class_labels = None  # e.g. encoder.classes_.tolist(), set above

    if X_train is None:
        raise Exception(
            "No data loaded yet. Replace the placeholder section above "
            "with the real pipeline wiring shown in the comment block, "
            "once fetch_training_data() and preprocessing.py are hooked up."
        )

    feature_names = None   # e.g. X_train.columns.tolist()

    # Reference only, the 15 CICIDS2017 labels (Benign + 14 attack
    # types), in encoder.classes_ order (alphabetical, NOT this order):
    # ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk',
    #  'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator',
    #  'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator',
    #  'Web Attack - Brute Force', 'Web Attack - Sql Injection',
    #  'Web Attack - XSS']

    # --- Train ---
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train, class_labels=class_labels)
    y_pred = xgb_model.predict(X_test)

    # Save trained model so you don't retrain every run
    xgb_model.save_model("xgboost_model.joblib")

    # --- Feature importance ---
    importance_df = xgb_model.get_feature_importance(feature_names)
    print("\nTop 10 important features:")
    print(importance_df.head(10))
    importance_df.to_csv("xgboost_feature_importance.csv", index=False)

    # --- Evaluate (same class now, no separate import needed) ---
    metrics = xgb_model.evaluate(y_test, y_pred)
    print("\nMetrics:", metrics)

    cm_df = xgb_model.get_confusion_matrix(y_test, y_pred, class_labels)
    report_df = xgb_model.get_classification_report(y_test, y_pred, class_labels)

    # --- Export everything for Power BI ---
    xgb_model.export_for_powerbi(metrics, cm_df, report_df)

    print("\nXGBoost run complete. All results exported for Power BI.")