"""
xgboost_model.py

XGBoost Model class for network intrusion detection (CICIDS2017).
Evaluation metrics are now built directly into this same class.

Running this file directly (python xgboost_model.py) will train the
model, predict, evaluate, and export results for Power BI, all in
one run.
"""

import time
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class XGBoostModel:
    def __init__(self, random_state=42):
        # CHANGE HERE: tune these later based on your results
        # tree_method='hist' is important for a dataset this large,

        self.model = XGBClassifier(
            objective="multi:softmax",
            num_class=15,
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,  # uses all CPU cores available
        )
        self.is_trained = False
        self.model_name = "XGBoost"

    # ============================================
    # TRAINING / PREDICTION
    # ============================================
    def train(self, X_train, y_train):
        """
        X_train, y_train: output from your teammate's SMOTE class
        (training data only, already balanced, NOT PCA transformed)
        """
        print(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features...")
        start = time.time()

        self.model.fit(X_train, y_train)

        elapsed = time.time() - start
        print(f"Training complete in {elapsed:.2f} seconds.")
        self.is_trained = True

    def predict(self, X_test):
        if not self.is_trained:
            raise Exception("Model is not trained yet. Call train() first.")
        return self.model.predict(X_test)

    def get_feature_importance(self, feature_names=None):
        """
        Returns feature importance as a DataFrame. Since PCA is NOT
        used here, this ties directly to the ORIGINAL feature names
        (e.g. Flow Duration, Packet Length), useful for answering
        which traffic characteristics matter most for detection.
        """
        importances = self.model.feature_importances_

        # CHANGE HERE: pass in real column names,
        # e.g. X_train.columns.tolist()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values(by="Importance", ascending=False)

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
    # EVALUATION METRICS
    # ============================================
    def evaluate(self, y_true, y_pred):
        """
        Core metrics. 'weighted' average is used since this is a
        multiclass problem (Benign + multiple attack types) with
        imbalanced classes.
        """
        metrics = {
            "Model": self.model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted"),
            "Recall": recall_score(y_true, y_pred, average="weighted"),
            "F1_Score": f1_score(y_true, y_pred, average="weighted"),
        }
        return metrics

    def get_confusion_matrix(self, y_true, y_pred, class_labels=None):
        """
        Returns confusion matrix as a labeled DataFrame, ready to
        export as CSV for Power BI (heatmap visual).
        """
        cm = confusion_matrix(y_true, y_pred)

        # CHANGE HERE: pass actual class names
        # e.g. label_encoder.classes_.tolist()
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
        report = classification_report(
            y_true, y_pred, target_names=class_labels, output_dict=True
        )
        return pd.DataFrame(report).transpose()

    def export_for_powerbi(
        self,
        metrics_dict,
        cm_df,
        report_df,
        metrics_path="xgboost_metrics_summary.csv",
        cm_path="xgboost_confusion_matrix.csv",
        report_path="xgboost_classification_report.csv",
    ):
        """
        Exports all evaluation outputs as CSV, ready to load into
        Power BI for the Model Performance dashboard page.
        """
        pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)
        cm_df.to_csv(cm_path)
        report_df.to_csv(report_path)

        print(f"Exported: {metrics_path}")
        print(f"Exported: {cm_path}")
        print(f"Exported: {report_path}")


# ============================================
# INDEPENDENT MAIN RUN
# Running this file directly trains AND evaluates XGBoost only.
# ============================================
if __name__ == "__main__":

    # ============================================
    # CHANGE HERE: load your real, final dataset
    # (output of SQL cleaning + your teammate's SMOTE class)
    # ============================================
    # Example:
    # X_train = pd.read_csv("X_train_smote.csv")
    # y_train = pd.read_csv("y_train_smote.csv").values.ravel()
    # X_test = pd.read_csv("X_test.csv")
    # y_test = pd.read_csv("y_test.csv").values.ravel()

    X_train, y_train, X_test, y_test = (
        None,
        None,
        None,
        None,
    )  # placeholders, remove once real data is loaded

    if X_train is None:
        raise Exception(
            "No data loaded yet. Replace the placeholder section above "
            "with your real X_train, y_train, X_test, y_test once the "
            "SMOTE step and final dataset are ready."
        )

    # CHANGE HERE: pass real feature names and class labels
    feature_names = None  # e.g. X_train.columns.tolist()
    class_labels = None  # e.g. label_encoder.classes_.tolist()

    # Reference only, the 15 CICIDS2017 labels (Benign + 14 attack
    # types). Your actual class_labels list above should match
    # whatever your LabelEncoder produced, in the SAME order, since
    # encoders sort alphabetically by default, not in this order:
    # ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk',
    #  'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator',
    #  'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator',
    #  'Web Attack - Brute Force', 'Web Attack - Sql Injection',
    #  'Web Attack - XSS']

    # --- Train ---
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train)
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
