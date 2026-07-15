"""
rule_based_model.py

Rule-Based Signature Engine (Heuristic Baseline)
Network Intrusion Detection - CICIDS2017 Dataset

This is NOT a machine learning model. It applies hardcoded
if-else rules (thresholds) based on known attack behavior, similar
to how traditional signature-based intrusion detection systems
(like Snort) work. It exists as a BASELINE to compare against your
ML models (Logistic Regression, Decision Tree, Random Forest,
XGBoost). If XGBoost barely beats this simple rule engine, that's
an important finding for your evaluation section.

Evaluation metrics are built directly into this same class,
matching the structure of xgboost_model.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class RuleBasedModel:
    def __init__(self):
        self.model_name = "Rule-Based Signature Engine"
        self.is_fitted = False

        # ============================================
        # CHANGE HERE: these thresholds are placeholder starting
        # points based on general known attack behavior. You MUST
        # adjust these once you see your actual data's distributions
        # (check df.describe() or percentiles per attack type during
        # your EDA phase). Real signature systems tune these from
        # real traffic statistics, not guesses.
        # ============================================
        self.thresholds = {
            # PortScan: very short flows, very few packets sent,
            # often single packet probes
            "portscan_duration_max": 100,  # microseconds
            "portscan_fwd_packets_max": 2,
            # DoS/DDoS: abnormally high packet rate or byte rate
            "dos_flow_packets_per_sec_min": 5000,
            "dos_flow_bytes_per_sec_min": 1000000,
            # Brute Force (FTP-Patator / SSH-Patator): targets
            # specific well-known ports
            "ftp_port": 21,
            "ssh_port": 22,
            "bruteforce_duration_min": 1000000,  # long repeated attempts
            # Web Attacks (Brute Force / XSS / SQL Injection):
            # target web ports
            "web_ports": [80, 443],
            # Heartbleed: targets HTTPS port with unusually large
            # flow byte count in a single flow
            "heartbleed_port": 443,
            "heartbleed_bytes_min": 100000,
            # Bot: long-lived, low-volume periodic connections
            "bot_duration_min": 5000000,
            "bot_fwd_packets_max": 50,
        }

        # ============================================
        # CHANGE HERE: this list must match your actual LabelEncoder
        # class order exactly (print label_encoder.classes_ to check),
        # since predictions are returned as encoded numbers based on
        # this list's index positions
        # ============================================
        self.class_labels = [
            "BENIGN",
            "Bot",
            "DDoS",
            "DoS GoldenEye",
            "DoS Hulk",
            "DoS Slowhttptest",
            "DoS slowloris",
            "FTP-Patator",
            "Heartbleed",
            "Infiltration",
            "PortScan",
            "SSH-Patator",
            "Web Attack - Brute Force",
            "Web Attack - Sql Injection",
            "Web Attack - XSS",
        ]

    # ============================================
    # OPTIONAL: auto-tune thresholds from training data
    # ============================================
    def fit(self, X_train, y_train=None):
        """
        Not 'training' in the ML sense, this just lets you optionally
        auto-calculate some thresholds from real data percentiles
        instead of hardcoded guesses. Safe to skip and rely on the
        manual thresholds above if you prefer full manual control.

        CHANGE HERE: column names below (e.g. 'Flow Duration') must
        match your actual cleaned dataset's column names exactly.
        """
        try:
            self.thresholds["dos_flow_packets_per_sec_min"] = X_train[
                "Flow Packets/s"
            ].quantile(0.95)
            self.thresholds["portscan_duration_max"] = X_train[
                "Flow Duration"
            ].quantile(0.05)
            print("Thresholds auto-tuned from training data percentiles.")
        except KeyError as e:
            print(f"Column not found ({e}), keeping manual default thresholds.")

        self.is_fitted = True

    # ============================================
    # PREDICTION (the actual rule logic)
    # ============================================
    def predict(self, X_test):
        """
        Applies rules row by row. Returns predictions as ENCODED
        integers matching self.class_labels order, so it plugs into
        the same evaluation functions as your XGBoost model.

        CHANGE HERE: column names (e.g. 'Destination Port',
        'Flow Duration') must match your actual cleaned dataset's
        column names exactly. Adjust or add rules based on which
        features your final dataset actually keeps.
        """
        predictions = []
        t = self.thresholds

        for _, row in X_test.iterrows():
            label = "BENIGN"  # default assumption if no rule matches

            duration = row.get("Flow Duration", 0)
            fwd_packets = row.get("Total Fwd Packets", 0)
            flow_pps = row.get("Flow Packets/s", 0)
            flow_bps = row.get("Flow Bytes/s", 0)
            dst_port = row.get("Destination Port", -1)

            # --- PortScan ---
            if (
                duration <= t["portscan_duration_max"]
                and fwd_packets <= t["portscan_fwd_packets_max"]
            ):
                label = "PortScan"

            # --- DoS / DDoS (grouped generically here) ---
            elif (
                flow_pps >= t["dos_flow_packets_per_sec_min"]
                or flow_bps >= t["dos_flow_bytes_per_sec_min"]
            ):
                label = "DoS Hulk"  # CHANGE HERE: split into DDoS/GoldenEye/
                # Slowloris/Slowhttptest if you add more
                # specific rules per type later

            # --- Brute Force (FTP/SSH) ---
            elif dst_port == t["ftp_port"] and duration >= t["bruteforce_duration_min"]:
                label = "FTP-Patator"
            elif dst_port == t["ssh_port"] and duration >= t["bruteforce_duration_min"]:
                label = "SSH-Patator"

            # --- Heartbleed ---
            elif (
                dst_port == t["heartbleed_port"]
                and flow_bps >= t["heartbleed_bytes_min"]
            ):
                label = "Heartbleed"

            # --- Bot ---
            elif (
                duration >= t["bot_duration_min"]
                and fwd_packets <= t["bot_fwd_packets_max"]
            ):
                label = "Bot"

            # --- Web Attacks (generic, since flow-level features
            # alone can't distinguish Brute Force vs XSS vs SQL
            # Injection without payload inspection) ---
            elif dst_port in t["web_ports"]:
                label = "Web Attack - Brute Force"  # CHANGE HERE: this is
                # a rough placeholder,
                # flag as a known
                # limitation in your BRD

            predictions.append(self.class_labels.index(label))

        return np.array(predictions)

    # ============================================
    # EVALUATION METRICS (same structure as XGBoostModel)
    # ============================================
    def evaluate(self, y_true, y_pred):
        metrics = {
            "Model": self.model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1_Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        return metrics

    def get_confusion_matrix(self, y_true, y_pred, class_labels=None):
        cm = confusion_matrix(y_true, y_pred)

        if class_labels is None:
            class_labels = self.class_labels

        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        return cm_df

    def get_classification_report(self, y_true, y_pred, class_labels=None):
        if class_labels is None:
            class_labels = self.class_labels

        report = classification_report(
            y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0
        )
        return pd.DataFrame(report).transpose()

    def export_for_powerbi(
        self,
        metrics_dict,
        cm_df,
        report_df,
        metrics_path="rulebased_metrics_summary.csv",
        cm_path="rulebased_confusion_matrix.csv",
        report_path="rulebased_classification_report.csv",
    ):
        pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)
        cm_df.to_csv(cm_path)
        report_df.to_csv(report_path)

        print(f"Exported: {metrics_path}")
        print(f"Exported: {cm_path}")
        print(f"Exported: {report_path}")


# ============================================
# INDEPENDENT MAIN RUN
# Running this file directly applies the rules AND evaluates them.
# ============================================
if __name__ == "__main__":

    # ============================================
    # CHANGE HERE: load your real, final dataset
    # Rule-based models don't need SMOTE'd training data since
    # there's no real "training," but you can still pass X_train
    # into .fit() if you want auto-tuned thresholds. y_test should
    # be ENCODED the same way as your XGBoost pipeline for fair
    # comparison.
    # ============================================
    # Example:
    # X_train = pd.read_csv("X_train_smote.csv")  # optional, for auto-tuning
    # X_test = pd.read_csv("X_test.csv")
    # y_test = pd.read_csv("y_test.csv").values.ravel()

    X_train, X_test, y_test = (
        None,
        None,
        None,
    )  # placeholders, remove once real data is loaded

    if X_test is None:
        raise Exception(
            "No data loaded yet. Replace the placeholder section above "
            "with your real X_test, y_test (and optionally X_train) "
            "once the final cleaned dataset is ready."
        )

    rule_model = RuleBasedModel()

    # Optional: auto-tune some thresholds using training data stats
    if X_train is not None:
        rule_model.fit(X_train)

    y_pred = rule_model.predict(X_test)

    # --- Evaluate ---
    metrics = rule_model.evaluate(y_test, y_pred)
    print("\nMetrics:", metrics)

    cm_df = rule_model.get_confusion_matrix(y_test, y_pred)
    report_df = rule_model.get_classification_report(y_test, y_pred)

    # --- Export for Power BI ---
    rule_model.export_for_powerbi(metrics, cm_df, report_df)

    print("\nRule-Based run complete. All results exported for Power BI.")
