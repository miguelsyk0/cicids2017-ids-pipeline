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
    def __init__(self, class_labels=None):
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
        # class_labels defaults to the hardcoded list below, but you can
        # (and should, once available) pass in encoder.classes_.tolist()
        # from preprocessing.encode_labels() instead:
        #
        #   from preprocessing import load_artifact
        #   encoder = load_artifact("artifacts/label_encoder.joblib")
        #   rule_model = RuleBasedModel(class_labels=encoder.classes_.tolist())
        #
        # This matters because xgboost_model.py ALSO needs this exact
        # same order (via its own class_labels_ / encoder), and right now
        # that order is hand-typed independently in two files. If the two
        # ever drift apart (e.g. a class is dropped or renamed and only
        # one file gets updated), predictions would be silently
        # mislabeled without either file raising an error. Passing the
        # same saved encoder into both models removes that risk entirely.
        # ============================================
        self.class_labels = class_labels or [
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

        CHANGE HERE: column names below must match your actual cleaned
        dataset's column names exactly -- these use the project's real
        cic_typed schema (snake_case), NOT the original CICIDS2017 CSV
        headers (which use "Flow Duration" style with spaces/capitals).
        """
        try:
            self.thresholds["dos_flow_packets_per_sec_min"] = X_train[
                "flow_packets_s"
            ].quantile(0.95)
            self.thresholds["portscan_duration_max"] = X_train[
                "flow_duration"
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
        Applies rules to the whole DataFrame at once (vectorized via
        np.select), rather than row-by-row. Confirmed by timing: row-by-row
        iterrows() projects to ~13-60s on a full ~566K-row test set;
        np.select does the same first-match-wins logic in a fraction of
        that. Returns predictions as ENCODED integers matching
        self.class_labels order, so it plugs into the same evaluation
        functions as your XGBoost model.

        Column names below match the project's actual cic_typed schema
        (snake_case) -- NOT the original CICIDS2017 CSV headers. Adjust
        or add rules based on which features your final dataset keeps.

        Rule order matters: np.select evaluates conditions top-to-bottom
        and takes the FIRST match, mirroring the original if/elif chain's
        priority (e.g. a row matching both the DoS and Bot thresholds
        gets DoS, since that condition is listed first).
        """
        t = self.thresholds

        duration = X_test.get("flow_duration", pd.Series(0, index=X_test.index))
        fwd_packets = X_test.get("total_fwd_packets", pd.Series(0, index=X_test.index))
        flow_pps = X_test.get("flow_packets_s", pd.Series(0, index=X_test.index))
        flow_bps = X_test.get("flow_bytes_s", pd.Series(0, index=X_test.index))
        dst_port = X_test.get("destination_port", pd.Series(-1, index=X_test.index))

        conditions = [
            (duration <= t["portscan_duration_max"]) & (fwd_packets <= t["portscan_fwd_packets_max"]),
            (flow_pps >= t["dos_flow_packets_per_sec_min"]) | (flow_bps >= t["dos_flow_bytes_per_sec_min"]),
            (dst_port == t["ftp_port"]) & (duration >= t["bruteforce_duration_min"]),
            (dst_port == t["ssh_port"]) & (duration >= t["bruteforce_duration_min"]),
            (dst_port == t["heartbleed_port"]) & (flow_bps >= t["heartbleed_bytes_min"]),
            (duration >= t["bot_duration_min"]) & (fwd_packets <= t["bot_fwd_packets_max"]),
            dst_port.isin(t["web_ports"]),
        ]
        choices = [
            "PortScan",
            "DoS Hulk",  # CHANGE HERE: split into DDoS/GoldenEye/Slowloris/
                         # Slowhttptest if you add more specific rules later
            "FTP-Patator",
            "SSH-Patator",
            "Heartbleed",
            "Bot",
            "Web Attack - Brute Force",  # CHANGE HERE: rough placeholder,
                                         # flag as a known limitation in your BRD
        ]

        labels = np.select(conditions, choices, default="BENIGN")
        label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
        return np.array([label_to_index[label] for label in labels])

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
    # Real pipeline wiring:
    #
    # from data_fetch import fetch_training_data
    # from preprocessing import split_data, load_artifact
    #
    # df = pd.concat(fetch_training_data(chunksize=200_000), ignore_index=True)
    # df = df.drop(columns=["binary_label"])
    # X_train, X_test, y_train, y_test = split_data(df, target_col="label")
    #
    # IMPORTANT: this model reads RAW, UNSCALED, named features directly --
    # consistent with the project's rule-based-layer design (thresholds
    # like "duration <= 100" only mean something in real units). Do NOT
    # pass X_train/X_test through fit_scaler() or encode_categoricals()
    # before this model -- that's for XGBoost/Isolation Forest only.
    #
    # encoder = load_artifact("artifacts/label_encoder.joblib")
    # rule_model = RuleBasedModel(class_labels=encoder.classes_.tolist())
    # y_test_encoded = encoder.transform(y_test)
    # ============================================

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