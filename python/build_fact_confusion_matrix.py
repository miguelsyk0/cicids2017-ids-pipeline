"""
build_fact_confusion_matrix.py

Builds fact_confusion_matrix_multiclass.csv and
fact_confusion_matrix_binary.csv from REAL source files only:

    Multiclass (Model_ID 1 = XGBoost, Model_ID 3 = Rule-Based):
        xgboost_confusion_matrix.csv, rulebased_confusion_matrix.csv
        -- both square label x label matrices as written by each
        model's get_confusion_matrix() / export_for_powerbi().

    Binary (Model_ID 2 = Isolation Forest):
        isolation_forest_anomaly_scores.csv -- crosstabbed here into
        a 2x2 BENIGN/ATTACK matrix. NOT built from a hand-typed
        actual-multiclass-vs-predicted-binary table: confirmed in
        this project that isolation_forest_model.py never sees the
        multiclass label at train/eval time (evaluate() takes
        binary_label only), so no real source data exists yet for
        a class-level breakdown of Isolation Forest's misses. That
        richer view is deferred -- see BRD limitations -- and should
        only be built later by threading the original multiclass
        y_test through run_isolation_forest() in pipeline.py before
        scores_df is constructed.

Like build_fact_classification_metrics.py, every class name is
resolved against the LIVE dim_label / dim_binary_label tables in
Azure SQL, never a hardcoded list.

Usage:
    python build_fact_confusion_matrix.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from db_connection import get_engine

METRICS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "metrics")
)

MULTICLASS_MODELS = {
    1: {"name": "XGBoost", "file": "xgboost_confusion_matrix.csv"},
    3: {"name": "Rule-Based Signature Engine", "file": "rulebased_confusion_matrix.csv"},
}
BINARY_MODEL = {"model_id": 2, "name": "IsolationForest", "file": "isolation_forest_anomaly_scores.csv"}


def fetch_dim_label(engine) -> pd.DataFrame:
    return pd.read_sql("SELECT label_id, label FROM dim_label", engine)


def fetch_dim_binary_label(engine) -> pd.DataFrame:
    return pd.read_sql("SELECT binary_label_id, binary_label FROM dim_binary_label", engine)


def build_multiclass_fact(model_id: int, filename: str, dim_label_df: pd.DataFrame) -> pd.DataFrame:
    path = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing real confusion matrix for Model_ID {model_id}: {path}. "
            f"Run the model's own script (or run_pipeline.py) first -- "
            f"don't hand-build this file."
        )

    cm = pd.read_csv(path, index_col=0)  # square: index=actual, columns=predicted

    label_to_id = dict(zip(dim_label_df["label"], dim_label_df["label_id"]))
    unmapped = (set(cm.index) | set(cm.columns)) - set(label_to_id.keys())
    if unmapped:
        raise ValueError(
            f"Model_ID {model_id}: confusion matrix has class names with no "
            f"matching dim_label row: {sorted(unmapped)}."
        )

    long_df = cm.stack().reset_index()
    long_df.columns = ["actual_label", "predicted_label", "count_val"]
    long_df["model_id"] = model_id
    long_df["actual_label_id"] = long_df["actual_label"].map(label_to_id)
    long_df["predicted_label_id"] = long_df["predicted_label"].map(label_to_id)

    n_classes = len(dim_label_df)
    if cm.shape != (n_classes, n_classes):
        raise ValueError(
            f"Model_ID {model_id}: expected a {n_classes}x{n_classes} matrix, "
            f"got {cm.shape}. This is the same shape check that previously "
            f"caught a binary-collapsed stub standing in for a real "
            f"multiclass matrix -- stopping rather than writing bad data."
        )

    return long_df[["model_id", "actual_label_id", "predicted_label_id", "count_val"]]


def build_binary_fact(dim_binary_label_df: pd.DataFrame) -> pd.DataFrame:
    path = os.path.join(METRICS_DIR, BINARY_MODEL["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    scores_df = pd.read_csv(path)
    if not {"true_label", "predicted_label"}.issubset(scores_df.columns):
        raise ValueError(
            f"{path} is missing true_label/predicted_label columns -- "
            f"can't build a real crosstab from it."
        )

    bl_to_id = dict(zip(dim_binary_label_df["binary_label"], dim_binary_label_df["binary_label_id"]))
    unmapped = (set(scores_df["true_label"]) | set(scores_df["predicted_label"])) - set(bl_to_id.keys())
    if unmapped:
        raise ValueError(
            f"Isolation Forest scores file has label values with no "
            f"matching dim_binary_label row: {sorted(unmapped)}."
        )

    ct = pd.crosstab(scores_df["true_label"], scores_df["predicted_label"])
    long_df = ct.stack().reset_index()
    long_df.columns = ["actual_binary_label", "predicted_binary_label", "count_val"]
    long_df["model_id"] = BINARY_MODEL["model_id"]
    long_df["actual_binary_label_id"] = long_df["actual_binary_label"].map(bl_to_id)
    long_df["predicted_binary_label_id"] = long_df["predicted_binary_label"].map(bl_to_id)

    if len(long_df) != 4:  # 2x2
        raise ValueError(f"Expected 4 rows (2x2 binary matrix), got {len(long_df)}.")

    return long_df[["model_id", "actual_binary_label_id", "predicted_binary_label_id", "count_val"]]


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    engine = get_engine()

    dim_label_df = fetch_dim_label(engine)
    dim_binary_label_df = fetch_dim_binary_label(engine)

    mc_parts = []
    for model_id, info in MULTICLASS_MODELS.items():
        print(f"Building multiclass confusion matrix for Model_ID {model_id} ({info['name']})...")
        mc_parts.append(build_multiclass_fact(model_id, info["file"], dim_label_df))
    multiclass_df = pd.concat(mc_parts, ignore_index=True)

    print(f"Building binary confusion matrix for Model_ID {BINARY_MODEL['model_id']} "
          f"({BINARY_MODEL['name']}) from real anomaly scores...")
    binary_df = build_binary_fact(dim_binary_label_df)

    # Validation: each model's matrix should sum to the same test-set size
    # as its classification report did.
    for model_id, group in multiclass_df.groupby("model_id"):
        total = group["count_val"].sum()
        print(f"Model_ID {model_id} confusion matrix total: {total:,}")
    binary_total = binary_df["count_val"].sum()
    print(f"Model_ID {BINARY_MODEL['model_id']} confusion matrix total: {binary_total:,}")

    mc_path = os.path.join(METRICS_DIR, "fact_confusion_matrix_multiclass.csv")
    bin_path = os.path.join(METRICS_DIR, "fact_confusion_matrix_binary.csv")
    multiclass_df.to_csv(mc_path, index=False)
    binary_df.to_csv(bin_path, index=False)

    print(f"\nExported: {mc_path} ({len(multiclass_df)} rows)")
    print(f"Exported: {bin_path} ({len(binary_df)} rows)")
    print("\nNOTE: no multiclass-actual-vs-binary-predicted view is built here "
          "for Isolation Forest -- deferred as a documented BRD limitation "
          "(see module docstring). Only a real 2x2 BENIGN/ATTACK matrix is "
          "produced for Model_ID 2.")


if __name__ == "__main__":
    main()