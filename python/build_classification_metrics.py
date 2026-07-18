"""
build_fact_classification_metrics.py

Replaces the hand-assembled fact_classification_metrics.csv with a
script-driven build. Reads the three REAL per-model classification
report exports (rulebased_classification_report.csv,
isolation_forest_classification_report.csv,
xgboost_classification_report.csv -- as written by each model's
export_for_powerbi()) and assembles two grain-separated fact tables:

    fact_classification_metrics_multiclass.csv  (XGBoost, Rule-Based)
    fact_classification_metrics_binary.csv      (Isolation Forest)

WHY THIS EXISTS (see conversation history): a hand-built version of
this fact table previously substituted a 2-row, hand-typed stub for
the Rule-Based model's real 15-row classification report, and used a
synthetic "Class_ID 16" for Isolation Forest's ATTACK bucket that had
no corresponding dim_label row. This script removes both failure
modes by (a) reading the real CSVs directly instead of hand-typing
numbers, and (b) resolving every label -> ID mapping against the
LIVE dim_label / dim_binary_label / dim_model tables in Azure SQL,
never a hardcoded list that can silently drift out of sync (e.g.
LabelEncoder's alphabetical order vs. dim_label's undefined
SELECT DISTINCT order, or rule_based_model.py's hardcoded
"Web Attack - Sql Injection" vs. dim_label's
"Web Attack - SQL Injection").

Usage:
    python build_fact_classification_metrics.py

Requires a working .env (same as the rest of the pipeline) so it can
query dim_label / dim_binary_label / dim_model for the current, real
ID mappings.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from db_connection import get_engine  # reuse the project's existing engine

METRICS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "metrics")
)

# Model_ID assignment must match dim_model (see
# sql/dim_model_and_binary_label.sql) -- run that script first.
MODEL_FILES = {
    1: {"name": "XGBoost", "task": "Multiclass",
        "report": "xgboost_classification_report.csv"},
    2: {"name": "IsolationForest", "task": "Binary",
        "report": "isolation_forest_classification_report.csv"},
    3: {"name": "Rule-Based Signature Engine", "task": "Multiclass",
        "report": "rulebased_classification_report.csv"},
}

# Rows that classification_report(output_dict=True) adds beyond the
# per-class rows -- must be excluded before writing to the fact table,
# they are NOT classes and have no label_id / binary_label_id to map to.
SUMMARY_ROWS = {"accuracy", "macro avg", "weighted avg"}


def load_report(model_id: int) -> pd.DataFrame:
    info = MODEL_FILES[model_id]
    path = os.path.join(METRICS_DIR, info["report"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing real classification report for {info['name']}: {path}. "
            f"This script only reads real model output -- run the model's "
            f"own script (or run_pipeline.py) first, don't hand-type this file."
        )
    df = pd.read_csv(path, index_col=0)
    df.index.name = "class_name"
    df = df.reset_index()
    df = df[~df["class_name"].isin(SUMMARY_ROWS)]
    return df


def fetch_dim_label(engine) -> pd.DataFrame:
    """
    Pulls the LIVE label -> label_id mapping straight from Azure SQL.
    Never hardcode this list -- see module docstring for why (dim_label's
    SELECT DISTINCT has no ORDER BY, so its row order isn't guaranteed
    to match LabelEncoder's alphabetical order or any hardcoded Python
    list, even though it happened to match when manually checked).
    """
    return pd.read_sql("SELECT label_id, label FROM dim_label", engine)


def fetch_dim_binary_label(engine) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT binary_label_id, binary_label FROM dim_binary_label", engine
    )


def fetch_dim_model(engine) -> pd.DataFrame:
    return pd.read_sql("SELECT model_id, model_name, task_type FROM dim_model", engine)


def build_multiclass_fact(model_id: int, dim_label_df: pd.DataFrame) -> pd.DataFrame:
    report_df = load_report(model_id)

    label_to_id = dict(zip(dim_label_df["label"], dim_label_df["label_id"]))

    unmapped = set(report_df["class_name"]) - set(label_to_id.keys())
    if unmapped:
        raise ValueError(
            f"Model_ID {model_id} ({MODEL_FILES[model_id]['name']}): "
            f"classification report contains class names with no matching "
            f"dim_label row: {sorted(unmapped)}. This is exactly the kind "
            f"of silent-mismatch bug this script exists to prevent -- fix "
            f"the naming (e.g. casing drift between rule_based_model.py's "
            f"hardcoded fallback list and dim_label) before proceeding."
        )

    report_df = report_df.copy()
    report_df["model_id"] = model_id
    report_df["label_id"] = report_df["class_name"].map(label_to_id)

    # --- Validation: shape and totals must match expectations ---
    n_classes = len(dim_label_df)
    if len(report_df) != n_classes:
        raise ValueError(
            f"Model_ID {model_id}: expected {n_classes} class rows "
            f"(one per dim_label row), got {len(report_df)}. This is the "
            f"exact shape mismatch that let a 2-row binary stub get "
            f"substituted for a 15-row multiclass report previously -- "
            f"stopping rather than writing a malformed fact table."
        )

    out = report_df.rename(columns={
        "precision": "precision_score",
        "recall": "recall_score",
        "f1-score": "f1_score",
    })[["model_id", "label_id", "precision_score", "recall_score", "f1_score", "support"]]

    return out


def build_binary_fact(model_id: int, dim_binary_label_df: pd.DataFrame) -> pd.DataFrame:
    report_df = load_report(model_id)

    bl_to_id = dict(zip(dim_binary_label_df["binary_label"], dim_binary_label_df["binary_label_id"]))

    unmapped = set(report_df["class_name"]) - set(bl_to_id.keys())
    if unmapped:
        raise ValueError(
            f"Model_ID {model_id}: classification report contains binary "
            f"class names with no matching dim_binary_label row: "
            f"{sorted(unmapped)}. Check predict_labels()'s "
            f"benign_label/anomaly_label arguments match dim_binary_label "
            f"exactly (case-sensitive)."
        )

    report_df = report_df.copy()
    report_df["model_id"] = model_id
    report_df["binary_label_id"] = report_df["class_name"].map(bl_to_id)

    if len(report_df) != len(dim_binary_label_df):
        raise ValueError(
            f"Model_ID {model_id}: expected {len(dim_binary_label_df)} "
            f"binary class rows, got {len(report_df)}."
        )

    out = report_df.rename(columns={
        "precision": "precision_score",
        "recall": "recall_score",
        "f1-score": "f1_score",
    })[["model_id", "binary_label_id", "precision_score", "recall_score", "f1_score", "support"]]

    return out


def validate_support_totals(multiclass_df: pd.DataFrame, binary_df: pd.DataFrame):
    """
    Cross-model sanity check: every model's classification report should
    sum to the SAME test-set size (assuming all splits used the same
    source data and random_state), regardless of whether the model is
    binary or multiclass. This is the same check that originally
    surfaced how support numbers should relate to each other -- now
    automated instead of eyeballed.
    """
    totals = {}
    for model_id, group in multiclass_df.groupby("model_id"):
        totals[model_id] = group["support"].sum()
    for model_id, group in binary_df.groupby("model_id"):
        totals[model_id] = group["support"].sum()

    unique_totals = set(round(t) for t in totals.values())
    if len(unique_totals) > 1:
        print(
            f"WARNING: support totals differ across models: {totals}. "
            f"Expected all models' test sets to be the same size if they "
            f"share the same source data and random_state. Investigate "
            f"before trusting cross-model comparisons in Power BI."
        )
    else:
        print(f"Support totals match across all models: {unique_totals.pop():,.0f} rows. OK.")


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    engine = get_engine()

    dim_label_df = fetch_dim_label(engine)
    dim_binary_label_df = fetch_dim_binary_label(engine)
    dim_model_df = fetch_dim_model(engine)
    print(f"Loaded dim_label ({len(dim_label_df)} rows), "
          f"dim_binary_label ({len(dim_binary_label_df)} rows), "
          f"dim_model ({len(dim_model_df)} rows) from Azure SQL.")

    multiclass_parts = []
    binary_parts = []

    for model_id, info in MODEL_FILES.items():
        if info["task"] == "Multiclass":
            print(f"Building multiclass fact rows for Model_ID {model_id} ({info['name']})...")
            multiclass_parts.append(build_multiclass_fact(model_id, dim_label_df))
        else:
            print(f"Building binary fact rows for Model_ID {model_id} ({info['name']})...")
            binary_parts.append(build_binary_fact(model_id, dim_binary_label_df))

    multiclass_df = pd.concat(multiclass_parts, ignore_index=True)
    binary_df = pd.concat(binary_parts, ignore_index=True)

    validate_support_totals(multiclass_df, binary_df)

    mc_path = os.path.join(METRICS_DIR, "fact_classification_metrics_multiclass.csv")
    bin_path = os.path.join(METRICS_DIR, "fact_classification_metrics_binary.csv")
    multiclass_df.to_csv(mc_path, index=False)
    binary_df.to_csv(bin_path, index=False)

    print(f"\nExported: {mc_path} ({len(multiclass_df)} rows)")
    print(f"Exported: {bin_path} ({len(binary_df)} rows)")
    print("\nDone. Load both CSVs into Power BI, or bulk-insert into "
          "fact_classification_metrics_multiclass / _binary via SQL.")


if __name__ == "__main__":
    main()