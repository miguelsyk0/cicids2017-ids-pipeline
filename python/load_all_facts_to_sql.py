"""
load_all_facts_to_sql.py

Final step of the CSV -> SQL migration. Two jobs:

  1. Builds the two tables that never had SQL homes:
       fact_model_metrics       <- xgboost/rulebased/isolation_forest
                                    _metrics_summary.csv (one row each)
       dim_feature + 
       fact_feature_importance  <- xgboost_feature_importance.csv
                                    (XGBoost-only, per module docstring
                                    in xgboost_model.py)

  2. Loads ALL SIX fact/dim outputs directly into Azure SQL via
     to_sql(if_exists="append"), so Power BI can point at live tables
     instead of any CSV:
       fact_classification_metrics_multiclass / _binary
           (from build_fact_classification_metrics.py)
       fact_confusion_matrix_multiclass / _binary
           (from build_fact_confusion_matrix.py)
       fact_model_metrics, dim_feature, fact_feature_importance
           (built here)

Run AFTER:
    sql/dim_model_and_binary_label.sql
    sql/fact_classification_metrics_schema.sql
    sql/fact_confusion_matrix_schema.sql
    sql/dim_feature_and_remaining_facts_schema.sql
    python/build_fact_classification_metrics.py
    python/build_fact_confusion_matrix.py

Usage:
    python load_all_facts_to_sql.py
"""

import os
import sys
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from db_connection import get_engine

METRICS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "metrics")
)

MODEL_SUMMARY_FILES = {
    1: "xgboost_metrics_summary.csv",
    2: "isolation_forest_metrics_summary.csv",
    3: "rulebased_metrics_summary.csv",
}

FEATURE_IMPORTANCE_FILE = "xgboost_feature_importance.csv"  # XGBoost-only
FEATURE_IMPORTANCE_MODEL_ID = 1

ANOMALY_SCORES_FILE = "isolation_forest_anomaly_scores.csv"
ANOMALY_SCORES_MODEL_ID = 2  # IsolationForest only, per dim_model

# Already-built fact CSVs (from the two earlier scripts) -> target SQL table
DIRECT_LOAD_FILES = {
    "fact_classification_metrics_multiclass.csv": "fact_classification_metrics_multiclass",
    "fact_classification_metrics_binary.csv": "fact_classification_metrics_binary",
    "fact_confusion_matrix_multiclass.csv": "fact_confusion_matrix_multiclass",
    "fact_confusion_matrix_binary.csv": "fact_confusion_matrix_binary",
}


def build_fact_model_metrics() -> pd.DataFrame:
    rows = []
    for model_id, filename in MODEL_SUMMARY_FILES.items():
        path = os.path.join(METRICS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path)
        if len(df) != 1:
            raise ValueError(f"{filename}: expected exactly 1 row, got {len(df)}")
        row = df.iloc[0]
        rows.append({
            "model_id": model_id,
            "accuracy": row["Accuracy"],
            "precision_weighted": row["Precision_Weighted"],
            "recall_weighted": row["Recall_Weighted"],
            "f1_weighted": row["F1_Weighted"],
            "precision_macro": row["Precision_Macro"],
            "recall_macro": row["Recall_Macro"],
            "f1_macro": row["F1_Macro"],
        })
    return pd.DataFrame(rows)


def build_feature_importance_source(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)  # columns: Feature, Importance (see xgboost_model.py's
                             # get_feature_importance())
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    df["rank_val"] = df.index + 1
    return df


def load_dim_feature(engine, df: pd.DataFrame) -> dict:
    """
    Loads feature names into dim_feature WITHOUT specifying feature_id --
    it's an IDENTITY column, SQL Server assigns it. Explicitly assigning
    our own sequential IDs and inserting them (the original bug here)
    fails with IntegrityError unless IDENTITY_INSERT is turned on, which
    to_sql() doesn't do. Simpler and more correct to just let SQL
    generate them, then read the real assignments back.

    Returns:
        dict mapping feature_name -> the feature_id SQL Server actually
        assigned, for use in building fact_feature_importance.
    """
    unique_names = pd.DataFrame({"feature_name": df["Feature"].unique()})

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM fact_feature_importance"))  # child
                                                                     # first (FK)
        conn.execute(text("DELETE FROM dim_feature"))
    unique_names.to_sql("dim_feature", engine, if_exists="append", index=False)
    print(f"Loaded {len(unique_names)} rows into dim_feature")

    reloaded = pd.read_sql("SELECT feature_id, feature_name FROM dim_feature", engine)
    return dict(zip(reloaded["feature_name"], reloaded["feature_id"]))


def build_fact_feature_importance(df: pd.DataFrame, name_to_id: dict) -> pd.DataFrame:
    unmapped = set(df["Feature"]) - set(name_to_id.keys())
    if unmapped:
        raise ValueError(f"Features with no dim_feature row after load: {sorted(unmapped)}")

    return pd.DataFrame({
        "model_id": FEATURE_IMPORTANCE_MODEL_ID,
        "feature_id": df["Feature"].map(name_to_id),
        "importance_score": df["Importance"],
        "rank_val": df["rank_val"],
    })


def build_fact_anomaly_scores(engine, path: str) -> pd.DataFrame:
    """
    Loads the raw anomaly scores CSV (true_label/predicted_label as TEXT,
    per IsolationForestModel.export_for_powerbi()) and joins against
    dim_binary_label to resolve real FK ids -- mirrors the manual SSMS
    migration already done for the existing fact_anomaly_scores data,
    but repeatable on every pipeline rerun instead of a one-off.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)

    expected_cols = {"true_label", "predicted_label", "anomaly_score"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"{path} missing expected columns. Found: {df.columns.tolist()}, "
            f"expected at least: {expected_cols}"
        )

    dim_binary = pd.read_sql(
        "SELECT binary_label_id, binary_label FROM dim_binary_label", engine
    )
    label_to_id = dict(zip(dim_binary["binary_label"], dim_binary["binary_label_id"]))

    unmapped = set(df["true_label"]) | set(df["predicted_label"])
    unmapped -= set(label_to_id.keys())
    if unmapped:
        raise ValueError(
            f"Labels in {path} not found in dim_binary_label: {sorted(unmapped)}"
        )

    return pd.DataFrame({
        "model_id": ANOMALY_SCORES_MODEL_ID,
        "actual_binary_label_id": df["true_label"].map(label_to_id),
        "predicted_binary_label_id": df["predicted_label"].map(label_to_id),
        "anomaly_score": df["anomaly_score"],
    })


def load_table(engine, df: pd.DataFrame, table_name: str):
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table_name}"))  # idempotent reruns --
                                                            # avoids duplicate
                                                            # rows if this
                                                            # script is run
                                                            # more than once
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into {table_name}")


def main():
    engine = get_engine()

    print("Building fact_model_metrics from real summary CSVs...")
    model_metrics_df = build_fact_model_metrics()
    load_table(engine, model_metrics_df, "fact_model_metrics")

    print("Building dim_feature + fact_feature_importance from real XGBoost export...")
    fi_source_df = build_feature_importance_source(os.path.join(METRICS_DIR, FEATURE_IMPORTANCE_FILE))
    name_to_id = load_dim_feature(engine, fi_source_df)  # deletes + reloads
                                                           # dim_feature AND
                                                           # fact_feature_importance
                                                           # (child, for the FK)
    fact_fi_df = build_fact_feature_importance(fi_source_df, name_to_id)
    fact_fi_df.to_sql("fact_feature_importance", engine, if_exists="append", index=False)
    print(f"Loaded {len(fact_fi_df)} rows into fact_feature_importance")

    print("\nBuilding fact_anomaly_scores from real Isolation Forest export...")
    anomaly_df = build_fact_anomaly_scores(engine, os.path.join(METRICS_DIR, ANOMALY_SCORES_FILE))
    load_table(engine, anomaly_df, "fact_anomaly_scores")

    print("\nLoading already-built classification/confusion CSVs directly into SQL...")
    for filename, table_name in DIRECT_LOAD_FILES.items():
        path = os.path.join(METRICS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path} -- run build_fact_classification_metrics.py "
                f"and build_fact_confusion_matrix.py first."
            )
        df = pd.read_csv(path)
        load_table(engine, df, table_name)

    print("\nMigration complete. Every fact/dim table Power BI needs now "
          "lives in Azure SQL -- no CSV imports required.")


if __name__ == "__main__":
    main()