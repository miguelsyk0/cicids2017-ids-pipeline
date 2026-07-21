"""
load_anomaly_scores_to_sql.py

Loads isolation_forest_anomaly_scores.csv (per-row anomaly scores from
IsolationForestModel.export_for_powerbi()) into Azure SQL as
fact_anomaly_scores. Reuses db_connection.get_engine() -- same connection
pattern as the rest of the pipeline, no separate credential handling.
"""
import os
import pandas as pd
from db_connection import get_engine

METRICS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "metrics")
)


def load_anomaly_scores(table_name="fact_anomaly_scores", if_exists="replace"):
    path = os.path.join(METRICS_DIR, "isolation_forest_anomaly_scores.csv")
    df = pd.read_csv(path)

    # Sanity check -- catches a silently mismatched export (e.g. an old
    # CSV from before a column rename) before it reaches SQL as garbage.
    expected_cols = {"true_label", "predicted_label", "anomaly_score"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"{path} is missing expected columns. Found: {df.columns.tolist()}, "
            f"expected at least: {expected_cols}"
        )

    engine = get_engine()
    df.to_sql(table_name, engine, if_exists=if_exists, index=False, chunksize=1000)
    print(f"Loaded {len(df):,} rows into {table_name}")


if __name__ == "__main__":
    load_anomaly_scores()