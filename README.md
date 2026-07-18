# ITE 17 -- Network Intrusion Detection System (CICIDS2017)

A three-model IDS built on the CICIDS2017 dataset, comparing a hardcoded
signature engine, an unsupervised anomaly detector, and a supervised
multiclass classifier -- staged through Azure SQL and exported for a
Power BI dashboard.

## Models

| Model | File | Task | Input |
|---|---|---|---|
| Rule-Based Signature Engine | `python/models/rule_based_model.py` | Multiclass (attack type), baseline | Raw, unscaled features |
| Isolation Forest | `python/models/isolation_forest_model.py` | Binary (BENIGN vs ATTACK), unsupervised | Scaled, unbalanced, continuous-only features |
| XGBoost | `python/models/xgboost_model.py` | Multiclass (attack type) | Scaled, one-hot encoded, SMOTE-balanced |

Each model gets its **own preprocessing branch** -- see the docstring in
`python/pipeline.py` for why one shared `X_train` sliced three ways would
be wrong here (Isolation Forest specifically loses ~2x false-positive
rate if fed one-hot dummy columns; confirmed by testing, not just theory).

All three models share the same evaluation interface
(`evaluate()`, `get_confusion_matrix()`, `get_classification_report()`,
`export_for_powerbi()`) so their metrics dicts line up for a single
comparison table.

## Repository layout

```
python/
  db_connection.py      -- Azure SQL engine/connection, reads .env
  data_fetch.py          -- fetch_table() / fetch_training_data() wrappers
  preprocessing.py        -- shared split/encode/scale/SMOTE pipeline
  import.py               -- one-time raw CSV -> Azure SQL loader (raw_flows)
  pipeline.py              -- end-to-end orchestration: fetch -> preprocess -> train all 3 models -> export
  models/
    rule_based_model.py
    isolation_forest_model.py
    xgboost_model.py
sql/
  properlyTypedTable.sql      -- raw_flows -> cic_typed (TRY_CAST typing)
  deduplication.sql
  removeRows(failedNumericConversion).sql
  remInvalidNegativeValues.sql
  standardizedLabelsql.sql    -- normalizes Web Attack label variants
  engineeredColumns.sql        -- adds binary_label, port_group
  starSchema.sql               -- builds dim_* / fact_network_flow (Power BI)
  starSchemaForeignKeys.sql
  sanityChecks.sql
  rebuild_pipeline.sql         -- all of the above, in dependency order
data/metrics/                  -- CSV outputs consumed by Power BI (git-ignored, created at runtime)
artifacts/                      -- saved scaler / encoder / model .joblib files
```

## Setup

1. **Python dependencies**
   ```
   pip install -r requirements.txt
   ```
   You also need the **ODBC Driver 18 for SQL Server** installed at the
   OS level (not a pip package) -- see `requirements.txt` for
   platform-specific install commands.

2. **Environment variables** -- create a `.env` in the project root
   (never commit this):
   ```
   AZURE_SQL_SERVER=your-server.database.windows.net
   AZURE_SQL_DATABASE=your-database
   AZURE_SQL_USERNAME=your-username
   AZURE_SQL_PASSWORD=your-password
   AZURE_SQL_DRIVER=ODBC Driver 18 for SQL Server
   ```
   Verify connectivity:
   ```
   python python/db_connection.py
   ```

3. **Azure firewall** -- add your (and teammates') public IP under
   SQL Server -> Networking -> Firewall rules, or the connection is
   refused outright.

## Running the pipeline

**One-time data load** (raw CSVs -> `raw_flows` table):
```
python python/import.py
```
Edit `CSV_FOLDER` at the top of that file first. This step is pure
load-as-is; no cleaning happens here.

**Database rebuild** (raw_flows -> cic_typed -> star schema), run once
against Azure SQL:
```
sql/rebuild_pipeline.sql
```
This assembles every script in `sql/` in dependency order. Run the
individual scripts instead if you only need to re-run one stage.

**Train + evaluate all three models:**
```
python python/pipeline.py
```
This fetches `cic_typed` in chunks, branches into the three
preprocessing paths described above, trains/evaluates each model, and
writes:
- per-model metrics summary, confusion matrix, classification report to `data/metrics/`
- a combined `model_comparison_summary.csv`
- fitted scaler/encoder/model artifacts to `artifacts/`

Individual model files (`rule_based_model.py`,
`isolation_forest_model.py`, `xgboost_model.py`) can also be run
directly for isolated testing, but their `__main__` blocks currently
raise on placeholder data until wired to real inputs -- use
`pipeline.py` for an actual end-to-end run.

## Design decisions worth knowing before you touch this code

- **Three separate preprocessing branches, not one shared `X_train`.**
  Rule-Based wants raw units; Isolation Forest wants scaled-but-
  unbalanced continuous-only features (no categoricals, no
  `hour_of_day`); XGBoost wants scaled + one-hot + SMOTE-balanced +
  integer-encoded labels. Slicing one shared frame three ways looks
  efficient but silently breaks each model's assumptions.
- **Split always happens before scaling/encoding/SMOTE**, fit only on
  `X_train`, to avoid test-set leakage.
- **BorderlineSMOTE only synthesizes "danger" (boundary) samples** --
  requesting N synthetic samples for a class with none can silently
  produce zero, no error. `apply_smote()` checks post-resample counts
  against the request and warns if they don't match.
- **`XGBoost` runs before `Rule-Based` in `pipeline.py`** so the
  rule-based branch can reuse the saved `label_encoder.joblib` instead
  of a second, independently hand-typed label list.

## Known limitations (documented, not yet fixed)

- The rule engine's DoS/DDoS/GoldenEye/Slowloris/Slowhttptest variants
  all collapse into a single `"DoS Hulk"` bucket -- it has no logic to
  distinguish them. Flagged as a placeholder in `rule_based_model.py`.
- The rule engine's default (fallback) `class_labels` list is
  hand-typed and can drift from `sql/standardizedLabelsql.sql`'s actual
  label strings. Always pass `encoder.classes_.tolist()` from a saved
  `LabelEncoder` when one is available -- `pipeline.py` does this
  automatically after the first XGBoost run.
- SQL-level deduplication (`sql/deduplication.sql`) partitions on
  `source_port, destination_port, timestamp, flow_duration,
  total_fwd_packets, total_backward_packets, label` -- it does **not**
  include `protocol`. Two flows differing only by protocol but
  otherwise identical on those columns would be deduplicated as if
  they were the same flow.
- Feature-level duplicates can still cross the train/test split even
  after SQL dedup (rows can share identical feature values while
  differing on IP/port/timestamp, which SQL dedup doesn't catch).
  `pipeline.py::run_xgboost()` prints a diagnostic percentage of
  test-set rows that are exact feature-duplicates of a training row --
  check this number before trusting XGBoost's reported accuracy.
- `Isolation Forest`'s `contamination` can be set from the empirical
  attack ratio in `y_train` (`estimate_contamination()`), which
  technically leans on label information for a hyperparameter choice
  even though `.fit()` itself never sees labels -- worth stating
  explicitly as a stated limitation in the BRD if you use it instead of
  `'auto'`.

## Security note

`.env` is never committed. `python/import.py` currently has Azure SQL
credentials as literal strings at the top of the file for one-time
local use -- **move these to `.env` via `db_connection.get_engine()`
before this file is committed anywhere**, and rotate the password in
Azure if it has ever been pushed to a remote repository.