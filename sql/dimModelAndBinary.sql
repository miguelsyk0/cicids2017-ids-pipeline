-- ============================================================
-- dim_model_and_binary_label.sql
--
-- Two small dimensions needed before fact_classification_metrics
-- can be split by grain (see fact_classification_metrics_schema.sql):
--
--   dim_model:        formalizes the Model_ID values that were
--                      previously implicit/hand-assigned (1=XGBoost,
--                      2=IsolationForest, 3=RuleBased) in the old
--                      hand-built fact_classification_metrics.csv.
--
--   dim_binary_label:  Isolation Forest predicts BENIGN/ATTACK, not
--                      one of the 15 dim_label rows. Previously this
--                      was faked as a synthetic "Class_ID 16" with no
--                      real dim_label row behind it. This dimension
--                      gives that binary outcome a real, small table
--                      to join against instead.
-- ============================================================

IF OBJECT_ID('dbo.dim_model', 'U') IS NOT NULL
    DROP TABLE dbo.dim_model;
GO

CREATE TABLE dim_model (
    model_id   INT IDENTITY(1,1) PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    task_type  VARCHAR(20) NOT NULL  -- 'Multiclass' or 'Binary' -- keeps
                                      -- the distinction from
                                      -- pipeline.py's build_comparison_table()
                                      -- visible at the dimension level too
);

-- Explicit INSERT with fixed IDs (not SELECT DISTINCT) -- this is exactly
-- the kind of table where undefined row order previously caused a
-- silent Class_ID mismatch risk in dim_label. Never again.
SET IDENTITY_INSERT dim_model ON;
INSERT INTO dim_model (model_id, model_name, task_type) VALUES
    (1, 'XGBoost',         'Multiclass'),
    (2, 'IsolationForest', 'Binary'),
    (3, 'Rule-Based Signature Engine', 'Multiclass');
SET IDENTITY_INSERT dim_model OFF;
GO

IF OBJECT_ID('dbo.dim_binary_label', 'U') IS NOT NULL
    DROP TABLE dbo.dim_binary_label;
GO

CREATE TABLE dim_binary_label (
    binary_label_id INT IDENTITY(1,1) PRIMARY KEY,
    binary_label     VARCHAR(10) NOT NULL
);

SET IDENTITY_INSERT dim_binary_label ON;
INSERT INTO dim_binary_label (binary_label_id, binary_label) VALUES
    (1, 'BENIGN'),
    (2, 'ATTACK');
SET IDENTITY_INSERT dim_binary_label OFF;
GO