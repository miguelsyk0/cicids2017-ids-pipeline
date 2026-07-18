-- ============================================================
-- fact_classification_metrics_schema.sql
--
-- Replaces the single hand-assembled fact_classification_metrics
-- table with two grain-separated fact tables:
--
--   fact_classification_metrics_multiclass -- one row per
--       (model, one of the 15 dim_label classes). Used by XGBoost
--       and Rule-Based, both of which predict the full attack-type
--       label.
--
--   fact_classification_metrics_binary -- one row per
--       (model, BENIGN/ATTACK). Used by Isolation Forest, which
--       only makes a binary inlier/outlier call.
--
-- WHY SPLIT: forcing both grains into one Class_ID column is what
-- produced the fake "Class_ID 16" sentinel in the old hand-built
-- CSV (Isolation Forest's ATTACK bucket has no corresponding
-- dim_label row -- it aggregates all 14 attack types). Splitting
-- by grain means every FK in both tables now points at a REAL
-- dimension row, and Power BI visuals built against each fact
-- table naturally can't mix binary and multiclass results on the
-- same chart by accident.
--
-- Run dim_model_and_binary_label.sql BEFORE this script.
-- ============================================================

IF OBJECT_ID('dbo.fact_classification_metrics_multiclass', 'U') IS NOT NULL
    DROP TABLE dbo.fact_classification_metrics_multiclass;
GO

CREATE TABLE fact_classification_metrics_multiclass (
    classification_metric_id INT IDENTITY(1,1) PRIMARY KEY,
    model_id   INT NOT NULL,
    label_id   INT NOT NULL,
    precision_score FLOAT NOT NULL,
    recall_score    FLOAT NOT NULL,
    f1_score        FLOAT NOT NULL,
    support         FLOAT NOT NULL,  -- FLOAT to match sklearn's own
                                      -- classification_report output dtype
    CONSTRAINT FK_factmc_model FOREIGN KEY (model_id) REFERENCES dim_model(model_id),
    CONSTRAINT FK_factmc_label FOREIGN KEY (label_id) REFERENCES dim_label(label_id)
);
GO

IF OBJECT_ID('dbo.fact_classification_metrics_binary', 'U') IS NOT NULL
    DROP TABLE dbo.fact_classification_metrics_binary;
GO

CREATE TABLE fact_classification_metrics_binary (
    classification_metric_id INT IDENTITY(1,1) PRIMARY KEY,
    model_id         INT NOT NULL,
    binary_label_id  INT NOT NULL,
    precision_score FLOAT NOT NULL,
    recall_score    FLOAT NOT NULL,
    f1_score        FLOAT NOT NULL,
    support         FLOAT NOT NULL,
    CONSTRAINT FK_factbin_model FOREIGN KEY (model_id) REFERENCES dim_model(model_id),
    CONSTRAINT FK_factbin_binlabel FOREIGN KEY (binary_label_id) REFERENCES dim_binary_label(binary_label_id)
);
GO