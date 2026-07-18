-- ============================================================
-- dim_feature_and_remaining_facts_schema.sql
--
-- Completes the single-source-of-truth migration: the last two
-- CSV-imported tables (fact_model_metrics, fact_feature_importance
-- + dim_feature) get real SQL homes here. Both are low-risk by
-- construction (verified: fact_model_metrics has no per-class
-- grain to mismatch; fact_feature_importance is XGBoost-only, no
-- cross-model collapse possible) -- but they still get an FK into
-- dim_model, same as every other fact table, so nothing in this
-- schema can point at a model that doesn't exist.
--
-- Run dim_model_and_binary_label.sql BEFORE this script.
-- ============================================================

IF OBJECT_ID('dbo.fact_model_metrics', 'U') IS NOT NULL
    DROP TABLE dbo.fact_model_metrics;
GO

CREATE TABLE fact_model_metrics (
    metric_id           INT IDENTITY(1,1) PRIMARY KEY,
    model_id             INT NOT NULL,
    accuracy             FLOAT NOT NULL,
    precision_weighted    FLOAT NOT NULL,
    recall_weighted       FLOAT NOT NULL,
    f1_weighted           FLOAT NOT NULL,
    precision_macro       FLOAT NOT NULL,
    recall_macro          FLOAT NOT NULL,
    f1_macro              FLOAT NOT NULL,
    CONSTRAINT FK_factmm_model FOREIGN KEY (model_id) REFERENCES dim_model(model_id)
);
GO

IF OBJECT_ID('dbo.fact_feature_importance', 'U') IS NOT NULL
    DROP TABLE dbo.fact_feature_importance;
GO

IF OBJECT_ID('dbo.dim_feature', 'U') IS NOT NULL
    DROP TABLE dbo.dim_feature;
GO

CREATE TABLE dim_feature (
    feature_id    INT IDENTITY(1,1) PRIMARY KEY,
    feature_name   VARCHAR(100) NOT NULL UNIQUE  -- e.g. "flow_duration",
                                                   -- "fwd_packet_length_mean"
);
GO

CREATE TABLE fact_feature_importance (
    feature_importance_id INT IDENTITY(1,1) PRIMARY KEY,
    model_id                INT NOT NULL,
    feature_id               INT NOT NULL,
    importance_score          FLOAT NOT NULL,
    rank_val                  INT NOT NULL,  -- 1 = most important, for
                                              -- this model
    CONSTRAINT FK_factfi_model   FOREIGN KEY (model_id)   REFERENCES dim_model(model_id),
    CONSTRAINT FK_factfi_feature FOREIGN KEY (feature_id) REFERENCES dim_feature(feature_id)
);
GO