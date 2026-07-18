-- ============================================================
-- fact_confusion_matrix_schema.sql
--
-- Replaces the CSV-imported fact_confusion_matrix (which used the
-- fabricated dim_attack_class_actual/_predicted tables and a fake
-- Class_ID 16 sentinel) with two grain-separated tables, same
-- pattern as fact_classification_metrics_multiclass / _binary.
--
--   fact_confusion_matrix_multiclass -- 15x15 actual x predicted,
--       used by XGBoost and Rule-Based (both predict one of the
--       15 real dim_label classes, never a generic "ATTACK").
--
--   fact_confusion_matrix_binary -- 2x2 actual x predicted,
--       used by Isolation Forest ONLY (BENIGN/ATTACK vs
--       BENIGN/ATTACK). Per this conversation's finding: Isolation
--       Forest's classifier never sees the multiclass label at
--       all (isolation_forest_model.py evaluates against
--       binary_label only), so a multiclass-actual-vs-binary-
--       predicted view has no real source data behind it yet.
--       Deferred as a documented BRD limitation, not built here.
--
-- Run dim_model_and_binary_label.sql BEFORE this script if you
-- haven't already (needed for both tables' Model_ID FK).
-- ============================================================

IF OBJECT_ID('dbo.fact_confusion_matrix_multiclass', 'U') IS NOT NULL
    DROP TABLE dbo.fact_confusion_matrix_multiclass;
GO

CREATE TABLE fact_confusion_matrix_multiclass (
    confusion_id      INT IDENTITY(1,1) PRIMARY KEY,
    model_id           INT NOT NULL,
    actual_label_id     INT NOT NULL,
    predicted_label_id  INT NOT NULL,
    count_val           INT NOT NULL,
    CONSTRAINT FK_factcmmc_model    FOREIGN KEY (model_id)          REFERENCES dim_model(model_id),
    CONSTRAINT FK_factcmmc_actual   FOREIGN KEY (actual_label_id)    REFERENCES dim_label(label_id),
    CONSTRAINT FK_factcmmc_pred     FOREIGN KEY (predicted_label_id) REFERENCES dim_label(label_id)
);
GO

IF OBJECT_ID('dbo.fact_confusion_matrix_binary', 'U') IS NOT NULL
    DROP TABLE dbo.fact_confusion_matrix_binary;
GO

CREATE TABLE fact_confusion_matrix_binary (
    confusion_id            INT IDENTITY(1,1) PRIMARY KEY,
    model_id                 INT NOT NULL,
    actual_binary_label_id    INT NOT NULL,
    predicted_binary_label_id INT NOT NULL,
    count_val                 INT NOT NULL,
    CONSTRAINT FK_factcmbin_model  FOREIGN KEY (model_id)                 REFERENCES dim_model(model_id),
    CONSTRAINT FK_factcmbin_actual FOREIGN KEY (actual_binary_label_id)    REFERENCES dim_binary_label(binary_label_id),
    CONSTRAINT FK_factcmbin_pred   FOREIGN KEY (predicted_binary_label_id) REFERENCES dim_binary_label(binary_label_id)
);
GO