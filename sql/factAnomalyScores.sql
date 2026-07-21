IF OBJECT_ID('dbo.fact_iso_confusion', 'U') IS NOT NULL
    DROP TABLE dbo.fact_iso_confusion;
GO

CREATE TABLE fact_iso_confusion (
    iso_confusion_id INT IDENTITY(1,1) PRIMARY KEY,
    true_label VARCHAR(10) NOT NULL,
    predicted_label VARCHAR(10) NOT NULL,
    row_count INT NOT NULL
);
GO

IF OBJECT_ID('dbo.fact_anomaly_scores', 'U') IS NOT NULL
    DROP TABLE dbo.fact_anomaly_scores;
GO

CREATE TABLE fact_anomaly_scores (
    anomaly_score_id INT IDENTITY(1,1) PRIMARY KEY,
    true_label VARCHAR(10) NOT NULL,
    predicted_label VARCHAR(10) NOT NULL,
    anomaly_score FLOAT NOT NULL
);
GO