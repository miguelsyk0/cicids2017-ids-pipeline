-- ============================================================
-- Rebuild pipeline assembled from the SQL scripts in /sql
-- Dependency order used: raw_flows -> cic_typed -> dimensions/fact
-- ============================================================

-- ============ ingestion fix: raw_flows source_day tagging ============
-- This assumes raw_flows already exists and contains a source_file column,
-- because the ingestion loader writes that column during CSV import.
UPDATE raw_flows
SET source_day = 'Thursday-Afternoon'
WHERE source_file = 'thursday-workinghours-afternoon-infileteration.pcap_iscx.csv';

-- ============ properlyTypedTable.sql ============
IF OBJECT_ID('dbo.cic_typed', 'U') IS NOT NULL
    DROP TABLE dbo.cic_typed;
GO

SELECT
    TRY_CAST(source_port AS INT)                       AS source_port,
    TRY_CAST(destination_port AS INT)                  AS destination_port,
    protocol,
    TRY_CONVERT(DATETIME, [timestamp], 103)             AS [timestamp],
    TRY_CAST(flow_duration AS FLOAT)                    AS flow_duration,
    TRY_CAST(total_fwd_packets AS FLOAT)                AS total_fwd_packets,
    TRY_CAST(total_backward_packets AS FLOAT)           AS total_backward_packets,
    TRY_CAST(total_length_of_fwd_packets AS FLOAT)      AS total_length_of_fwd_packets,
    TRY_CAST(total_length_of_bwd_packets AS FLOAT)      AS total_length_of_bwd_packets,
    TRY_CAST(fwd_packet_length_max AS FLOAT)            AS fwd_packet_length_max,
    TRY_CAST(fwd_packet_length_min AS FLOAT)            AS fwd_packet_length_min,
    TRY_CAST(fwd_packet_length_mean AS FLOAT)           AS fwd_packet_length_mean,
    TRY_CAST(fwd_packet_length_std AS FLOAT)            AS fwd_packet_length_std,
    TRY_CAST(bwd_packet_length_max AS FLOAT)            AS bwd_packet_length_max,
    TRY_CAST(bwd_packet_length_min AS FLOAT)            AS bwd_packet_length_min,
    TRY_CAST(bwd_packet_length_mean AS FLOAT)           AS bwd_packet_length_mean,
    TRY_CAST(bwd_packet_length_std AS FLOAT)            AS bwd_packet_length_std,
    TRY_CAST(flow_bytes_s AS FLOAT)                     AS flow_bytes_s,
    TRY_CAST(flow_packets_s AS FLOAT)                   AS flow_packets_s,
    TRY_CAST(flow_iat_mean AS FLOAT)                    AS flow_iat_mean,
    TRY_CAST(flow_iat_std AS FLOAT)                     AS flow_iat_std,
    TRY_CAST(flow_iat_max AS FLOAT)                     AS flow_iat_max,
    TRY_CAST(flow_iat_min AS FLOAT)                     AS flow_iat_min,
    TRY_CAST(fwd_iat_total AS FLOAT)                     AS fwd_iat_total,
    TRY_CAST(fwd_iat_mean AS FLOAT)                      AS fwd_iat_mean,
    TRY_CAST(fwd_iat_std AS FLOAT)                       AS fwd_iat_std,
    TRY_CAST(fwd_iat_max AS FLOAT)                      AS fwd_iat_max,
    TRY_CAST(fwd_iat_min AS FLOAT)                      AS fwd_iat_min,
    TRY_CAST(bwd_iat_total AS FLOAT)                     AS bwd_iat_total,
    TRY_CAST(bwd_iat_mean AS FLOAT)                      AS bwd_iat_mean,
    TRY_CAST(bwd_iat_std AS FLOAT)                       AS bwd_iat_std,
    TRY_CAST(bwd_iat_max AS FLOAT)                      AS bwd_iat_max,
    TRY_CAST(bwd_iat_min AS FLOAT)                      AS bwd_iat_min,
    TRY_CAST(fwd_psh_flags AS FLOAT)                     AS fwd_psh_flags,
    TRY_CAST(bwd_psh_flags AS FLOAT)                     AS bwd_psh_flags,
    TRY_CAST(fwd_urg_flags AS FLOAT)                     AS fwd_urg_flags,
    TRY_CAST(bwd_urg_flags AS FLOAT)                     AS bwd_urg_flags,
    TRY_CAST(fwd_header_length AS FLOAT)                 AS fwd_header_length,
    TRY_CAST(bwd_header_length AS FLOAT)                 AS bwd_header_length,
    TRY_CAST(fwd_packets_s AS FLOAT)                     AS fwd_packets_s,
    TRY_CAST(bwd_packets_s AS FLOAT)                     AS bwd_packets_s,
    TRY_CAST(min_packet_length AS FLOAT)                 AS min_packet_length,
    TRY_CAST(max_packet_length AS FLOAT)                 AS max_packet_length,
    TRY_CAST(packet_length_mean AS FLOAT)                AS packet_length_mean,
    TRY_CAST(packet_length_std AS FLOAT)                 AS packet_length_std,
    TRY_CAST(packet_length_variance AS FLOAT)            AS packet_length_variance,
    TRY_CAST(fin_flag_count AS FLOAT)                    AS fin_flag_count,
    TRY_CAST(syn_flag_count AS FLOAT)                    AS syn_flag_count,
    TRY_CAST(rst_flag_count AS FLOAT)                    AS rst_flag_count,
    TRY_CAST(psh_flag_count AS FLOAT)                    AS psh_flag_count,
    TRY_CAST(ack_flag_count AS FLOAT)                    AS ack_flag_count,
    TRY_CAST(urg_flag_count AS FLOAT)                    AS urg_flag_count,
    TRY_CAST(cwe_flag_count AS FLOAT)                    AS cwe_flag_count,
    TRY_CAST(ece_flag_count AS FLOAT)                    AS ece_flag_count,
    TRY_CAST(down_up_ratio AS FLOAT)                     AS down_up_ratio,
    TRY_CAST(average_packet_size AS FLOAT)               AS average_packet_size,
    TRY_CAST(avg_fwd_segment_size AS FLOAT)              AS avg_fwd_segment_size,
    TRY_CAST(avg_bwd_segment_size AS FLOAT)              AS avg_bwd_segment_size,
    TRY_CAST(fwd_avg_bytes_bulk AS FLOAT)                AS fwd_avg_bytes_bulk,
    TRY_CAST(fwd_avg_packets_bulk AS FLOAT)              AS fwd_avg_packets_bulk,
    TRY_CAST(fwd_avg_bulk_rate AS FLOAT)                 AS fwd_avg_bulk_rate,
    TRY_CAST(bwd_avg_bytes_bulk AS FLOAT)                AS bwd_avg_bytes_bulk,
    TRY_CAST(bwd_avg_packets_bulk AS FLOAT)              AS bwd_avg_packets_bulk,
    TRY_CAST(bwd_avg_bulk_rate AS FLOAT)                 AS bwd_avg_bulk_rate,
    TRY_CAST(subflow_fwd_packets AS FLOAT)               AS subflow_fwd_packets,
    TRY_CAST(subflow_fwd_bytes AS FLOAT)                 AS subflow_fwd_bytes,
    TRY_CAST(subflow_bwd_packets AS FLOAT)               AS subflow_bwd_packets,
    TRY_CAST(subflow_bwd_bytes AS FLOAT)                 AS subflow_bwd_bytes,
    TRY_CAST(init_win_bytes_forward AS FLOAT)            AS init_win_bytes_forward,
    TRY_CAST(init_win_bytes_backward AS FLOAT)           AS init_win_bytes_backward,
    TRY_CAST(act_data_pkt_fwd AS FLOAT)                  AS act_data_pkt_fwd,
    TRY_CAST(min_seg_size_forward AS FLOAT)              AS min_seg_size_forward,
    TRY_CAST(active_mean AS FLOAT)                       AS active_mean,
    TRY_CAST(active_std AS FLOAT)                        AS active_std,
    TRY_CAST(active_max AS FLOAT)                        AS active_max,
    TRY_CAST(active_min AS FLOAT)                        AS active_min,
    TRY_CAST(idle_mean AS FLOAT)                         AS idle_mean,
    TRY_CAST(idle_std AS FLOAT)                          AS idle_std,
    TRY_CAST(idle_max AS FLOAT)                          AS idle_max,
    TRY_CAST(idle_min AS FLOAT)                          AS idle_min,
    NULLIF(LTRIM(RTRIM(label)),'')                       AS label,
    CASE WHEN NULLIF(LTRIM(RTRIM(label)),'') = 'BENIGN' THEN 'BENIGN' ELSE 'ATTACK' END AS binary_label,
    CASE WHEN destination_port IN (80, 443, 8080) THEN 'Web'
         WHEN destination_port = 21 THEN 'FTP'
         WHEN destination_port = 22 THEN 'SSH'
         ELSE 'Other'
    END AS port_group,
    source_day
INTO cic_typed
FROM raw_flows;

-- ============ deduplication.sql ============
WITH Deduped AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY source_port, destination_port, [timestamp],
                            flow_duration, total_fwd_packets, total_backward_packets, label
               ORDER BY (SELECT NULL)
           ) AS RowNum
    FROM cic_typed
)
DELETE FROM Deduped
WHERE RowNum > 1;

SELECT COUNT(*) AS RowCountAfterDedup FROM cic_typed;

-- ============ removeRows(failedNumericConversion).sql ============
DELETE FROM cic_typed
WHERE flow_duration IS NULL
   OR destination_port IS NULL
   OR label IS NULL;

SELECT COUNT(*) AS RowCountAfterNullRemoval FROM cic_typed;

-- ============ remInvalidNegativeValues.sql ============
DELETE FROM cic_typed
WHERE flow_duration < 0
   OR flow_iat_mean < 0
   OR flow_iat_max < 0
   OR flow_iat_min < 0
   OR fwd_header_length < 0
   OR bwd_header_length < 0
   OR min_seg_size_forward < 0;

SELECT COUNT(*) AS FinalRowCount FROM cic_typed;

-- ============ standardizedLabelsql.sql ============
UPDATE cic_typed
SET label = 'Web Attack - Brute Force'
WHERE label LIKE 'Web Attack%Brute Force%';

UPDATE cic_typed
SET label = 'Web Attack - XSS'
WHERE label LIKE 'Web Attack%XSS%';

UPDATE cic_typed
SET label = 'Web Attack - SQL Injection'
WHERE label LIKE 'Web Attack%Sql Injection%';

-- ============ starSchema.sql ============
IF OBJECT_ID('dbo.fact_network_flow', 'U') IS NOT NULL
    DROP TABLE dbo.fact_network_flow;
GO

IF OBJECT_ID('dbo.dim_label', 'U') IS NOT NULL
    DROP TABLE dbo.dim_label;
GO

IF OBJECT_ID('dbo.dim_day', 'U') IS NOT NULL
    DROP TABLE dbo.dim_day;
GO

IF OBJECT_ID('dbo.dim_port_group', 'U') IS NOT NULL
    DROP TABLE dbo.dim_port_group;
GO

IF OBJECT_ID('dbo.dim_protocol', 'U') IS NOT NULL
    DROP TABLE dbo.dim_protocol;
GO

-- Dimension: Label
CREATE TABLE dim_label (
    label_id INT IDENTITY(1,1) PRIMARY KEY,
    label VARCHAR(50) NOT NULL,
    binary_label VARCHAR(10) NOT NULL
);

INSERT INTO dim_label (label, binary_label)
SELECT DISTINCT label, binary_label
FROM cic_typed
WHERE ISNULL(LTRIM(RTRIM(label)), '') <> '';

-- Dimension: Day
CREATE TABLE dim_day (
    day_id INT IDENTITY(1,1) PRIMARY KEY,
    source_day VARCHAR(30) NOT NULL
);

INSERT INTO dim_day (source_day)
SELECT DISTINCT source_day
FROM cic_typed;

-- Dimension: Port Group
CREATE TABLE dim_port_group (
    port_group_id INT IDENTITY(1,1) PRIMARY KEY,
    port_group VARCHAR(20) NOT NULL
);

INSERT INTO dim_port_group (port_group)
SELECT DISTINCT port_group
FROM cic_typed;

-- Dimension: Protocol
CREATE TABLE dim_protocol (
    protocol_id INT IDENTITY(1,1) PRIMARY KEY,
    protocol VARCHAR(20) NOT NULL
);

INSERT INTO dim_protocol (protocol)
SELECT DISTINCT protocol
FROM cic_typed
WHERE ISNULL(LTRIM(RTRIM(protocol)), '') <> '';

-- Fact table: one row per flow, foreign keys to each dimension,
-- plus every numeric measurement
SELECT
    dl.label_id,
    dd.day_id,
    dp.port_group_id,
    dpr.protocol_id,
    c.source_port,
    c.destination_port,
    c.[timestamp],
    c.flow_duration,
    c.total_fwd_packets,
    c.total_backward_packets,
    c.total_length_of_fwd_packets,
    c.total_length_of_bwd_packets,
    c.fwd_packet_length_max,
    c.fwd_packet_length_min,
    c.fwd_packet_length_mean,
    c.fwd_packet_length_std,
    c.bwd_packet_length_max,
    c.bwd_packet_length_min,
    c.bwd_packet_length_mean,
    c.bwd_packet_length_std,
    c.flow_bytes_s,
    c.flow_packets_s,
    c.flow_iat_mean,
    c.flow_iat_std,
    c.flow_iat_max,
    c.flow_iat_min,
    c.fwd_iat_total,
    c.fwd_iat_mean,
    c.fwd_iat_std,
    c.fwd_iat_max,
    c.fwd_iat_min,
    c.bwd_iat_total,
    c.bwd_iat_mean,
    c.bwd_iat_std,
    c.bwd_iat_max,
    c.bwd_iat_min,
    c.fwd_psh_flags,
    c.bwd_psh_flags,
    c.fwd_urg_flags,
    c.bwd_urg_flags,
    c.fwd_header_length,
    c.bwd_header_length,
    c.fwd_packets_s,
    c.bwd_packets_s,
    c.min_packet_length,
    c.max_packet_length,
    c.packet_length_mean,
    c.packet_length_std,
    c.packet_length_variance,
    c.fin_flag_count,
    c.syn_flag_count,
    c.rst_flag_count,
    c.psh_flag_count,
    c.ack_flag_count,
    c.urg_flag_count,
    c.cwe_flag_count,
    c.ece_flag_count,
    c.down_up_ratio,
    c.average_packet_size,
    c.avg_fwd_segment_size,
    c.avg_bwd_segment_size,
    c.fwd_avg_bytes_bulk,
    c.fwd_avg_packets_bulk,
    c.fwd_avg_bulk_rate,
    c.bwd_avg_bytes_bulk,
    c.bwd_avg_packets_bulk,
    c.bwd_avg_bulk_rate,
    c.subflow_fwd_packets,
    c.subflow_fwd_bytes,
    c.subflow_bwd_packets,
    c.subflow_bwd_bytes,
    c.init_win_bytes_forward,
    c.init_win_bytes_backward,
    c.act_data_pkt_fwd,
    c.min_seg_size_forward,
    c.active_mean,
    c.active_std,
    c.active_max,
    c.active_min,
    c.idle_mean,
    c.idle_std,
    c.idle_max,
    c.idle_min
INTO fact_network_flow
FROM cic_typed c
JOIN dim_label dl        ON c.label = dl.label AND c.binary_label = dl.binary_label
JOIN dim_day dd          ON c.source_day = dd.source_day
JOIN dim_port_group dp   ON c.port_group = dp.port_group
JOIN dim_protocol dpr    ON c.protocol = dpr.protocol;

-- ============ starSchemaForeignKeys.sql ============
IF OBJECT_ID('dbo.FK_fact_label', 'F') IS NOT NULL
    ALTER TABLE dbo.fact_network_flow DROP CONSTRAINT FK_fact_label;
GO

IF OBJECT_ID('dbo.FK_fact_day', 'F') IS NOT NULL
    ALTER TABLE dbo.fact_network_flow DROP CONSTRAINT FK_fact_day;
GO

IF OBJECT_ID('dbo.FK_fact_portgroup', 'F') IS NOT NULL
    ALTER TABLE dbo.fact_network_flow DROP CONSTRAINT FK_fact_portgroup;
GO

IF OBJECT_ID('dbo.FK_fact_protocol', 'F') IS NOT NULL
    ALTER TABLE dbo.fact_network_flow DROP CONSTRAINT FK_fact_protocol;
GO

ALTER TABLE fact_network_flow
ADD CONSTRAINT FK_fact_label FOREIGN KEY (label_id) REFERENCES dim_label(label_id);

ALTER TABLE fact_network_flow
ADD CONSTRAINT FK_fact_day FOREIGN KEY (day_id) REFERENCES dim_day(day_id);

ALTER TABLE fact_network_flow
ADD CONSTRAINT FK_fact_portgroup FOREIGN KEY (port_group_id) REFERENCES dim_port_group(port_group_id);

ALTER TABLE fact_network_flow
ADD CONSTRAINT FK_fact_protocol FOREIGN KEY (protocol_id) REFERENCES dim_protocol(protocol_id);

-- ============ sanityChecks.sql ============
SELECT dl.label, COUNT(*) AS Count
FROM fact_network_flow f
JOIN dim_label dl ON f.label_id = dl.label_id
GROUP BY dl.label
ORDER BY Count DESC;

SELECT dd.source_day, COUNT(*) AS Count
FROM fact_network_flow f
JOIN dim_day dd ON f.day_id = dd.day_id
GROUP BY dd.source_day;

SELECT dl.binary_label, COUNT(*) AS Count
FROM fact_network_flow f
JOIN dim_label dl ON f.label_id = dl.label_id
GROUP BY dl.binary_label;
