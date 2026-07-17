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