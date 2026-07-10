ALTER TABLE cic_typed ADD binary_label VARCHAR(10);
GO

UPDATE cic_typed
SET binary_label = CASE WHEN label = 'BENIGN' THEN 'BENIGN' ELSE 'ATTACK' END;
GO

ALTER TABLE cic_typed ADD port_group VARCHAR(20);
GO

UPDATE cic_typed
SET port_group = CASE
    WHEN destination_port IN (80, 443, 8080) THEN 'Web'
    WHEN destination_port = 21 THEN 'FTP'
    WHEN destination_port = 22 THEN 'SSH'
    ELSE 'Other'
END;
GO