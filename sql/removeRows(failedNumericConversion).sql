DELETE FROM cic_typed
WHERE flow_duration IS NULL
   OR destination_port IS NULL
   OR label IS NULL;

SELECT COUNT(*) AS RowCountAfterNullRemoval FROM cic_typed;

DELETE FROM dbo.cic_typed
WHERE LTRIM(RTRIM(ISNULL(label,''))) = '' OR LTRIM(RTRIM(ISNULL(protocol,''))) = '';