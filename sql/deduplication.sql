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