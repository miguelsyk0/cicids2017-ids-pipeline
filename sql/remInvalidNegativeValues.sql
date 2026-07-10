DELETE FROM cic_typed
WHERE flow_duration < 0
   OR flow_iat_mean < 0
   OR flow_iat_max < 0
   OR flow_iat_min < 0
   OR fwd_header_length < 0
   OR bwd_header_length < 0
   OR min_seg_size_forward < 0;

SELECT COUNT(*) AS FinalRowCount FROM cic_typed;