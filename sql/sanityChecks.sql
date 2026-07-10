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