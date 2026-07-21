-- ============================================================
-- fixTimestampAMPM.sql
--
-- Recovers the lost AM/PM marker for Thursday and Friday flows,
-- where source_day already encodes it as text ("Thursday-Morning",
-- "Thursday-Afternoon", "Friday-Morning", "Friday-Afternoon-PortScan",
-- "Friday-Afternoon-DDoS") but hour_of_day (derived from [timestamp]
-- at pipeline runtime) was collapsed to the 1-12 range on import.
--
-- Scope: this is a categorical lookup, not a heuristic. Validated
-- against real hour distributions (see timestampRolloverCheck.sql,
-- Part A) -- every "Morning" split sits in the 8-12 band and every
-- "Afternoon" split sits in the 1-5 band, confirming a clean 12-hour
-- clock split with the AM/PM marker stripped.
--
-- Monday/Tuesday/Wednesday are DELIBERATELY EXCLUDED. Rollover
-- detection via %%physloc%% (see timestampRolloverCheck.sql, Part C)
-- came back as noise (~9-13% of rows per day flagged as "drops",
-- spread evenly across the full sequence rather than concentrated at
-- one rollover point) -- there is no reliable ordinal signal left in
-- cic_typed to recover AM/PM for these three days. This is documented
-- as a stated BRD limitation, not something this script attempts to
-- patch.
--
-- Edge case handled: hour = 12 in an Afternoon file is already noon
-- (12:00 in 24-hour time) -- adding 12 would incorrectly roll it to
-- hour 24. Only hours 1-11 get shifted.
--
-- Run this AFTER properlyTypedTable.sql, BEFORE deduplication.sql --
-- see note at the bottom on why placement matters for future reruns.
-- ============================================================

-- Preview affected rows before committing (sanity check row counts
-- match Part A's per-day totals for the Afternoon splits)
SELECT
    source_day,
    COUNT(*) AS rows_to_shift
FROM cic_typed
WHERE source_day LIKE '%Afternoon%'
  AND DATEPART(HOUR, [timestamp]) <> 12
GROUP BY source_day;

-- The actual correction
UPDATE cic_typed
SET [timestamp] = DATEADD(HOUR, 12, [timestamp])
WHERE source_day LIKE '%Afternoon%'
  AND DATEPART(HOUR, [timestamp]) <> 12;

-- Verification: hour ranges per day should now show Afternoon splits
-- in the 13-23 band (or 12 exactly, for the untouched noon rows) and
-- Morning splits unchanged in 8-12.
SELECT
    source_day,
    MIN(DATEPART(HOUR, [timestamp])) AS min_hour,
    MAX(DATEPART(HOUR, [timestamp])) AS max_hour,
    COUNT(*) AS total_flows
FROM cic_typed
WHERE [timestamp] IS NOT NULL
GROUP BY source_day
ORDER BY source_day;

-- ============================================================
-- PLACEMENT NOTE for rebuild_pipeline.sql:
-- deduplication.sql partitions on [timestamp] as part of its
-- uniqueness key. Running this correction AFTER dedup would mean
-- future full rebuilds dedup on the WRONG (pre-correction) timestamp
-- values. It shouldn't change which rows are flagged as duplicates in
-- practice (this is a per-row, deterministic +12 shift, not a value
-- that creates new collisions), but running it BEFORE dedup keeps the
-- dedup key consistent with the corrected data and avoids relying on
-- that assumption. Insert this step into rebuild_pipeline.sql directly
-- after the properlyTypedTable.sql block and before the
-- deduplication.sql block.
--
-- IMPORTANT FOLLOW-UP: this only fixes the SQL-side timestamp. Any
-- previously trained model (XGBoost's feature importance results
-- currently in the BRD, Section 7.2.3/7.2.4) was trained on
-- hour_of_day BEFORE this correction. Those results need to be
-- regenerated post-fix before Section 7.2.4's "no timing pattern"
-- finding can be treated as validated -- flagging this as a queued
-- retrain, not doing it in this pass.
-- ============================================================