-- Default rate by origination vintage.
CREATE OR REPLACE TABLE vintage_default_rate AS
SELECT
    vintage_quarter,
    COUNT(*) AS observations,
    AVG(default_12m) AS default_rate_12m
FROM features
GROUP BY vintage_quarter
ORDER BY vintage_quarter;

-- Stage migration between consecutive months.
CREATE OR REPLACE TABLE stage_migration AS
WITH s AS (
    SELECT
        loan_id,
        report_month,
        stage,
        LEAD(stage) OVER (PARTITION BY loan_id ORDER BY report_month) AS next_stage
    FROM staging
)
SELECT stage, next_stage, COUNT(*) AS n
FROM s
WHERE next_stage IS NOT NULL
GROUP BY stage, next_stage
ORDER BY stage, next_stage;
