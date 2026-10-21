-- Requires table `loan_ecl` (loan_id, report_month, scenario, stage, ead, ecl) from the Python ECL engine.

CREATE OR REPLACE TABLE ecl_by_scenario AS
SELECT
    report_month,
    scenario,
    stage,
    SUM(ead) AS ead,
    SUM(ecl) AS ecl,
    SUM(ecl) / NULLIF(SUM(ead), 0) AS coverage
FROM loan_ecl
GROUP BY report_month, scenario, stage;

-- Probability-weighted ECL.
CREATE OR REPLACE TABLE ecl_weighted AS
SELECT
    e.report_month,
    e.stage,
    SUM(e.ecl * w.weight) AS ecl_weighted,
    SUM(e.ead * w.weight) AS ead_weighted,
    SUM(e.ecl * w.weight) / NULLIF(SUM(e.ead * w.weight), 0) AS coverage_weighted
FROM ecl_by_scenario e
JOIN scenario_weights w USING (scenario)
GROUP BY e.report_month, e.stage;
