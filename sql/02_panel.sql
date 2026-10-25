-- Monthly loan panel with default flag. Default = 90+ DPD (3 months).

CREATE OR REPLACE TABLE panel AS
SELECT
    p.loan_id,
    p.report_month,
    l.orig_date,
    date_diff('month', l.orig_date, p.report_month) AS months_on_book,
    p.current_upb,
    p.dpd_months,
    COALESCE(p.zero_bal_code, '') AS zero_bal_code,
    CASE WHEN p.dpd_months >= 3 THEN 1 ELSE 0 END AS in_default,
    MIN(CASE WHEN p.dpd_months >= 3 THEN p.report_month END)
        OVER (PARTITION BY p.loan_id) AS first_default_month
FROM performance p
JOIN loans l USING (loan_id);

-- Modelling sample: performing observations only, with a 12-month forward default target.
-- The row where a loan leaves the pool (prepayment or maturity) is not an observation of a live loan and is
-- excluded: its missing balance would reveal the outcome.
-- Observations in the last 12 months of the data are dropped: their 12-month window is not fully
-- observed (right-censored), so a "no default" label would be unreliable.
CREATE OR REPLACE TABLE panel_target AS
SELECT
    *,
    CASE
        WHEN first_default_month IS NOT NULL
         AND first_default_month >  report_month
         AND first_default_month <= report_month + INTERVAL 12 MONTH
        THEN 1 ELSE 0
    END AS default_12m
FROM panel
WHERE in_default = 0
  AND zero_bal_code = ''
  AND report_month <= CAST((SELECT MAX(report_month) FROM panel) - INTERVAL 12 MONTH AS DATE);
