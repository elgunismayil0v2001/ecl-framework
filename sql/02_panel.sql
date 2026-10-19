-- Monthly loan panel with default flag. Default = 90+ DPD (3 months).

CREATE OR REPLACE TABLE panel AS
SELECT
    p.loan_id,
    p.report_month,
    l.orig_date,
    date_diff('month', l.orig_date, p.report_month) AS months_on_book,
    p.current_upb,
    p.dpd_months,
    CASE WHEN p.dpd_months >= 3 THEN 1 ELSE 0 END AS in_default,
    MIN(CASE WHEN p.dpd_months >= 3 THEN p.report_month END)
        OVER (PARTITION BY p.loan_id) AS first_default_month
FROM performance p
JOIN loans l USING (loan_id);

-- Modelling sample: performing observations only, with a 12-month forward default target.
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
WHERE in_default = 0;
