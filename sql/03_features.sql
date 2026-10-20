-- Features for PD / LGD / EAD models.

CREATE OR REPLACE TABLE features AS
SELECT
    t.loan_id,
    t.report_month,
    t.months_on_book,
    t.default_12m,
    t.dpd_months,
    l.credit_score,
    l.orig_ltv,
    l.orig_dti,
    l.orig_rate,
    t.current_upb / NULLIF(l.orig_upb, 0) AS balance_ratio,
    MAX(t.dpd_months) OVER (
        PARTITION BY t.loan_id ORDER BY t.report_month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    ) AS max_dpd_last_12m,
    date_trunc('quarter', l.orig_date) AS vintage_quarter,
    m.unemployment,
    m.hpi_yoy,
    m.gdp_yoy
FROM panel_target t
JOIN loans l USING (loan_id)
LEFT JOIN macro m
       ON m.month = t.report_month AND m.scenario = 'base';

-- Same features for every performing observation, including the last 12 months that have no
-- complete target. Used to score the portfolio (staging and ECL need the latest months too).
CREATE OR REPLACE TABLE features_scoring AS
SELECT
    t.loan_id,
    t.report_month,
    t.months_on_book,
    t.dpd_months,
    l.credit_score,
    l.orig_ltv,
    l.orig_dti,
    l.orig_rate,
    t.current_upb / NULLIF(l.orig_upb, 0) AS balance_ratio,
    MAX(t.dpd_months) OVER (
        PARTITION BY t.loan_id ORDER BY t.report_month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    ) AS max_dpd_last_12m,
    date_trunc('quarter', l.orig_date) AS vintage_quarter,
    m.unemployment,
    m.hpi_yoy,
    m.gdp_yoy
FROM panel t
JOIN loans l USING (loan_id)
LEFT JOIN macro m
       ON m.month = t.report_month AND m.scenario = 'base'
WHERE t.in_default = 0;
