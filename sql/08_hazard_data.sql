-- Person-period data for the discrete-time hazard model (lifetime PD).
-- One row per performing loan-month. default_next = 1 if the loan is 90+ DPD in the next month.
-- Loans with no next month (prepaid or liquidated) are dropped, which treats prepayment as censoring.

CREATE OR REPLACE TABLE hazard_obs AS
SELECT
    t.loan_id,
    t.report_month,
    t.months_on_book,
    t.dpd_months,
    t.current_upb,
    l.credit_score,
    l.orig_ltv,
    l.orig_dti,
    l.orig_rate,
    l.term_months,
    m.unemployment,
    m.hpi_yoy,
    m.gdp_yoy,
    CASE WHEN n.dpd_months >= 3 THEN 1 ELSE 0 END AS default_next
FROM panel t
JOIN loans l ON l.loan_id = t.loan_id
JOIN panel n
  ON n.loan_id = t.loan_id
 AND n.report_month = CAST(t.report_month + INTERVAL 1 MONTH AS DATE)
LEFT JOIN macro m ON m.month = t.report_month AND m.scenario = 'base'
WHERE t.in_default = 0;
