-- Resolved defaults with a realised loss, for the LGD model.
-- Zero balance codes treated as credit events: 02 third-party sale, 03 short sale, 09 REO disposition,
-- 15 note sale, 16 reperforming loan sale. LGD = actual loss / UPB at removal, clipped to [0, 1].
-- Only resolved defaults are used, so cured loans and unresolved defaults are not represented (workout bias).

CREATE OR REPLACE TABLE lgd_data AS
SELECT
    p.loan_id,
    p.report_month AS resolution_month,
    date_diff('month', l.orig_date, p.report_month) AS months_on_book,
    l.credit_score,
    l.orig_ltv,
    l.orig_dti,
    l.orig_rate,
    p.zero_bal_removal_upb AS default_upb,
    p.actual_loss,
    LEAST(GREATEST(p.actual_loss / NULLIF(p.zero_bal_removal_upb, 0), 0.0), 1.0) AS lgd,
    m.hpi_yoy,
    m.unemployment
FROM performance p
JOIN loans l ON l.loan_id = p.loan_id
LEFT JOIN macro m ON m.month = p.report_month AND m.scenario = 'base'
WHERE p.zero_bal_code IN ('02', '03', '09', '15', '16')
  AND p.actual_loss IS NOT NULL
  AND p.zero_bal_removal_upb > 0;
