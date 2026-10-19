-- IFRS 9 staging. Requires table `scores` (loan_id, report_month, pd_12m, pd_12m_orig)
-- written by the Python PD model (performing observations only).
-- Staging reads from `panel` (all months) so that defaulted loans reach Stage 3.
-- All thresholds are assumptions: document and test them.

CREATE OR REPLACE VIEW staging AS
SELECT
    p.loan_id,
    p.report_month,
    p.dpd_months,
    s.pd_12m,
    s.pd_12m_orig,
    CASE
        WHEN p.dpd_months >= 3 THEN 3                                   -- default
        WHEN p.dpd_months >= 1                                          -- 30 DPD backstop
          OR (s.pd_12m / NULLIF(s.pd_12m_orig, 0) >= 2.0                -- relative PD increase
              AND s.pd_12m - s.pd_12m_orig >= 0.005)                    -- and absolute increase
        THEN 2
        ELSE 1
    END AS stage
FROM panel p
LEFT JOIN scores s USING (loan_id, report_month);
