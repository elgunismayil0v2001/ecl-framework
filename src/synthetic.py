"""Synthetic data for tests, CI and smoke runs. No real data needed."""
import numpy as np
import pandas as pd


def make_pd_dataset(n=20000, seed=0):
    """Loan-level data with a known default process. `vintage` (0 to 7) supports time-based splits."""
    rng = np.random.default_rng(seed)
    credit_score = rng.normal(700, 50, n).clip(500, 830)
    ltv = rng.uniform(40, 97, n)
    dti = rng.uniform(10, 50, n)
    logit = -3.4 - 0.015 * (credit_score - 700) + 0.025 * (ltv - 70) + 0.03 * (dti - 30)
    p_true = 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame(
        {
            "credit_score": credit_score,
            "ltv": ltv,
            "dti": dti,
            "p_true": p_true,
            "default": rng.binomial(1, p_true),
            "vintage": rng.integers(0, 8, n),
        }
    )


def make_loan_panel(n_loans=300, months=36, seed=0):
    """Return (loans, performance) DataFrames matching sql/01_schema.sql.

    Delinquency follows a simple Markov chain. 90+ DPD (dpd_months >= 3) is absorbing.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2018-01-01")
    loan_rows, perf_rows = [], []
    for i in range(n_loans):
        loan_id = f"L{i:05d}"
        orig = start + pd.DateOffset(months=int(rng.integers(0, 12)))
        upb = float(rng.uniform(100_000, 400_000))
        loan_rows.append(
            (
                loan_id,
                orig,
                upb,
                float(rng.uniform(40, 97)),
                float(rng.uniform(10, 50)),
                int(rng.integers(560, 820)),
                float(rng.uniform(0.03, 0.07)),
                360,
                "CA",
            )
        )
        dpd = 0
        for m in range(months):
            if dpd < 3:
                u = rng.random()
                if dpd == 0:
                    dpd = 1 if u < 0.02 else 0
                else:
                    dpd = 0 if u < 0.3 else (dpd + 1 if u < 0.8 else dpd)
            perf_rows.append(
                (loan_id, orig + pd.DateOffset(months=m), upb * (1 - m / 360), dpd, "")
            )
    loans = pd.DataFrame(
        loan_rows,
        columns=[
            "loan_id",
            "orig_date",
            "orig_upb",
            "orig_ltv",
            "orig_dti",
            "credit_score",
            "orig_rate",
            "term_months",
            "property_state",
        ],
    )
    performance = pd.DataFrame(
        perf_rows,
        columns=["loan_id", "report_month", "current_upb", "dpd_months", "zero_bal_code"],
    )
    return loans, performance
