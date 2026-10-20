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


def make_hazard_dataset(n_loans=4000, months=48, seed=0):
    """Person-period data with a known monthly default hazard that depends on age and macro variables."""
    rng = np.random.default_rng(seed)
    t = np.arange(months + 12)
    unemployment = 5.0 + 2.0 * np.sin(2 * np.pi * t / 36) + rng.normal(0, 0.1, len(t))
    # different cycle lengths keep the three macro series from being collinear
    hpi_yoy = 0.03 + 0.04 * np.sin(2 * np.pi * t / 29 + 1.0) + rng.normal(0, 0.005, len(t))
    gdp_yoy = 0.02 + 0.025 * np.sin(2 * np.pi * t / 21 + 2.0) + rng.normal(0, 0.003, len(t))

    credit_score = rng.normal(700, 50, n_loans).clip(500, 830)
    ltv = rng.uniform(40, 97, n_loans)
    dti = rng.uniform(10, 50, n_loans)
    rate = rng.uniform(0.03, 0.07, n_loans)
    orig = rng.integers(0, 12, n_loans)

    alive = np.ones(n_loans, dtype=bool)
    parts = []
    for age in range(months):
        cal = orig + age
        logit = (
            -7.6
            - 0.012 * (credit_score - 700)
            + 0.02 * (ltv - 75)
            + 0.01 * (dti - 35)
            + 10.0 * (rate - 0.05)
            + 0.30 * (unemployment[cal] - 5.0)
            - 4.0 * (hpi_yoy[cal] - 0.03)
            - 6.0 * (gdp_yoy[cal] - 0.02)
            + 0.9 * np.log1p(age)
            - 0.015 * age
        )
        hazard = 1.0 / (1.0 + np.exp(-logit))
        default = (rng.random(n_loans) < hazard) & alive
        idx = np.flatnonzero(alive)
        parts.append(
            pd.DataFrame(
                {
                    "loan_id": idx,
                    "months_on_book": age,
                    "dpd_months": 0,
                    "credit_score": credit_score[idx],
                    "orig_ltv": ltv[idx],
                    "orig_dti": dti[idx],
                    "orig_rate": rate[idx],
                    "unemployment": unemployment[cal[idx]],
                    "hpi_yoy": hpi_yoy[cal[idx]],
                    "gdp_yoy": gdp_yoy[cal[idx]],
                    "default_next": default[idx].astype(int),
                    "true_hazard": hazard[idx],
                }
            )
        )
        alive &= ~default
        alive &= rng.random(n_loans) > 0.005  # prepayment: the loan leaves the sample
    return pd.concat(parts, ignore_index=True)


def make_lgd_dataset(n=6000, seed=0):
    """Resolved defaults with a known LGD process (clipped to [0, 1])."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "orig_ltv": rng.uniform(50, 97, n),
            "credit_score": rng.normal(700, 50, n).clip(500, 830),
            "months_on_book": rng.integers(6, 84, n),
            "hpi_yoy": rng.normal(0.02, 0.06, n),
            "unemployment": rng.normal(5.5, 1.5, n),
            "orig_dti": rng.uniform(15, 50, n),
            "orig_rate": rng.uniform(0.03, 0.07, n),
        }
    )
    logit = (
        -1.4
        + 0.035 * (df["orig_ltv"] - 80)
        - 0.004 * (df["credit_score"] - 700)
        - 4.0 * df["hpi_yoy"]
        + 0.1 * (df["unemployment"] - 5.5)
    )
    mean_lgd = 1.0 / (1.0 + np.exp(-logit))
    df["lgd"] = (mean_lgd + rng.normal(0, 0.20, n)).clip(0, 1)
    df["true_lgd"] = mean_lgd
    return df
