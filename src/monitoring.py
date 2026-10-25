"""Model monitoring: PSI over time, PD backtest, stage migration.

Traffic lights: PSI below 0.10 green, below 0.25 amber, otherwise red. Backtest: z-score of observed
defaults against expected defaults (sum of predicted PDs), |z| up to 1.96 green, up to 2.58 amber,
otherwise red. These thresholds are conventions, justify them in the documentation.
"""
import numpy as np
import pandas as pd

from src.validation import psi


def psi_flag(value):
    if value < 0.10:
        return "green"
    return "amber" if value < 0.25 else "red"


def z_flag(z):
    if abs(z) <= 1.96:
        return "green"
    return "amber" if abs(z) <= 2.58 else "red"


def psi_over_time(df, value_col, time_col, ref_periods=12, bins=10, min_obs=200):
    """PSI of `value_col` in each period after the reference window, against the reference window."""
    periods = np.sort(df[time_col].unique())
    if len(periods) <= ref_periods:
        raise ValueError(f"Need more than {ref_periods} periods, found {len(periods)}")
    reference = df.loc[df[time_col].isin(periods[:ref_periods]), value_col].dropna().to_numpy()
    rows = []
    for period in periods[ref_periods:]:
        values = df.loc[df[time_col] == period, value_col].dropna().to_numpy()
        if len(values) < min_obs:
            continue
        value = psi(reference, values, bins)
        rows.append({"period": period, "n": len(values), "psi": value, "flag": psi_flag(value)})
    return pd.DataFrame(rows, columns=["period", "n", "psi", "flag"])


# Drift that is expected by construction: loans age and amortise, and macro variables follow the economy.
EXPECTED_DRIFT = frozenset({"months_on_book", "balance_ratio", "unemployment", "hpi_yoy", "gdp_yoy"})


def feature_psi(df, features, time_col, ref_periods=12, recent_periods=6, bins=10, expected=EXPECTED_DRIFT):
    """PSI of each feature: reference window against the most recent periods.

    Features whose drift is expected (age, balance, macro) get the flag "expected" instead of a traffic light.
    """
    periods = np.sort(df[time_col].unique())
    if len(periods) < ref_periods + recent_periods:
        raise ValueError("Not enough periods for the reference and recent windows")
    ref = df[df[time_col].isin(periods[:ref_periods])]
    recent = df[df[time_col].isin(periods[-recent_periods:])]
    rows = []
    for f in features:
        a, b = ref[f].dropna().to_numpy(), recent[f].dropna().to_numpy()
        if len(a) == 0 or len(b) == 0 or np.nanstd(a) == 0:
            continue
        value = psi(a, b, bins)
        flag = "expected" if f in expected else psi_flag(value)
        rows.append({"feature": f, "psi": value, "flag": flag})
    table = pd.DataFrame(rows, columns=["feature", "psi", "flag"])
    return table.sort_values("psi", ascending=False, ignore_index=True)


def add_quarter(df, col="report_month", out_col="quarter"):
    return df.assign(**{out_col: pd.to_datetime(df[col]).dt.to_period("Q").astype(str)})


def backtest_by_period(df, pd_col, target_col, period_col, default_correlation=0.01):
    """Predicted vs observed defaults per period, with z-scores and a traffic light.

    z_binomial treats every observation as independent. Defaults are correlated (they share the economy), so
    the variance is raised with an equicorrelation term: Var = sum s_i^2 + rho * ((sum s_i)^2 - sum s_i^2)
    with s_i = sqrt(p_i (1 - p_i)) and rho = `default_correlation` (an assumption; 0.005 to 0.03 is common
    for mortgages). The flag uses the adjusted z. Use one observation per loan and period (for example
    quarter-end snapshots), otherwise repeated rows of the same loan inflate n.
    """
    rows = []
    for period, g in df.groupby(period_col):
        p = g[pd_col].to_numpy(dtype=float)
        expected = float(p.sum())
        observed = float(g[target_col].sum())
        sigma = np.sqrt(p * (1.0 - p))
        var_independent = float((sigma**2).sum())
        variance = var_independent + default_correlation * (float(sigma.sum()) ** 2 - var_independent)
        z_binomial = (observed - expected) / np.sqrt(var_independent) if var_independent > 0 else np.nan
        z = (observed - expected) / np.sqrt(variance) if variance > 0 else np.nan
        rows.append(
            {
                period_col: period,
                "n": len(g),
                "mean_pd": float(p.mean()),
                "observed_rate": observed / len(g),
                "expected_defaults": expected,
                "observed_defaults": observed,
                "rel_error": observed / expected - 1.0 if expected > 0 else np.nan,
                "z_binomial": z_binomial,
                "z": z,
                "flag": "n/a" if np.isnan(z) else z_flag(z),
            }
        )
    return pd.DataFrame(rows)


def stage_transitions(df, include_exit=True):
    """Counts of stage changes between consecutive months. `df` needs loan_id, report_month, stage."""
    d = df[["loan_id", "report_month", "stage"]].copy()
    d["report_month"] = pd.to_datetime(d["report_month"])
    nxt = d.rename(columns={"stage": "next_stage"}).assign(
        report_month=lambda x: x["report_month"] - pd.DateOffset(months=1)
    )
    merged = d.merge(nxt, on=["loan_id", "report_month"], how="left")
    last = d["report_month"].max()
    if include_exit:
        gone = merged["next_stage"].isna() & (merged["report_month"] < last)
        merged["next_stage"] = merged["next_stage"].astype(object)
        merged.loc[gone, "next_stage"] = "exit"
    merged = merged.dropna(subset=["next_stage"])
    counts = merged.groupby(["stage", "next_stage"], as_index=False).size().rename(columns={"size": "n"})
    return counts


def migration_matrix(counts):
    """Counts and row-normalised probabilities from a (stage, next_stage, n) table."""
    c = counts.pivot_table(index="stage", columns="next_stage", values="n", aggfunc="sum", fill_value=0)
    stages = [1, 2, 3]
    cols = stages + (["exit"] if "exit" in c.columns else [])
    c = c.reindex(index=stages, columns=cols, fill_value=0)
    probs = c.div(c.sum(axis=1).replace(0, np.nan), axis=0)
    return c, probs
