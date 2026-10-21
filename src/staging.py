"""IFRS 9 staging in Python, mirroring sql/04_staging.sql, plus SICR threshold sensitivity.

Stage 3: 90+ days past due (default).
Stage 2: 30+ days past due (backstop), or a significant increase in credit risk (SICR):
         12-month PD at least `rel_threshold` times the PD at origination AND at least
         `abs_threshold` higher in absolute terms.
Stage 1: everything else.
All thresholds are assumptions: justify them with the sensitivity table.
"""
import numpy as np
import pandas as pd

DEFAULT_REL = 2.0
DEFAULT_ABS = 0.005


def assign_stage(df, rel_threshold=DEFAULT_REL, abs_threshold=DEFAULT_ABS, dpd_backstop=1, default_dpd=3):
    pd_now = df["pd_12m"].to_numpy(dtype=float)
    pd_orig = df["pd_12m_orig"].to_numpy(dtype=float)
    dpd = df["dpd_months"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pd_now / np.where(pd_orig == 0, np.nan, pd_orig)
    sicr = (ratio >= rel_threshold) & ((pd_now - pd_orig) >= abs_threshold)
    stage = np.where(dpd >= default_dpd, 3, np.where((dpd >= dpd_backstop) | sicr, 2, 1))
    return stage.astype(int)


def sicr_sensitivity(df, rel_grid=(1.5, 2.0, 2.5, 3.0), abs_grid=(0.0025, 0.005, 0.01), ead_col="ead"):
    """Share of performing loans (by count and by exposure) in Stage 2 for each threshold pair."""
    performing = df[df["dpd_months"] < 3]
    rows = []
    for rel in rel_grid:
        for abs_ in abs_grid:
            stage = assign_stage(performing, rel, abs_)
            in2 = stage == 2
            share_ead = np.nan
            if ead_col in performing.columns and performing[ead_col].sum() > 0:
                share_ead = float(performing.loc[in2, ead_col].sum() / performing[ead_col].sum())
            rows.append(
                {
                    "rel_threshold": rel,
                    "abs_threshold": abs_,
                    "stage2_share_count": float(in2.mean()) if len(performing) else np.nan,
                    "stage2_share_ead": share_ead,
                }
            )
    return pd.DataFrame(rows)
