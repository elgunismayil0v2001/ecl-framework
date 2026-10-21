import numpy as np
import pandas as pd

from src.staging import assign_stage, sicr_sensitivity


def _frame(rows):
    return pd.DataFrame(rows, columns=["dpd_months", "pd_12m", "pd_12m_orig", "ead"])


def test_stage_rules_with_default_thresholds():
    df = _frame(
        [
            (3, 0.50, 0.01, 100),  # 90+ DPD: Stage 3
            (4, np.nan, np.nan, 100),  # defaulted, no score: still Stage 3
            (1, 0.01, 0.01, 100),  # 30 DPD backstop: Stage 2
            (0, 0.010, 0.004, 100),  # PD x2.5 and +0.6 pp: SICR, Stage 2
            (0, 0.010, 0.006, 100),  # PD x1.7: not enough, Stage 1
            (0, 0.003, 0.001, 100),  # PD x3 but only +0.2 pp: Stage 1
            (0, 0.020, 0.000, 100),  # no PD at origination: ratio undefined, Stage 1
            (0, np.nan, 0.01, 100),  # no current PD: Stage 1
        ]
    )
    assert list(assign_stage(df)) == [3, 3, 2, 2, 1, 1, 1, 1]


def test_stricter_thresholds_never_increase_stage_2():
    rng = np.random.default_rng(0)
    orig = rng.uniform(0.002, 0.02, 2000)
    df = pd.DataFrame(
        {
            "dpd_months": rng.choice([0, 0, 0, 1], 2000),
            "pd_12m_orig": orig,
            "pd_12m": orig * rng.lognormal(0.2, 0.6, 2000),
            "ead": rng.uniform(1e5, 3e5, 2000),
        }
    )
    base = (assign_stage(df, 2.0, 0.005) == 2).sum()
    assert (assign_stage(df, 3.0, 0.005) == 2).sum() <= base
    assert (assign_stage(df, 2.0, 0.01) == 2).sum() <= base
    assert (assign_stage(df, 1.5, 0.0025) == 2).sum() >= base


def test_sensitivity_table_shape_and_monotonicity():
    rng = np.random.default_rng(1)
    orig = rng.uniform(0.002, 0.02, 3000)
    df = pd.DataFrame(
        {
            "dpd_months": np.zeros(3000),
            "pd_12m_orig": orig,
            "pd_12m": orig * rng.lognormal(0.3, 0.7, 3000),
            "ead": rng.uniform(1e5, 3e5, 3000),
        }
    )
    table = sicr_sensitivity(df)
    assert len(table) == 12
    for _, g in table.groupby("abs_threshold"):
        shares = g.sort_values("rel_threshold")["stage2_share_count"].to_numpy()
        assert np.all(np.diff(shares) <= 1e-12)
    assert table["stage2_share_ead"].between(0, 1).all()


def test_sensitivity_ignores_defaulted_loans():
    df = _frame([(3, 0.5, 0.01, 100), (0, 0.02, 0.005, 100)])
    table = sicr_sensitivity(df, rel_grid=(2.0,), abs_grid=(0.005,))
    assert table["stage2_share_count"].iloc[0] == 1.0  # one performing loan, and it is in Stage 2
