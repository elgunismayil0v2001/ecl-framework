import numpy as np
import pandas as pd

from src.monitoring import (
    add_quarter,
    backtest_by_period,
    feature_psi,
    migration_matrix,
    psi_flag,
    psi_over_time,
    stage_transitions,
    z_flag,
)


def _scores(n_months=24, per_month=1000, shift_after=None, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(n_months):
        mean = 0.0 if shift_after is None or m < shift_after else 1.0
        frames.append(
            pd.DataFrame(
                {
                    "report_month": pd.Timestamp("2019-01-01") + pd.DateOffset(months=m),
                    "pd_12m": 1 / (1 + np.exp(-(rng.normal(mean, 1.0, per_month) - 3.5))),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_flags():
    assert psi_flag(0.05) == "green" and psi_flag(0.15) == "amber" and psi_flag(0.40) == "red"
    assert z_flag(1.0) == "green" and z_flag(-2.0) == "amber" and z_flag(3.0) == "red"


def test_stable_scores_give_green_psi():
    table = psi_over_time(_scores(), "pd_12m", "report_month", ref_periods=12)
    assert len(table) == 12
    assert (table["flag"] == "green").all()


def test_drifting_scores_are_flagged_red():
    table = psi_over_time(_scores(shift_after=18), "pd_12m", "report_month", ref_periods=12)
    early = table[table["period"] < pd.Timestamp("2020-07-01")]
    late = table[table["period"] >= pd.Timestamp("2020-07-01")]
    assert (early["flag"] == "green").all()
    assert (late["flag"] == "red").all()


def test_psi_over_time_needs_enough_periods_and_skips_small_months():
    try:
        psi_over_time(_scores(n_months=10), "pd_12m", "report_month", ref_periods=12)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    small = psi_over_time(_scores(per_month=100), "pd_12m", "report_month", min_obs=200)
    assert len(small) == 0


def test_feature_psi_ranks_the_drifting_feature_first():
    rng = np.random.default_rng(1)
    months = pd.date_range("2019-01-01", periods=24, freq="MS")
    df = pd.DataFrame(
        {
            "report_month": np.repeat(months, 500),
            "stable": rng.normal(0, 1, 24 * 500),
            "drifting": rng.normal(0, 1, 24 * 500) + np.repeat(np.linspace(0, 3, 24), 500),
            "constant": 1.0,
        }
    )
    table = feature_psi(df, ["stable", "drifting", "constant"], "report_month", 12, 6)
    assert list(table["feature"])[0] == "drifting"
    assert table.iloc[0]["flag"] == "red"
    assert "constant" not in set(table["feature"])


def test_backtest_is_green_for_calibrated_and_red_for_biased_pds():
    rng = np.random.default_rng(2)
    n = 40_000
    p = rng.uniform(0.01, 0.10, n)
    df = pd.DataFrame(
        {
            "report_month": np.repeat(pd.date_range("2019-01-01", periods=8, freq="3MS"), n // 8),
            "pd_12m": p,
            "default_12m": rng.binomial(1, p),
        }
    )
    df = add_quarter(df)
    good = backtest_by_period(df, "pd_12m", "default_12m", "quarter")
    assert len(good) == 8
    assert (good["flag"] != "red").all()
    biased = backtest_by_period(df.assign(pd_12m=df["pd_12m"] * 0.5), "pd_12m", "default_12m", "quarter")
    assert (biased["flag"] == "red").all()
    assert (biased["z"] > 0).all()  # under-prediction: more observed than expected


def test_backtest_columns_and_arithmetic():
    df = pd.DataFrame({"q": ["a"] * 4, "p": [0.1, 0.1, 0.2, 0.2], "y": [0, 1, 0, 0]})
    t = backtest_by_period(df, "p", "y", "q").iloc[0]
    assert np.isclose(t["expected_defaults"], 0.6)
    assert t["observed_defaults"] == 1.0
    assert np.isclose(t["z"], 0.4 / np.sqrt(0.09 + 0.09 + 0.16 + 0.16))


def test_add_quarter_labels():
    df = pd.DataFrame({"report_month": pd.to_datetime(["2020-01-01", "2020-05-01", "2020-12-01"])})
    assert list(add_quarter(df)["quarter"]) == ["2020Q1", "2020Q2", "2020Q4"]


def _panel():
    rows = [
        # loan 1: stays in stage 1, then leaves (prepaid) after month 3
        (1, "2020-01-01", 1), (1, "2020-02-01", 1), (1, "2020-03-01", 1),
        # loan 2: 1 -> 2 -> 3 and stays to the end of the data
        (2, "2020-01-01", 1), (2, "2020-02-01", 2), (2, "2020-03-01", 3), (2, "2020-04-01", 3),
        # loan 3 covers the last month so the data ends in April
        (3, "2020-01-01", 1), (3, "2020-02-01", 1), (3, "2020-03-01", 2), (3, "2020-04-01", 1),
    ]
    return pd.DataFrame(rows, columns=["loan_id", "report_month", "stage"])


def test_stage_transitions_count_changes_and_exits():
    counts = stage_transitions(_panel()).set_index(["stage", "next_stage"])["n"]
    assert counts[(1, 1)] == 3  # loan 1 twice, loan 3 once
    assert counts[(1, 2)] == 2  # loan 2 (January) and loan 3 (February)
    assert counts[(2, 3)] == 1
    assert counts[(3, 3)] == 1
    assert counts[(2, 1)] == 1  # loan 3 cured
    assert counts[(1, "exit")] == 1  # loan 1 left after March; loans at the last month are not exits


def test_migration_matrix_rows_sum_to_one_and_fill_missing_stages():
    counts = stage_transitions(_panel())
    c, p = migration_matrix(counts)
    assert list(c.index) == [1, 2, 3]
    assert "exit" in c.columns
    assert np.allclose(p.sum(axis=1).dropna(), 1.0)
    only_stage_1 = pd.DataFrame({"stage": [1], "next_stage": [1], "n": [5]})
    c2, p2 = migration_matrix(only_stage_1)
    assert c2.loc[1, 1] == 5 and c2.loc[3].sum() == 0
    assert np.isnan(p2.loc[3, 1])


def test_expected_drift_features_are_not_flagged_red():
    rng = np.random.default_rng(4)
    months = pd.date_range("2019-01-01", periods=24, freq="MS")
    df = pd.DataFrame(
        {
            "report_month": np.repeat(months, 500),
            "months_on_book": rng.normal(0, 1, 24 * 500) + np.repeat(np.linspace(0, 5, 24), 500),
            "credit_score": rng.normal(0, 1, 24 * 500) + np.repeat(np.linspace(0, 5, 24), 500),
        }
    )
    t = feature_psi(df, ["months_on_book", "credit_score"], "report_month", 12, 6).set_index("feature")
    assert t.loc["months_on_book", "flag"] == "expected"  # a closed book ages by construction
    assert t.loc["credit_score", "flag"] == "red"
    assert t.loc["months_on_book", "psi"] > 1.0  # the PSI value is still reported
