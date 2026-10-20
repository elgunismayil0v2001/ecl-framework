import numpy as np
import pandas as pd

from src.splits import time_based_split


def _frame(n_periods=10):
    return pd.DataFrame({"t": np.repeat(np.arange(n_periods), 5), "x": np.arange(n_periods * 5)})


def test_split_leaves_a_gap_between_train_and_oot():
    train, oot = time_based_split(_frame(10), "t", oot_periods=2, gap_periods=3)
    assert sorted(oot["t"].unique()) == [8, 9]
    assert sorted(train["t"].unique()) == [0, 1, 2, 3, 4]
    assert oot["t"].min() - train["t"].max() > 3 - 1


def test_split_has_no_overlap_and_keeps_all_rows_of_each_period():
    train, oot = time_based_split(_frame(10), "t", oot_periods=3, gap_periods=2)
    assert set(train["t"]).isdisjoint(set(oot["t"]))
    assert len(train) == 5 * len(train["t"].unique())


def test_split_works_with_month_dates():
    months = pd.date_range("2018-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"report_month": np.repeat(months, 3)})
    train, oot = time_based_split(df, "report_month", oot_periods=6, gap_periods=12)
    assert train["report_month"].max() == pd.Timestamp("2019-06-01")
    assert oot["report_month"].min() == pd.Timestamp("2020-07-01")


def test_split_raises_when_history_is_too_short():
    try:
        time_based_split(_frame(10), "t", oot_periods=6, gap_periods=12)
    except ValueError as e:
        assert "Not enough periods" in str(e)
    else:
        raise AssertionError("expected ValueError")
