import numpy as np
import pandas as pd

from src.synthetic import make_pd_dataset
from src.woe import WoEBinner


def test_iv_is_positive_for_informative_feature_and_small_for_noise():
    df = make_pd_dataset(n=40000, seed=0)
    informative = WoEBinner().fit(df["credit_score"], df["default"])
    noise = WoEBinner().fit(np.random.default_rng(1).normal(size=len(df)), df["default"])
    assert informative.iv_ > 0.05
    assert noise.iv_ < 0.02


def test_woe_follows_risk_direction_and_bad_rate_is_monotonic():
    df = make_pd_dataset(n=40000, seed=0)
    b = WoEBinner(monotonic=True).fit(df["credit_score"], df["default"])
    rates = b.table_["bad_rate"].to_numpy()
    assert np.all(np.diff(rates) <= 1e-12)  # higher score, lower bad rate
    assert np.all(np.diff(b.woe_) <= 1e-12)
    low_score_woe = b.transform(pd.Series([520.0]))[0]
    high_score_woe = b.transform(pd.Series([820.0]))[0]
    assert low_score_woe > high_score_woe  # WoE is higher for riskier values


def test_monotonic_merging_removes_violations():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 100, 20000)
    p = 0.10 + 0.08 * (x > 50) - 0.05 * ((x > 20) & (x < 40))  # non-monotone pattern
    y = rng.binomial(1, p)
    b = WoEBinner(max_bins=10, monotonic=True).fit(x, y)
    diffs = np.diff(b.table_["bad_rate"].to_numpy())
    assert np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12)


def test_min_bin_share_is_respected():
    df = make_pd_dataset(n=20000, seed=2)
    b = WoEBinner(max_bins=20, min_bin_share=0.10, monotonic=False).fit(df["dti"], df["default"])
    assert (b.table_["n"] / b.table_["n"].sum()).min() >= 0.10 - 1e-9


def test_missing_values_get_their_own_bin_when_seen_in_training():
    df = make_pd_dataset(n=20000, seed=4)
    x = df["credit_score"].copy()
    x[::10] = np.nan
    b = WoEBinner().fit(x, df["default"])
    assert b.has_missing_
    assert "missing" in set(b.table_["bin"])
    out = b.transform(pd.Series([np.nan, 700.0]))
    assert out[0] == b.missing_woe_


def test_missing_at_transform_without_training_missing_gives_zero_woe():
    df = make_pd_dataset(n=5000, seed=5)
    b = WoEBinner().fit(df["credit_score"], df["default"])
    assert not b.has_missing_
    assert b.transform(pd.Series([np.nan]))[0] == 0.0


def test_constant_feature_has_single_bin_and_zero_iv():
    y = np.random.default_rng(0).binomial(1, 0.1, 1000)
    b = WoEBinner().fit(np.full(1000, 5.0), y)
    assert len(b.table_) == 1
    assert abs(b.iv_) < 1e-9


def test_transform_handles_unseen_extreme_values():
    df = make_pd_dataset(n=5000, seed=6)
    b = WoEBinner().fit(df["ltv"], df["default"])
    out = b.transform(pd.Series([-1e9, 1e9]))
    assert np.all(np.isfinite(out))
