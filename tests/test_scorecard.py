import numpy as np

from src.gates import assert_gates, evaluate_pd_model
from src.pd_model import Scorecard, apply_logit_shift, calibration_shift, points
from src.splits import time_based_split
from src.synthetic import make_pd_dataset
from src.validation import gini

FEATURES = ["credit_score", "ltv", "dti"]


def _fit():
    df = make_pd_dataset(n=40000, seed=0)
    train, oot = time_based_split(df, "vintage", oot_periods=2, gap_periods=0)
    return Scorecard(FEATURES).fit(train, "default"), train, oot


def test_all_coefficients_are_positive():
    sc, _, _ = _fit()
    assert (sc.coef_ > 0).all()
    assert set(sc.selected_) == set(FEATURES)


def test_scorecard_passes_validation_gates_out_of_time():
    sc, train, oot = _fit()
    report = evaluate_pd_model(
        oot["default"].to_numpy(), sc.predict_proba(oot), sc.predict_proba(train)
    )
    assert_gates(report)


def test_scorecard_ranks_close_to_the_true_probabilities():
    sc, _, oot = _fit()
    p = sc.predict_proba(oot)
    assert gini(oot["default"], p) > 0.8 * gini(oot["default"], oot["p_true"])


def test_probabilities_are_valid_and_riskier_loans_score_higher():
    sc, _, oot = _fit()
    p = sc.predict_proba(oot)
    assert np.all((p > 0) & (p < 1))
    risky = oot.assign(credit_score=520.0, ltv=95.0, dti=48.0)
    safe = oot.assign(credit_score=800.0, ltv=45.0, dti=15.0)
    assert sc.predict_proba(risky).mean() > 5 * sc.predict_proba(safe).mean()


def test_useless_feature_is_dropped_by_iv_filter():
    df = make_pd_dataset(n=40000, seed=0)
    df["noise"] = np.random.default_rng(9).normal(size=len(df))
    sc = Scorecard(FEATURES + ["noise"]).fit(df, "default")
    assert "noise" in sc.dropped_
    assert "noise" not in sc.selected_
    table = sc.iv_table()
    assert set(table["feature"]) == set(FEATURES + ["noise"])


def test_wrong_sign_feature_is_removed():
    df = make_pd_dataset(n=40000, seed=0)
    rng = np.random.default_rng(10)
    # A feature that is almost a copy of credit_score gets a negative weight when added with noise
    df["score_copy"] = df["credit_score"] + rng.normal(0, 1, len(df))
    sc = Scorecard(["credit_score", "score_copy", "ltv", "dti"], iv_min=0.0).fit(df, "default")
    assert (sc.coef_ > 0).all()


def test_scorecard_handles_missing_feature_values():
    df = make_pd_dataset(n=20000, seed=1)
    df.loc[df.index[::7], "dti"] = np.nan
    sc = Scorecard(FEATURES).fit(df, "default")
    p = sc.predict_proba(df)
    assert not np.isnan(p).any()


def test_points_scaling_follows_pdo_rule():
    odds_50 = 1.0 / 51.0  # p such that good:bad odds are 50:1
    assert np.isclose(points(odds_50), 600.0, atol=1e-6)
    odds_100 = 1.0 / 101.0
    assert np.isclose(points(odds_100) - points(odds_50), 20.0, atol=1e-6)
    assert points(0.2) < points(0.02)


def test_calibration_shift_matches_target_rate_and_keeps_ranking():
    rng = np.random.default_rng(11)
    p = rng.uniform(0.005, 0.10, 5000)
    delta = calibration_shift(p, 0.08)
    shifted = apply_logit_shift(p, delta)
    assert np.isclose(shifted.mean(), 0.08, atol=1e-6)
    assert np.array_equal(np.argsort(p), np.argsort(shifted))
