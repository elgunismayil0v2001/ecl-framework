import numpy as np

from src.validation import (
    calibration_in_the_large,
    calibration_table,
    gini,
    ks,
    psi,
    rank_order_corr,
)


def test_gini_perfect_and_random():
    assert np.isclose(gini([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.05, 20000)
    assert abs(gini(y, rng.random(20000))) < 0.1


def test_ks_perfect_separation():
    assert np.isclose(ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)


def test_psi_identical_is_zero_and_shift_is_detected():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-9
    assert psi(x, x + 1.0) > 0.25


def test_calibration_table_counts_and_totals():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.2, 1000)
    y = rng.binomial(1, p)
    table = calibration_table(y, p, bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 1000
    assert np.isclose((table["default_rate"] * table["n"]).sum(), y.sum())


def test_calibration_in_the_large():
    y = np.array([0, 0, 0, 1])
    assert np.isclose(calibration_in_the_large(y, np.full(4, 0.25)), 0.0)
    assert calibration_in_the_large(y, np.full(4, 0.5)) > 0.9


def test_rank_order_corr_high_for_true_probabilities():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.3, 50000)
    y = rng.binomial(1, p)
    assert rank_order_corr(y, p) > 0.9
