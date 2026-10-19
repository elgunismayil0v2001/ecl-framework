import numpy as np
from src.rwa import capital_requirement, rwa, expected_loss
from src.validation import psi, ks


def test_capital_requirement_positive_and_increasing_in_pd():
    k = capital_requirement([0.001, 0.01, 0.05], 0.25)
    assert np.all(k > 0)
    assert np.all(np.diff(k) > 0)


def test_zero_lgd_gives_zero_rwa():
    assert rwa(0.02, 0.0, 100.0) == 0.0


def test_rwa_scales_with_ead():
    assert np.isclose(rwa(0.02, 0.25, 200.0), 2 * rwa(0.02, 0.25, 100.0))


def test_expected_loss():
    assert np.isclose(expected_loss(0.02, 0.25, 100.0), 0.5)


def test_psi_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-9


def test_ks_perfect_separation():
    assert np.isclose(ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
