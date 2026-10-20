import numpy as np

from src.ead import annuity_payment, ead_paths


def test_annuity_payment_matches_textbook_value():
    assert np.isclose(annuity_payment(100_000.0, 0.06, 360), 599.55, atol=0.01)


def test_zero_rate_payment_is_straight_line():
    assert np.isclose(annuity_payment(1200.0, 0.0, 12), 100.0)


def test_balance_never_increases_for_a_fixed_rate_loan():
    paths = ead_paths([100_000.0, 250_000.0], [0.06, 0.04], [360, 300], horizon=360, cpr=0.0)
    assert np.all(np.diff(paths, axis=1) <= 1e-9)


def test_loan_is_fully_repaid_at_maturity_without_prepayment():
    path = ead_paths(100_000.0, 0.06, 120, horizon=122, cpr=0.0)[0]
    assert path[0] == 100_000.0
    assert path[119] > 0  # last scheduled month still has a balance at its start
    assert path[120] < 1e-6 and path[121] < 1e-6


def test_zero_rate_path_is_linear():
    path = ead_paths(1200.0, 0.0, 12, horizon=12)[0]
    assert np.allclose(path, 1200.0 * (1 - np.arange(12) / 12.0))


def test_prepayment_lowers_the_expected_balance():
    no_prepay = ead_paths(100_000.0, 0.05, 240, horizon=60, cpr=0.0)[0]
    prepay = ead_paths(100_000.0, 0.05, 240, horizon=60, cpr=0.10)[0]
    assert np.all(prepay[1:] < no_prepay[1:])
    smm = 1 - (1 - 0.10) ** (1 / 12)
    assert np.isclose(prepay[1], (no_prepay[1]) * (1 - smm))


def test_matured_loans_have_zero_exposure_and_shapes_are_right():
    paths = ead_paths([50_000.0, 50_000.0], 0.05, [6, 60], horizon=24)
    assert paths.shape == (2, 24)
    assert np.allclose(paths[0, 7:], 0.0)
    assert paths[1, -1] > 0
