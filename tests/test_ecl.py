import numpy as np

from src.ecl import compute_ecl, marginal_pd, stage_horizon, weighted_ecl


def test_marginal_pds_sum_to_cumulative_pd():
    h = np.full(12, 0.01)
    assert np.isclose(marginal_pd(h).sum(), 1 - 0.99**12)


def test_zero_pd_gives_zero_ecl():
    assert compute_ecl(np.zeros(24), 0.4, 100.0) == 0.0


def test_closed_form_without_discounting():
    h = np.full(12, 0.01)
    expected = 0.25 * 100.0 * (1 - 0.99**12)
    assert np.isclose(compute_ecl(h, 0.25, 100.0, eir=0.0), expected)


def test_discounting_reduces_ecl():
    h = np.full(36, 0.01)
    assert compute_ecl(h, 0.25, 100.0, eir=0.05) < compute_ecl(h, 0.25, 100.0, eir=0.0)


def test_lifetime_ecl_exceeds_12_month_ecl():
    h = np.full(60, 0.01)
    assert compute_ecl(h, 0.25, 100.0, horizon=60) > compute_ecl(h, 0.25, 100.0, horizon=12)


def test_ecl_increases_with_pd_and_lgd():
    h = np.full(24, 0.01)
    base = compute_ecl(h, 0.25, 100.0)
    assert compute_ecl(h * 1.5, 0.25, 100.0) > base
    assert compute_ecl(h, 0.35, 100.0) > base


def test_stage_horizons():
    assert stage_horizon(1, 200) == 12
    assert stage_horizon(2, 200) == 200
    assert stage_horizon(3, 200) == 200
    assert stage_horizon(1, 6) == 6


def test_weighted_ecl_and_scenario_ordering():
    h = np.full(24, 0.01)
    ecl = {
        "upside": compute_ecl(h * 0.7, 0.25, 100.0),
        "base": compute_ecl(h, 0.25, 100.0),
        "downside": compute_ecl(h * 1.6, 0.30, 100.0),
    }
    assert ecl["upside"] < ecl["base"] < ecl["downside"]
    w = {"upside": 0.1, "base": 0.6, "downside": 0.3}
    assert ecl["base"] < weighted_ecl(ecl, w) < ecl["downside"]


def test_weights_must_sum_to_one():
    try:
        weighted_ecl({"a": 1.0, "b": 2.0}, {"a": 0.5, "b": 0.6})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
