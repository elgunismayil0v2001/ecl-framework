import numpy as np
import pandas as pd

from src.ead import ead_paths
from src.ecl import compute_ecl, marginal_pd
from src.ecl_pipeline import run_checks
from src.ecl_portfolio import (
    build_loan_ecl,
    coverage_by_stage,
    lifetime_horizon,
    marginal_pd_matrix,
    portfolio_ecl,
    weight_sensitivity,
    weighted_total,
)
from src.lgd import LGDModel
from src.lifetime_pd import HazardModel, make_scenarios, project_hazards
from src.synthetic import make_hazard_dataset, make_lgd_dataset

LAST_MACRO = {"unemployment": 5.0, "hpi_yoy": 0.03, "gdp_yoy": 0.02}
WEIGHTS = {"base": 0.6, "upside": 0.1, "downside": 0.3}
HORIZON = 120
_CACHE = {}


def _models():
    if "m" not in _CACHE:
        hz = make_hazard_dataset(n_loans=3000, months=48, seed=1)
        _CACHE["m"] = (HazardModel().fit(hz), LGDModel().fit(make_lgd_dataset(n=6000, seed=1)), hz)
    return _CACHE["m"]


def _portfolio(n=300, seed=0):
    _, _, hz = _models()
    rng = np.random.default_rng(seed)
    loans = hz[hz["months_on_book"] == 12].head(n).copy().reset_index(drop=True)
    loans["loan_id"] = np.arange(len(loans))
    loans["report_month"] = pd.Timestamp("2020-12-01")
    loans["term_months"] = 360
    loans["current_upb"] = rng.uniform(100_000, 300_000, len(loans))
    loans["stage"] = rng.choice([1, 2, 3], len(loans), p=[0.70, 0.25, 0.05])
    loans["dpd_months"] = loans["stage"].map({1: 0, 2: 1, 3: 3})
    return loans


def _ecl(loans=None, lgd=True, cpr=0.06):
    hazard, lgd_model, _ = _models()
    loans = _portfolio() if loans is None else loans
    scenarios = make_scenarios(LAST_MACRO, HORIZON)
    return build_loan_ecl(loans, hazard, lgd_model if lgd else None, scenarios, cpr=cpr, default_lgd=0.25)


def test_marginal_pd_matrix_matches_single_loan_function():
    h = np.random.default_rng(0).uniform(0.001, 0.02, (4, 30))
    m = marginal_pd_matrix(h)
    for i in range(4):
        assert np.allclose(m[i], marginal_pd(h[i]))


def test_vectorised_ecl_equals_the_single_loan_formula():
    hazard, _, _ = _models()
    loans = _portfolio(n=40)
    loans = loans[loans["stage"] < 3].head(6).reset_index(drop=True)
    path = make_scenarios(LAST_MACRO, 60)["base"]
    h = project_hazards(hazard, loans, path, 60)
    eir = loans["orig_rate"].to_numpy()
    ead = ead_paths(loans["current_upb"], eir, 348, 60, cpr=0.06)
    lgd = np.linspace(0.1, 0.4, len(loans))
    months = np.where(loans["stage"] == 1, 12, 60).astype(float)
    vec = portfolio_ecl(h, lgd, ead, eir, months)
    for i in range(len(loans)):
        single = compute_ecl(h[i], lgd[i], ead[i], eir=eir[i], horizon=int(months[i]))
        assert np.isclose(vec[i], single)


def test_stage_3_ecl_is_lgd_times_ead():
    out = _ecl()
    s3 = out[out["stage"] == 3]
    assert len(s3) > 0
    assert np.allclose(s3["ecl"], s3["lgd"] * s3["ead"])
    assert (s3["pd_12m"] == 1.0).all()


def test_ecl_is_bounded_by_lgd_times_ead_and_non_negative():
    out = _ecl()
    assert (out["ecl"] >= 0).all()
    assert (out["ecl"] <= out["lgd"] * out["ead"] + 1e-6).all()


def test_stage_2_lifetime_ecl_is_above_stage_1_twelve_month_ecl_for_the_same_loan():
    loans = _portfolio()
    loans = loans[loans["stage"] < 3].copy()
    as_stage_1 = _ecl(loans.assign(stage=1))
    as_stage_2 = _ecl(loans.assign(stage=2))
    assert (as_stage_2["ecl"].to_numpy() >= as_stage_1["ecl"].to_numpy() - 1e-9).all()
    assert as_stage_2["ecl"].sum() > 3 * as_stage_1["ecl"].sum()


def test_coverage_rises_with_stage_and_checks_pass():
    out = _ecl()
    cov = coverage_by_stage(out, WEIGHTS)
    w = cov[cov["scenario"] == "weighted"].set_index("stage")["coverage"]
    assert w[3] > w[2] > w[1] > 0
    checks = run_checks(out, WEIGHTS)
    assert all(checks.values()), checks


def test_scenarios_order_total_ecl_and_weights_sit_between():
    out = _ecl()
    totals = out.groupby("scenario")["ecl"].sum()
    assert totals["downside"] > totals["base"] > totals["upside"]
    weighted = weighted_total(out, WEIGHTS)
    assert totals["base"] < weighted < totals["downside"]


def test_downside_lgd_is_higher_than_base_lgd():
    out = _ecl()
    lgd = out.pivot_table(index="loan_id", columns="scenario", values="lgd")
    assert (lgd["downside"] > lgd["base"]).all()
    assert (lgd["base"] > lgd["upside"]).all()


def test_constant_lgd_is_used_without_an_lgd_model():
    out = _ecl(lgd=False)
    assert np.allclose(out["lgd"], 0.25)


def test_weight_sensitivity_rises_with_downside_weight():
    out = _ecl()
    sets = {
        "base only": {"base": 1.0, "upside": 0.0, "downside": 0.0},
        "default": WEIGHTS,
        "heavy downside": {"base": 0.4, "upside": 0.1, "downside": 0.5},
    }
    table = weight_sensitivity(out, sets).set_index("weights")
    assert table.loc["base only", "ecl"] < table.loc["default", "ecl"] < table.loc["heavy downside", "ecl"]


def test_weights_must_sum_to_one():
    out = _ecl()
    try:
        weighted_total(out, {"base": 0.5, "upside": 0.1, "downside": 0.3})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_matured_loans_have_zero_ecl_and_horizon_is_capped():
    loans = _portfolio(n=20)
    loans = loans[loans["stage"] < 3].copy()
    loans["term_months"] = loans["months_on_book"]  # nothing left to run
    out = _ecl(loans.assign(stage=1))
    assert np.allclose(out["ecl"], 0.0)
    assert lifetime_horizon(_portfolio(), cap=100) == 100
    assert lifetime_horizon(_portfolio(), cap=1000) == 348
