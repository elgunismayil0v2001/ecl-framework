import numpy as np
import pandas as pd

from src.ecl import marginal_pd
from src.ecl_portfolio import build_loan_ecl
from src.lgd import LGDModel
from src.lifetime_pd import (
    HazardModel,
    cumulative_pd,
    lifetime_curves,
    make_scenarios,
    project_hazards,
    scenario_ordering_ok,
)
from src.rollrate import RollRateModel
from src.simulate import make_person_period
from src.synthetic import make_lgd_dataset

_CACHE = {}
LAST_MACRO = {"unemployment": 4.0, "hpi_yoy": 0.04, "gdp_yoy": 0.02}


def _data():
    if "d" not in _CACHE:
        _CACHE["d"] = make_person_period(n_loans=5000, seed=3)
    return _CACHE["d"]


def _model():
    if "m" not in _CACHE:
        _CACHE["m"] = RollRateModel().fit(_data())
    return _CACHE["m"]


def _loans(n=400, dpd=0, age=12):
    d = _data()
    rows = d[(d["months_on_book"] == age) & (d["dpd_months"] == 0)].head(n).copy()
    rows["dpd_months"] = dpd
    rows["term_months"] = 360
    return rows.reset_index(drop=True)


def test_one_step_hazard_is_structurally_zero_for_current_loans_but_roll_rate_model_is_not():
    d = _data()
    assert d.loc[d["dpd_months"] == 0, "next_dpd_months"].max() <= 1  # delinquency moves one step at a time
    hazard_model = HazardModel().fit(d.assign(default_next=(d["next_dpd_months"] >= 3).astype(int)))
    scenarios = make_scenarios(LAST_MACRO, 24)
    loans = _loans()
    one_step = cumulative_pd(project_hazards(hazard_model, loans.assign(dpd_months=0), scenarios["base"]))
    roll = cumulative_pd(_model().project(loans, scenarios["base"]))
    assert one_step[:, -1].mean() < 0.2 * roll[:, -1].mean()  # the one-step model collapses the PD


def test_coefficients_have_the_simulated_signs():
    c = _model().coef_
    assert c["credit_score"] < 0
    assert c["orig_ltv"] > 0
    assert c["unemployment"] > 0
    assert c["hpi_yoy"] < 0


def test_twelve_month_pd_matches_observed_default_rates():
    d = _data()
    d = d[(d["month"] <= 107) & (d["months_on_book"].between(12, 36)) & (d["dpd_months"] == 0)]
    sample = d.sample(3000, random_state=1).copy()
    sample["term_months"] = 360
    predicted = []
    for _, g in sample.groupby("month"):
        macro = {c: float(g[c].iloc[0]) for c in LAST_MACRO}
        path = pd.DataFrame({c: np.full(12, v) for c, v in macro.items()})  # macro held at its current value
        predicted.append(cumulative_pd(_model().project(g, path))[:, -1])
    predicted = np.concatenate(predicted).mean()
    observed = sample["default_12m"].mean()
    assert 0.6 < predicted / observed < 1.5


def test_delinquent_loans_are_much_riskier_than_current_loans():
    path = make_scenarios(LAST_MACRO, 12)["base"]
    pd0 = cumulative_pd(_model().project(_loans(dpd=0), path))[:, -1].mean()
    pd1 = cumulative_pd(_model().project(_loans(dpd=1), path))[:, -1].mean()
    pd2 = cumulative_pd(_model().project(_loans(dpd=2), path))[:, -1].mean()
    assert pd0 < pd1 < pd2
    assert pd2 > 0.5 and pd2 > 15 * pd0


def test_hazards_are_valid_cumulative_pd_is_monotonic_and_matches_marginal_sum():
    path = make_scenarios(LAST_MACRO, 120)["base"]
    h = _model().project(_loans(n=50), path)
    assert h.shape == (50, 120)
    assert h.min() >= 0 and h.max() <= 1
    cum = cumulative_pd(h)
    assert np.all(np.diff(cum, axis=1) >= -1e-12)
    assert np.isclose(cum[0, -1], marginal_pd(h[0]).sum())
    assert 0.02 < cum[:, -1].mean() < 0.5  # a lifetime PD of a few percent to tens of percent


def test_hazards_are_zero_after_contractual_maturity():
    loans = _loans(n=20)
    loans["term_months"] = loans["months_on_book"] + 6
    h = _model().project(loans, make_scenarios(LAST_MACRO, 24)["base"])
    assert np.all(h[:, 6:] == 0)  # nothing after maturity
    assert np.all(h[:, :2] == 0)  # a current loan cannot reach 90 days late in one or two months
    assert np.all(h[:, 2:6] > 0)


def test_downside_above_base_above_upside():
    curves = lifetime_curves(_model(), _loans(), make_scenarios(LAST_MACRO, 120), 120)
    assert scenario_ordering_ok(curves)


def test_constant_states_are_handled():
    d = _data().copy()
    d.loc[d["dpd_months"] == 2, "next_dpd_months"] = 2  # nobody ever leaves state 2
    model = RollRateModel().fit(d)
    h = model.project(_loans(n=10, dpd=2), make_scenarios(LAST_MACRO, 12)["base"])
    assert h.sum() == 0.0


def test_ecl_engine_accepts_the_roll_rate_model():
    loans = _loans(n=200)
    loans["loan_id"] = np.arange(len(loans))
    loans["report_month"] = pd.Timestamp("2022-01-01")
    loans["current_upb"] = 200_000.0
    loans["stage"] = np.where(np.arange(len(loans)) % 5 == 0, 2, 1)
    lgd = LGDModel().fit(make_lgd_dataset(n=4000, seed=2))
    out = build_loan_ecl(loans, _model(), lgd, make_scenarios(LAST_MACRO, 120))
    totals = out.groupby("scenario")["ecl"].sum()
    assert totals["downside"] > totals["base"] > totals["upside"] > 0
    cover = out[out["scenario"] == "base"].groupby("stage").apply(lambda g: g["ecl"].sum() / g["ead"].sum())
    assert cover[2] > cover[1]
