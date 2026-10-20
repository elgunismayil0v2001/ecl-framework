import numpy as np
import pandas as pd

from src.ecl import marginal_pd
from src.lifetime_pd import (
    HazardModel,
    cumulative_pd,
    lifetime_curves,
    make_scenarios,
    project_hazards,
    scenario_ordering_ok,
)
from src.synthetic import make_hazard_dataset

LAST_MACRO = {"unemployment": 5.0, "hpi_yoy": 0.03, "gdp_yoy": 0.02}


def _fit():
    df = make_hazard_dataset(n_loans=4000, months=48, seed=0)
    return HazardModel().fit(df), df


def _loans(df, n=200):
    first = df[df["months_on_book"] == 12].head(n).copy()
    first["term_months"] = 360
    return first


def test_coefficient_signs_match_the_true_process():
    model, _ = _fit()
    c = model.coef_
    assert c["unemployment"] > 0
    assert c["hpi_yoy"] < 0
    assert c["gdp_yoy"] < 0
    assert c["credit_score"] < 0
    assert c["orig_ltv"] > 0


def test_average_predicted_hazard_matches_observed_default_rate():
    model, df = _fit()
    predicted = model.predict_hazard(df).mean()
    observed = df["default_next"].mean()
    assert abs(predicted / observed - 1) < 0.05


def test_hazard_by_age_bucket_is_calibrated():
    model, df = _fit()
    df = df.assign(pred=model.predict_hazard(df), bucket=pd.cut(df["months_on_book"], [-1, 11, 23, 35, 48]))
    t = df.groupby("bucket", observed=True).agg(pred=("pred", "mean"), obs=("default_next", "mean"))
    assert (np.abs(t["pred"] / t["obs"] - 1) < 0.25).all()


def test_scenario_profile_starts_at_last_values_and_ramps():
    paths = make_scenarios(LAST_MACRO, horizon=60)
    down = paths["downside"]
    assert down["unemployment"].iloc[0] == 5.0
    assert np.isclose(down["unemployment"].iloc[12], 8.0)  # +3 pp at the end of the ramp
    assert np.isclose(down["unemployment"].iloc[23], 8.0)  # held
    assert down["unemployment"].iloc[59] == 5.0  # reverted
    assert (paths["base"]["unemployment"] == 5.0).all()
    assert paths["upside"]["unemployment"].min() < 5.0


def test_cumulative_pd_is_monotonic_bounded_and_matches_marginal_sum():
    model, df = _fit()
    loans = _loans(df)
    h = project_hazards(model, loans, make_scenarios(LAST_MACRO, 120)["base"])
    cum = cumulative_pd(h)
    assert np.all(np.diff(cum, axis=1) >= -1e-12)
    assert cum.min() >= 0 and cum.max() <= 1
    assert np.isclose(cum[0, -1], marginal_pd(h[0]).sum())


def test_hazards_are_zero_after_contractual_maturity():
    model, df = _fit()
    loans = _loans(df, n=20)
    loans["term_months"] = loans["months_on_book"] + 6
    h = project_hazards(model, loans, make_scenarios(LAST_MACRO, 24)["base"])
    assert np.all(h[:, 6:] == 0)
    assert np.all(h[:, :6] > 0)


def test_downside_above_base_above_upside():
    model, df = _fit()
    loans = _loans(df)
    curves = lifetime_curves(model, loans, make_scenarios(LAST_MACRO, 120), 120)
    assert scenario_ordering_ok(curves)
    assert set(curves["scenario"]) == {"base", "upside", "downside"}
    for _, g in curves.groupby("scenario"):
        assert np.all(np.diff(g["cum_pd"].to_numpy()) >= -1e-12)


def test_ordering_check_fails_when_the_model_has_no_macro_features():
    df = make_hazard_dataset(n_loans=2000, months=36, seed=2)
    model = HazardModel(features=["credit_score", "orig_ltv"]).fit(df)
    loans = _loans(df, n=50)
    curves = lifetime_curves(model, loans, make_scenarios(LAST_MACRO, 24), 24)
    assert not scenario_ordering_ok(curves)


def test_missing_macro_columns_are_dropped_by_the_model():
    df = make_hazard_dataset(n_loans=2000, months=36, seed=3)
    df[["unemployment", "hpi_yoy", "gdp_yoy"]] = np.nan
    model = HazardModel().fit(df)
    assert not {"unemployment", "hpi_yoy", "gdp_yoy"} & set(model.features_)
    assert model.predict_hazard(df).shape == (len(df),)
