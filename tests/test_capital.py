import numpy as np
import pandas as pd

from src.capital import capital_sensitivity, el_vs_ecl, scenario_metrics, stress_test
from src.rwa import rwa


def _loan_ecl():
    rows = []
    for scenario, pd_mult, lgd_add in (("base", 1.0, 0.0), ("downside", 2.0, 0.10)):
        for stage, ead, pd_, lgd, ecl in [
            (1, 100_000.0, 0.01, 0.20, 150.0),
            (1, 200_000.0, 0.02, 0.25, 500.0),
            (2, 150_000.0, 0.05, 0.30, 1_800.0),
            (3, 120_000.0, 1.00, 0.30, 36_000.0),
        ]:
            rows.append(
                {
                    "scenario": scenario,
                    "stage": stage,
                    "ead": ead,
                    "pd_12m": min(pd_ * pd_mult, 1.0) if stage < 3 else 1.0,
                    "lgd": lgd + lgd_add,
                    "ecl": ecl * (pd_mult + lgd_add * 3),
                }
            )
    return pd.DataFrame(rows)


def test_rwa_excludes_defaulted_exposures_and_matches_the_formula():
    df = _loan_ecl()
    m = scenario_metrics(df, "base")
    perf = df[(df["scenario"] == "base") & (df["stage"] < 3)]
    expected = rwa(perf["pd_12m"], perf["lgd"], perf["ead"]).sum()
    assert np.isclose(m["rwa"], expected)
    assert np.isclose(m["ead_performing"], 450_000.0)
    assert np.isclose(m["ead_total"], 570_000.0)
    assert np.isclose(m["rwa_density"], expected / 450_000.0)


def test_downside_has_higher_rwa_el_and_ecl():
    df = _loan_ecl()
    base, down = scenario_metrics(df, "base"), scenario_metrics(df, "downside")
    assert down["rwa"] > base["rwa"]
    assert down["el_irb"] > base["el_irb"]
    assert down["ecl_total"] > base["ecl_total"]


def test_pd_floor_and_lgd_floor_raise_rwa_for_very_safe_loans():
    df = pd.DataFrame(
        [{"scenario": "base", "stage": 1, "ead": 1e5, "pd_12m": 0.0001, "lgd": 0.02, "ecl": 1.0}]
    )
    low = scenario_metrics(df, "base", lgd_floor=0.0)
    high = scenario_metrics(df, "base", lgd_floor=0.10)
    assert high["rwa"] > low["rwa"]
    assert low["rwa"] > 0  # the PD floor keeps RWA above zero


def test_el_vs_ecl_shortfall_arithmetic():
    m = scenario_metrics(_loan_ecl(), "base")
    r = el_vs_ecl(m)
    assert np.isclose(r["shortfall"], m["el_irb"] - m["ecl_performing"])
    assert np.isclose(r["ecl_performing"], 150 + 500 + 1_800)


def test_stress_test_arithmetic():
    df = _loan_ecl()
    base, down = scenario_metrics(df, "base"), scenario_metrics(df, "downside")
    s = stress_test(base, down, 0.12)
    assert np.isclose(s["capital"], 0.12 * base["rwa"])
    assert np.isclose(s["extra_provisions"], down["ecl_total"] - base["ecl_total"])
    assert np.isclose(
        s["capital_ratio_after"], (s["capital"] - s["extra_provisions"]) / down["rwa"]
    )
    assert s["change_pp"] < 0  # the stress lowers the ratio


def test_capital_sensitivity_is_monotonic_in_starting_ratio():
    df = _loan_ecl()
    table = capital_sensitivity(scenario_metrics(df, "base"), scenario_metrics(df, "downside"))
    assert len(table) == 4
    assert np.all(np.diff(table["capital_ratio_after"]) > 0)
