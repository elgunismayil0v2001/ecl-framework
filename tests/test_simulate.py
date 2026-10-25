import os
import tempfile

import numpy as np
import pandas as pd

from src.freddie_schema import LAYOUTS, count_fields
from src.load_freddie import validate_inputs
from src.load_macro import build_macro_table, read_fred_csv
from src.simulate import simulate, to_freddie_files, write_fred_csvs

_CACHE = {}


def _sim(n=3000, seed=0):
    if (n, seed) not in _CACHE:
        _CACHE[(n, seed)] = simulate(n, 120, seed)
    return _CACHE[(n, seed)]


def test_simulation_is_reproducible():
    a_loans, a_perf = simulate(300, 120, 5)
    b_loans, b_perf = simulate(300, 120, 5)
    assert a_loans.equals(b_loans) and a_perf.equals(b_perf)
    c_loans, _ = simulate(300, 120, 6)
    assert not a_loans.equals(c_loans)


def test_files_have_the_real_field_counts_and_pass_the_loader_checks():
    loans, perf = _sim()
    for layout, (n_orig, n_perf) in (("v1", (32, 32)), ("v2", (31, 35))):
        with tempfile.TemporaryDirectory() as d:
            orig_path, perf_path = to_freddie_files(loans, perf, d, layout)
            assert count_fields(orig_path) == n_orig
            assert count_fields(perf_path) == n_perf
            _, _, detected = validate_inputs(str(orig_path), str(perf_path))
            assert detected == layout


def test_every_loan_has_a_consecutive_history_that_ends_with_one_exit():
    loans, perf = _sim()
    perf = perf.sort_values(["i", "month"])
    for _, g in list(perf.groupby("i"))[:300]:
        assert (np.diff(g["month"].to_numpy()) == 1).all()
        assert (g["zero"].iloc[:-1] == "").all()  # only the last row can carry a zero balance code
    last_rows = perf.groupby("i").tail(1)
    still_alive = last_rows["zero"] == ""
    assert (last_rows.loc[still_alive, "month"] == 119).all()  # unfinished loans run to the end of the data


def test_defaults_are_absorbing_until_liquidation_and_losses_are_in_range():
    _, perf = _sim()
    perf = perf.sort_values(["i", "month"])
    for _, g in perf[perf["dpd"] >= 3].groupby("i"):
        assert (g["dpd"].to_numpy() >= 3).all()
    liq = perf[perf["loss"].notna()]
    lgd = liq["loss"] / liq["removal"]
    assert len(liq) > 100
    assert lgd.between(0, 1).all()
    assert 0.08 < lgd.mean() < 0.50


def test_file_loss_sign_follows_the_layout():
    loans, perf = _sim()
    for layout in ("v1", "v2"):
        with tempfile.TemporaryDirectory() as d:
            _, perf_path = to_freddie_files(loans, perf, d, layout)
            rows = pd.read_csv(perf_path, sep="|", header=None, dtype=str)
        losses = pd.to_numeric(rows[21], errors="coerce").dropna()  # actual loss is field 22 in both layouts
        losses = losses[losses != 0]  # a zero loss has no sign
        assert len(losses) > 0
        assert (np.sign(losses) == LAYOUTS[layout]["loss_sign"]).all()


def test_risk_drivers_have_the_right_direction():
    loans, perf = _sim(n=6000)
    ever = set(perf.loc[perf["dpd"] >= 3, "i"])
    loans = loans.assign(defaulted=loans["i"].isin(ever))
    by_score = loans.groupby(pd.qcut(loans["score"], 4, duplicates="drop"), observed=True)["defaulted"].mean()
    assert by_score.iloc[0] > 1.5 * by_score.iloc[-1]  # lowest score quartile vs highest
    by_ltv = loans.groupby(pd.cut(loans["ltv"], [0, 75, 100]), observed=True)["defaulted"].mean()
    assert by_ltv.iloc[1] > by_ltv.iloc[0]
    assert 0.04 < loans["defaulted"].mean() < 0.25


def test_the_2020_recession_shows_up_in_defaults():
    _, perf = _sim(n=6000)
    new_def = perf[perf["dpd"] == 3].assign(year=lambda d: 2015 + d["month"] // 12)
    per_year = new_def.groupby("year")["i"].nunique()
    live = perf[perf["zero"] == ""].assign(year=lambda d: 2015 + d["month"] // 12)
    exposed = live.groupby("year")["i"].nunique()
    rate = per_year / exposed
    assert rate[2020] > 2 * rate[2019]


def test_fred_files_read_back_into_a_complete_macro_table():
    with tempfile.TemporaryDirectory() as d:
        write_fred_csvs(d)
        series = {
            name: read_fred_csv(os.path.join(d, f"{sid}.csv"), sid)
            for name, sid in (("u", "UNRATE"), ("h", "CSUSHPINSA"), ("g", "GDPC1"))
        }
    table = build_macro_table(series["u"], series["h"], series["g"])
    full = table.dropna()
    assert full["month"].min() <= pd.Timestamp("2016-03-01")
    assert full["month"].max() >= pd.Timestamp("2024-10-01")
    peak = table.loc[table["unemployment"].idxmax()]
    assert pd.Timestamp("2020-03-01") <= peak["month"] <= pd.Timestamp("2020-06-01")
    assert peak["unemployment"] > 10
    covid_gdp = table.set_index("month").loc["2020-06-01", "gdp_yoy"]
    assert covid_gdp < -0.05
