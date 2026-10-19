import os
import tempfile

import numpy as np
import pandas as pd

from src.load_macro import build_macro_table, read_fred_csv


def _write(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")


def test_read_fred_csv_handles_both_header_styles_and_missing_values():
    with tempfile.TemporaryDirectory() as d:
        new_style = os.path.join(d, "a.csv")
        _write(new_style, "observation_date,UNRATE", ["2020-01-01,3.5", "2020-02-01,.", "2020-03-01,4.4"])
        old_style = os.path.join(d, "b.csv")
        _write(old_style, "DATE,UNRATE", ["2020-01-01,3.5", "2020-03-01,4.4"])
        a = read_fred_csv(new_style, "UNRATE")
        b = read_fred_csv(old_style, "UNRATE")
    assert list(a.values) == [3.5, 4.4]
    assert a.equals(b)


def _series(start, periods, freq, first, growth):
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.Series(first * growth ** np.arange(periods), index=idx)


def test_yoy_growth_and_forward_filled_quarterly_gdp():
    unrate = _series("2018-01-01", 36, "MS", 4.0, 1.0)
    hpi = _series("2018-01-01", 36, "MS", 100.0, 1.005)
    gdp = _series("2018-01-01", 12, "QS", 100.0, 1.01)
    t = build_macro_table(unrate, hpi, gdp)
    row = t[t["month"] == "2019-06-01"].iloc[0]
    assert np.isclose(row["hpi_yoy"], 1.005**12 - 1)
    assert np.isclose(row["gdp_yoy"], 1.01**4 - 1)
    assert row["unemployment"] == 4.0
    mid_quarter = t[t["month"] == "2019-05-01"].iloc[0]
    assert np.isclose(mid_quarter["gdp_yoy"], 1.01**4 - 1)
    assert (t["scenario"] == "base").all()


def test_first_year_has_no_yoy_values():
    unrate = _series("2018-01-01", 24, "MS", 4.0, 1.0)
    hpi = _series("2018-01-01", 24, "MS", 100.0, 1.005)
    gdp = _series("2018-01-01", 8, "QS", 100.0, 1.01)
    t = build_macro_table(unrate, hpi, gdp)
    first = t[t["month"] == "2018-06-01"].iloc[0]
    assert np.isnan(first["hpi_yoy"])


def test_lag_months_shifts_values_forward():
    unrate = pd.Series(np.arange(24, dtype=float), index=pd.date_range("2018-01-01", periods=24, freq="MS"))
    hpi = _series("2018-01-01", 24, "MS", 100.0, 1.005)
    gdp = _series("2018-01-01", 8, "QS", 100.0, 1.01)
    base = build_macro_table(unrate, hpi, gdp, lag_months=0).set_index("month")
    lagged = build_macro_table(unrate, hpi, gdp, lag_months=3).set_index("month")
    assert lagged.loc["2019-06-01", "unemployment"] == base.loc["2019-03-01", "unemployment"]
