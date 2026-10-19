"""Load FRED macro series into the `macro` table.

Download the three series as CSV from fred.stlouisfed.org and save them as
<dir>/UNRATE.csv, <dir>/CSUSHPINSA.csv and <dir>/GDPC1.csv.

Units: unemployment in percent; hpi_yoy and gdp_yoy as fractions (0.03 = 3%).
Quarterly GDP growth is carried forward to the months of the quarter (at most 2 months).
`lag_months` shifts all series forward in time so a model only uses information that was
available `lag_months` earlier (avoids look-ahead bias).
"""
from pathlib import Path

import pandas as pd

SERIES_IDS = {"unemployment": "UNRATE", "hpi": "CSUSHPINSA", "gdp": "GDPC1"}


def read_fred_csv(path, series_id):
    df = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    value_col = series_id if series_id in df.columns else df.columns[1]
    values = pd.to_numeric(df[value_col], errors="coerce")  # FRED writes "." for missing values
    s = pd.Series(values.to_numpy(), index=pd.to_datetime(df[date_col]), name=series_id)
    return s.dropna().sort_index()


def build_macro_table(unemployment, hpi, gdp, scenario="base", lag_months=0):
    start = min(unemployment.index.min(), hpi.index.min(), gdp.index.min())
    end = max(unemployment.index.max(), hpi.index.max(), gdp.index.max())
    months = pd.date_range(start.replace(day=1), end, freq="MS")

    unemp = unemployment.reindex(months)
    hpi_m = hpi.reindex(months)
    hpi_yoy = hpi_m / hpi_m.shift(12) - 1.0
    gdp_yoy = (gdp / gdp.shift(4) - 1.0).reindex(months).ffill(limit=2)

    table = pd.DataFrame(
        {
            "month": months,
            "scenario": scenario,
            "unemployment": unemp.to_numpy(),
            "hpi_yoy": hpi_yoy.to_numpy(),
            "gdp_yoy": gdp_yoy.to_numpy(),
        }
    )
    value_cols = ["unemployment", "hpi_yoy", "gdp_yoy"]
    if lag_months:
        table[value_cols] = table[value_cols].shift(lag_months)
    return table.dropna(how="all", subset=value_cols).reset_index(drop=True)


def load_macro(con, fred_dir, scenario="base", lag_months=0):
    fred_dir = Path(fred_dir)
    series = {
        name: read_fred_csv(fred_dir / f"{sid}.csv", sid) for name, sid in SERIES_IDS.items()
    }
    table = build_macro_table(
        series["unemployment"], series["hpi"], series["gdp"], scenario, lag_months
    )
    con.register("df_macro", table)
    con.execute("DELETE FROM macro WHERE scenario = ?", [scenario])
    con.execute(
        """INSERT INTO macro
           SELECT
               CAST(month AS DATE),
               scenario,
               CASE WHEN isnan(unemployment) THEN NULL ELSE unemployment END,
               CASE WHEN isnan(hpi_yoy) THEN NULL ELSE hpi_yoy END,
               CASE WHEN isnan(gdp_yoy) THEN NULL ELSE gdp_yoy END
           FROM df_macro"""
    )
    con.unregister("df_macro")
    return len(table)
