# Data

Raw data is not stored in git (`data/raw/` is ignored). Download it yourself.

## Freddie Mac Single-Family Loan-Level Dataset

1. Register for free on the Freddie Mac Single-Family Loan-Level Dataset page.
2. Download the sample files for one or two origination years and unzip them into `data/raw/`.
   File names look like `sample_orig_YYYY.txt` (origination) and `sample_svcg_YYYY.txt` (monthly performance).
3. Check the column layout against the user guide of your file version (`src/freddie_schema.py`).
   The loader stops with a clear error if a file has a different number of columns.

```bash
make data-freddie ORIG='data/raw/sample_orig_*.txt' PERF='data/raw/sample_svcg_*.txt'
```

Cleaning assumptions (see `src/load_freddie.py`): origination date is the first payment month minus one
month; 9999 / 999 codes become NULL; delinquency status `RA` is treated as 90+ days past due; `XX` rows
are dropped. Credit-event zero balance codes (short sale, REO) are kept but do not change the default
definition (90+ days past due).

## FRED macro data

Download these three series as CSV from fred.stlouisfed.org and save them in `data/raw/fred/`:

| Series | File | Meaning |
|---|---|---|
| UNRATE | `UNRATE.csv` | Unemployment rate (percent) |
| CSUSHPINSA | `CSUSHPINSA.csv` | S&P CoreLogic Case-Shiller house price index (used for year-on-year growth) |
| GDPC1 | `GDPC1.csv` | Real GDP (quarterly, used for year-on-year growth) |

Pass `--lag-months 3` (or edit the Makefile variable) if you want the model to use macro data with a
publication lag, which avoids look-ahead bias.

## Smoke test without real data

```bash
make data-synthetic
```
