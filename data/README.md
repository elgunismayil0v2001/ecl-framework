# Data

Raw data is not stored in git (`data/raw/` is ignored). Download it yourself and create the folders:

```bash
mkdir -p data/raw/fred
```

## Freddie Mac Single-Family Loan-Level Dataset

1. Register for free and sign in to Clarity Data Intelligence from the Freddie Mac Single-Family Loan-Level
   Dataset page (freddiemac.com/research/datasets/sf-loanlevel-dataset).
2. In the Data Download section, download the sample files `sample_YYYY.zip` (a random sample of 50,000 loans
   per origination year). Start with one or two years, for example 2018 and 2019.
3. Unzip them into `data/raw/`. The result should be two text files per year, for example
   `sample_orig_2019.txt` (origination) and `sample_svcg_2019.txt` (monthly performance). Standard Dataset files
   from Release 47 (July 2026) are named `orig_YYYYQn.txt` and `perf_YYYYQn.txt`.
4. Check what you have: `ls data/raw`.

```bash
make data-freddie
```

The default file patterns cover both naming schemes. To use other names:
`make data-freddie ORIG='data/raw/my_orig_*.txt' PERF='data/raw/my_perf_*.txt'` (several patterns can be
separated by commas).

### File layouts

Release 47 (July 2026) changed the layout of the Standard Dataset files. The loader detects the layout from
the number of fields per line and stops with a clear error if it does not recognise the file:

| Layout | Origination fields | Performance fields |
|---|---|---|
| v1 (before Release 47) | 32 | 32 |
| v2 (Release 47 and later) | 31 | 35 |

Do not mix layouts in one load. The sign of the realised loss also changed: in v2 a loss is positive. The
loader converts both to "positive = loss" (v1 is inferred from the release notes). `make quality` checks that
the mean LGD of resolved defaults is plausible, which catches a wrong sign.
Count the fields of a file yourself with: `head -1 data/raw/your_file.txt | awk -F'|' '{print NF}'`.

### Cleaning assumptions (see `src/load_freddie.py`)

Origination date is the first payment month minus one month; 9999 / 999 codes become NULL; delinquency status
`RA` is treated as 90+ days past due and `XX` rows are dropped. Credit-event zero balance codes (short sale,
REO) are kept but do not change the default definition (90+ days past due).

## FRED macro data

Download three series as CSV into `data/raw/fred/` (no account needed):

```bash
curl -L -o data/raw/fred/UNRATE.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
curl -L -o data/raw/fred/CSUSHPINSA.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CSUSHPINSA"
curl -L -o data/raw/fred/GDPC1.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPC1"
```

| Series | File | Meaning |
|---|---|---|
| UNRATE | `UNRATE.csv` | Unemployment rate (percent) |
| CSUSHPINSA | `CSUSHPINSA.csv` | S&P CoreLogic Case-Shiller house price index (used for year-on-year growth) |
| GDPC1 | `GDPC1.csv` | Real GDP (quarterly, used for year-on-year growth) |

Check each file with `head -3 data/raw/fred/UNRATE.csv`: it should show a date column and the series values.
Pass `LAG=3` to `make data-freddie` if you want the model to use macro data with a publication lag, which avoids
look-ahead bias.

## If you cannot download the real data: simulated data

```bash
make sim
```

This writes Freddie Mac style origination and performance files (real file layout) and FRED style macro CSVs to
`data/sim/`, then loads them. The pipeline then runs unchanged: `make quality`, `make pd`, `make lifetime`,
`make ecl`, `make monitor`, `make report`. The data are SIMULATED (credit score, LTV, DTI, loan age,
unemployment and house prices drive defaults; a recession in 2020), so every result describes the simulation, not
the mortgage market. State this clearly wherever you show results, and switch to real data when you can.

## Smoke test without real data

```bash
make data-synthetic
```
