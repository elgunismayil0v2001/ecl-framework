# Model documentation: IFRS 9 ECL and capital adequacy framework

Version 0.1.0. Items marked TBD depend on the data you load and are filled after running the pipelines.

## 1. Purpose and scope
This document describes a framework that estimates IFRS 9 expected credit loss (ECL) and an illustrative capital
position for a residential mortgage portfolio: PD, LGD and EAD models, staging, ECL under three macro scenarios,
IRB-style risk-weighted assets, a stress test, validation gates and monitoring.
It is a portfolio project built on public data, not the production model of a bank. Intended use: to show model
development, validation and monitoring practice. Out of scope: see section 9.

## 2. Data
Data source: TBD (real Freddie Mac data, or simulated data from `make sim`; if simulated, every result describes the
simulation and not the mortgage market).
Sources: Freddie Mac Single-Family Loan-Level Dataset (origination and monthly performance files) and three FRED
series (UNRATE, CSUSHPINSA, GDPC1). Origination years loaded and number of loans: TBD.
File layouts: the loader detects the layout before Release 47 (32 + 32 fields) and from Release 47, July 2026
(31 + 35 fields) and converts the realised loss to positive = loss.
Cleaning (`src/load_freddie.py`): origination date is the first payment month minus one month; codes 9999 and 999
become missing; interest rate is converted to a fraction; delinquency status RA is treated as 90+ days past due and
XX rows are dropped.
Default definition: 90+ days past due. Credit-event zero balance codes are kept for the LGD sample.
Windows: the PD target is default within the next 12 months. Observations in the last 12 months of the data are
excluded from development because their 12-month outcome window is right-censored, but they are scored for
staging and ECL. The reporting date is the last month in the data.
Data quality: up to 17 automatic checks (`make quality`): duplicate keys, orphan rows, date ranges, missing shares,
negative delinquency, right-censoring, plausible default rate, macro coverage. Results: TBD.

## 3. PD model

### 3.1 Target and sample
Target: default (90+ days past due) within the next 12 months, measured for each performing loan-month.
Observations in the last 12 months of the data are excluded from model development because their outcome
window is not fully observed. They are still scored for staging and ECL.

### 3.2 Development and validation samples
Time-based split: training on the early months, out-of-time (OOT) validation on the last months, with a
12-month gap between them so the outcome windows do not overlap.
Training period: TBD. Gap: TBD. OOT period: TBD. Observations and default rates by sample: TBD.

### 3.3 Scorecard (main model)
1. Weight of evidence (WoE) binning per feature: one bin per value for features with at most 10 distinct values
   (for example months in arrears), equal-count starting bins otherwise, minimum bin share 1%,
   monotonic bad rate enforced by merging adjacent bins, missing values in their own bin.
   WoE = ln(share of bads / share of goods), so higher WoE means higher risk.
2. Features with information value (IV) below 0.02 are dropped. IV table: `reports/tables/pd_iv_table.csv`.
3. Logistic regression (L2, C = 1) on WoE values.
4. Sign check: all coefficients must be positive. The feature with the most negative coefficient is
   removed and the model is refitted until all signs are positive.
5. Points scaling for reporting: 600 points at 50:1 good:bad odds, 20 points to double the odds.
Macro variables are not scorecard features: with one macro path per calendar month they let the model memorise
the cycle and make the score drift. The economy enters through the lifetime PD model and the scenarios.
Final features and coefficients: TBD. Binning tables: `reports/tables/pd_woe_bins.csv`.

### 3.4 Benchmark (LightGBM)
Gradient boosting with monotonic constraints in the expected risk direction (for example PD does not rise
with credit score). Used only as a benchmark of the scorecard. Feature importance (mean absolute SHAP value):
`reports/figures/pd_benchmark_importance.png`.

### 3.5 Validation results (out-of-time)
Gates (thresholds are project assumptions, justify them here): Gini at least 0.30, KS at least 0.20,
calibration error (average PD vs observed default rate) at most 15%, rank order correlation of
decile PD vs default rate at least 0.90, PSI train vs OOT at most 0.25.

<!-- PD_VALIDATION:START -->
| metric | benchmark | scorecard | gate |
|---|---|---|---|
| calibration_rel_error | 0.3448 | 0.1723 | <= 0.15 |
| gini | 0.4810 | 0.5045 | >= 0.3 |
| ks | 0.3545 | 0.3520 | >= 0.2 |
| psi_train_vs_test | 0.5863 | 0.1200 | <= 0.25 |
| rank_order_corr | 0.9879 | 0.8788 | >= 0.9 |
<!-- PD_VALIDATION:END -->

### 3.6 Calibration and lifetime PD
If the calibration gate fails because of the economic cycle, apply a logit intercept shift estimated on a
calibration window inside the training data (never on the OOT sample). Lifetime PD term structure: see
section 3.7.

### 3.7 Lifetime PD (roll-rate model)
Delinquency moves one month at a time (current, 30, 60, 90+ days late), so a one-month default probability is
zero for a current loan. A single-step hazard model therefore gives a lifetime PD that is far too low (in the
simulated data it gave 0.95% against about 24%). The lifetime PD comes from a roll-rate (Markov) model
(`src/rollrate.py`, data from `sql/08_hazard_data.sql`):
- States: 0 current, 1 = 30 days late, 2 = 60 days late, 3 = 90+ days late (default, absorbing).
- One multinomial logistic regression per starting state (0, 1, 2) for the state next month. Covariates: log age
  and age, credit score, LTV, DTI, rate and the macro variables (unemployment, house price growth, GDP growth).
  Loans that prepay or are liquidated leave the sample (censoring).
- Projection: the state probabilities are rolled forward month by month with macro values from the scenario;
  the probability of first reaching state 3 in each month, given no earlier default, is the monthly hazard. Loan
  characteristics stay constant, there are no transitions after contractual maturity, and prepayment is not a
  competing risk (it enters through the EAD path). Lifetime PD = 1 - product of (1 - hazard).
Macro scenarios (illustrative assumptions, relative to the last observed values): shock ramps up over 12 months,
holds for 12 months, reverts over 24 months. Downside: unemployment +3.0 pp, house price growth -10 pp, GDP
growth -4 pp. Upside: unemployment -1.0 pp, house price growth +3 pp, GDP growth +1 pp. Base: values held flat.
Scenario weights: base 60%, upside 10%, downside 30% (assumption).
Coefficients: `reports/tables/hazard_coefficients.csv` (one column per starting state, per standard deviation).
Curves: `reports/figures/lifetime_pd_curves.png`.

<!-- LIFETIME:START -->
| scenario | 12 months | 60 months | 120 months |
|---|---|---|---|
| base | 3.49 | 12.39 | 16.55 |
| downside | 5.32 | 25.19 | 28.49 |
| upside | 3.15 | 10.41 | 14.71 |

Cumulative PD in %.
<!-- LIFETIME:END -->

## 4. LGD and EAD models
### 4.1 LGD
Sample: resolved defaults with a realised loss (zero balance codes 02, 03, 09, 15, 16), `sql/07_lgd_data.sql`.
Realised LGD = actual loss / UPB at removal, clipped to [0, 1].
Model: fractional logit (weighted logistic regression) on LTV at origination, credit score, loan age,
house price growth and unemployment at resolution, DTI and rate. Predictions are between 0 and 1.
Limitations: cured loans and unresolved defaults are not in the sample (workout bias); the sign convention of
the loss field must be checked against the user guide; no downturn adjustment or regulatory floor is applied
(the model has a `floor` parameter if one is needed).
Results (average realised vs predicted LGD by LTV band): TBD. Table: `reports/tables/lgd_by_ltv_band.csv`.

### 4.2 EAD
EAD at month t is the expected balance at the start of month t from the amortisation schedule, re-amortised
each month, with a constant prepayment rate (default 6% CPR, assumption). Prepayment is not modelled as a
competing risk in the PD hazards. Portfolio path: `reports/tables/ead_portfolio_path.csv`.

## 5. Staging
Stage 3: 90+ days past due. Stage 2: 30+ days past due (backstop) or a significant increase in credit risk
(SICR): 12-month PD at least 2.0 times the PD at the first observation of the loan and at least 0.5 pp higher.
Stage 1: all other loans. `pd_12m_orig` is the scorecard PD of the same loan with a clean payment history (no arrears
now or in the last 12 months) at its current age. Comparing with the PD at the first observation was tried first:
it put almost every loan in Stage 2, because the PD rises with loan age (seasoning), which is not a deterioration.
The SICR rule therefore measures behavioural deterioration. Macro information enters through the scenarios, not
through staging. Thresholds are assumptions. Sensitivity of the Stage 2 share to the thresholds
(relative 1.5 to 3.0, absolute 0.25 to 1.0 pp): `reports/tables/sicr_sensitivity.csv`.
Chosen thresholds and justification: TBD. Stage shares at the reporting date: TBD.

## 6. ECL calculation
ECL = LGD x sum over months t of marginal PD(t) x EAD(t) x discount factor(t).
Horizon: 12 months for Stage 1, remaining contractual term for Stage 2. Stage 3: ECL = LGD x EAD.
PD: lifetime roll-rate model (section 3.7) under three macro scenarios. EAD: amortisation schedule with 6% CPR
(section 4.2). LGD: model of section 4.1, evaluated with each scenario's macro values 12 months ahead.
Discount rate: contractual rate of each loan (proxy for the effective interest rate).
Probability weights: base 60%, upside 10%, downside 30% (assumption). Sensitivity of total ECL to the weights:
`reports/tables/ecl_weight_sensitivity.csv`.
Coverage by stage, per scenario and probability-weighted (`reports/tables/ecl_coverage_by_stage.csv`):

<!-- ECL_COVERAGE:START -->
| scenario | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| base | 0.310 | 6.537 | 12.551 |
| downside | 0.822 | 12.356 | 20.252 |
| upside | 0.231 | 5.374 | 10.643 |
| weighted | 0.456 | 8.166 | 14.671 |

Coverage = ECL / EAD in %.
<!-- ECL_COVERAGE:END -->
Checks run by the pipeline: coverage rises from Stage 1 to Stage 3, downside ECL above base above upside,
ECL between zero and LGD x EAD for every loan, weights sum to one.

## 7. Capital adequacy
RWA: Basel IRB formula for residential mortgages (asset correlation 0.15, confidence level 99.9%, PD floor
0.05%), with the 12-month PD of each scenario and the model LGD. Defaulted exposures are excluded.
Capital is an assumption: a starting capital ratio (12%) on base RWA. Stress test: extra provisions
(downside ECL minus base ECL) reduce capital one for one; no profit, tax, dividends or management actions.
Expected loss (IRB) is compared with IFRS 9 ECL on performing exposures. The two differ because IFRS 9 is
point-in-time and forward-looking, while IRB parameters are meant to be through-the-cycle (PD) and downturn (LGD),
and because our PD and LGD are the same model estimates in both calculations.
Tables: `reports/tables/capital_by_scenario.csv`, `el_vs_ecl.csv`, `stress_test.csv`, `capital_sensitivity.csv`.

<!-- CAPITAL:START -->
| scenario | ead_total | rwa | rwa_density | el_irb | ecl_total |
|---|---|---|---|---|---|
| base | 912,384,355.7500 | 226,591,445.1600 | 0.2553 | 3,764,204.7935 | 6,795,826.7522 |
| upside | 912,384,355.7500 | 178,736,291.2884 | 0.2014 | 2,878,900.4840 | 5,461,285.1247 |
| downside | 912,384,355.7500 | 477,181,026.5392 | 0.5376 | 9,266,414.5661 | 14,048,203.1176 |

Stress test by assumed starting capital ratio:

| capital_ratio_start | capital_ratio_after | change_pp |
|---|---|---|
| 0.0800 | 0.0228 | -5.7210 |
| 0.1000 | 0.0323 | -6.7713 |
| 0.1200 | 0.0418 | -7.8216 |
| 0.1500 | 0.0560 | -9.3970 |
<!-- CAPITAL:END -->
Limitations: no downturn LGD or regulatory LGD floor, no defaulted-exposure RWA, no other risk types, no
off-balance-sheet exposures, parameters must be checked against the current Basel / CRR text.

## 8. Validation and monitoring
Validation before use (`src/gates.py`, `make pd`): out-of-time Gini, KS, calibration, rank order and PSI against the
thresholds in section 3.5. Backtesting and monitoring (`make monitor`):
- PSI of the PD score by month against a 12-month reference window (green below 0.10, amber below 0.25, red above).
- PSI of each model feature, reference window against the last 6 months. Age, balance and macro variables drift by
  construction (a closed book ages), so they are labelled "expected" instead of getting a traffic light.
- PD backtest by quarter: expected defaults (sum of predicted PDs) against observed defaults, z-score with
  traffic lights (|z| up to 1.96 green, up to 2.58 amber, above red).
- Stage migration matrix between consecutive months, including exits (prepayment or liquidation).
- Consistency of the scorecard 12-month PD (used for staging) with the roll-rate model 12-month PD (used for ECL and RWA).
Tables: `reports/tables/monitoring_*.csv`. Charts: `reports/figures/monitoring_*.png`.
Findings and actions: TBD.

## 9. Limitations and assumptions
Data and definitions
- Public mortgage data from one lender programme: not a bank's own portfolio. Results are illustrative.
- Default is 90+ days past due only. Credit-event zero balance codes (short sale, REO) do not change the definition.
- Origination date is the first payment month minus one month. PD at origination is the PD at the first
  observation of each loan.
Models
- Two PD models: the scorecard (staging, validation) and the roll-rate model (ECL, RWA, scenarios). They can
  disagree; the monitoring pipeline reports their consistency. The scorecard has no macro variables, so it does
  not follow the cycle, while the roll-rate model does.
- A closed book (all loans originated in a few years) ages as a whole: score and feature PSI show drift, and the
  default rate rises with seasoning. Use an open book for meaningful drift monitoring.
- Macro scenarios are illustrative shocks, not forecasts. Macro coefficients are estimated on a short history.
- LGD uses resolved defaults only (workout bias), no downturn adjustment and no regulatory floor.
- Prepayment is a constant 6% CPR and is not a competing risk in the roll-rate model, which overstates lifetime PD
  somewhat compared with realised default incidence.
- Stage 3 ECL is LGD x EAD without discounting. Defaulted exposures are not in RWA.
- Scenario weights, SICR thresholds, validation thresholds and the starting capital ratio are assumptions.
Capital
- IRB formula for residential mortgages only, with parameters to be checked against the current Basel / CRR text.
- The stress test ignores profit, tax, dividends and management actions. Capital is assumed, not observed.
Not covered: off-balance-sheet exposures, other risk types, governance, IT controls, a formal model approval process.

## 10. Appendix
Variable dictionary
| Variable | Meaning |
|---|---|
| credit_score | Credit score at origination |
| orig_ltv, orig_dti | Loan-to-value and debt-to-income at origination (percent) |
| orig_rate | Interest rate at origination (fraction) |
| months_on_book | Months since origination |
| dpd_months | Months past due at the observation date |
| max_dpd_last_12m | Worst delinquency in the last 12 months |
| balance_ratio | Current balance / original balance |
| unemployment, hpi_yoy, gdp_yoy | Unemployment rate (percent), house price growth, GDP growth (fractions) |
| default_12m | 1 if the loan reaches 90+ days past due in the next 12 months |
| next_dpd_months | Months past due in the next month (roll-rate model); default_next = 1 if it is 3 or more |
| pd_12m, pd_12m_orig | 12-month PD now and at the first observation |
| stage | IFRS 9 stage (1, 2, 3) |
