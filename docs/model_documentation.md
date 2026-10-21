# Model documentation: IFRS 9 ECL and capital adequacy framework

Write this like a bank model development document. Fill each section as you build.

## 1. Purpose and scope
Portfolio, products, intended use (provisioning and illustrative capital), what is out of scope.

## 2. Data
Sources, sample period, exclusions, data quality checks, default definition (90+ DPD), observation and performance windows.
Observations in the last 12 months of the data are excluded because their 12-month outcome window is right-censored.

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
1. Weight of evidence (WoE) binning per feature, equal-count starting bins, minimum bin share 5%,
   monotonic bad rate enforced by merging adjacent bins, missing values in their own bin.
   WoE = ln(share of bads / share of goods), so higher WoE means higher risk.
2. Features with information value (IV) below 0.02 are dropped. IV table: `reports/tables/pd_iv_table.csv`.
3. Logistic regression (L2, C = 1) on WoE values.
4. Sign check: all coefficients must be positive. The feature with the most negative coefficient is
   removed and the model is refitted until all signs are positive.
5. Points scaling for reporting: 600 points at 50:1 good:bad odds, 20 points to double the odds.
Final features and coefficients: TBD. Binning tables: `reports/tables/pd_woe_bins.csv`.

### 3.4 Benchmark (LightGBM)
Gradient boosting with monotonic constraints in the expected risk direction (for example PD does not rise
with credit score). Used only as a benchmark of the scorecard. Feature importance (mean absolute SHAP value):
`reports/figures/pd_benchmark_importance.png`.

### 3.5 Validation results (out-of-time)
Gates (thresholds are project assumptions, justify them here): Gini at least 0.30, KS at least 0.20,
calibration error (average PD vs observed default rate) at most 15%, rank order correlation of
decile PD vs default rate at least 0.90, PSI train vs OOT at most 0.25.

| Metric | Scorecard | Benchmark |
|---|---|---|
| Gini | TBD | TBD |
| KS | TBD | TBD |
| Calibration error | TBD | TBD |
| Rank order correlation | TBD | TBD |
| PSI train vs OOT | TBD | TBD |

### 3.6 Calibration and lifetime PD
If the calibration gate fails because of the economic cycle, apply a logit intercept shift estimated on a
calibration window inside the training data (never on the OOT sample). Lifetime PD term structure: see
section 3.7.

### 3.7 Lifetime PD (discrete-time hazard model)
Person-period data: one row per performing loan-month, target = default in the next month (`sql/08_hazard_data.sql`).
Loans that prepay or are liquidated leave the sample (prepayment is treated as censoring).
Model: logistic regression on log age, age, credit score, LTV, DTI, rate, months in arrears and macro variables
(unemployment, house price growth, GDP growth). Coefficients are reported per standard deviation and checked for
sign (unemployment positive, house price and GDP growth negative).
Projection: age increases one month per step, loan characteristics are constant, arrears are cured after the
first month, hazards are zero after contractual maturity. Lifetime PD = 1 - product of (1 - hazard).
Macro scenarios (illustrative assumptions, relative to the last observed values): shock ramps up over 12 months,
holds for 12 months, reverts over 24 months. Downside: unemployment +3.0 pp, house price growth -10 pp, GDP
growth -4 pp. Upside: unemployment -1.0 pp, house price growth +3 pp, GDP growth +1 pp. Base: values held flat.
Scenario weights: base 60%, upside 10%, downside 30% (assumption).
Results (cumulative PD at 12, 60 and 120 months by scenario): TBD. Curves: `reports/figures/lifetime_pd_curves.png`.

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
Stage 1: all other loans. `pd_12m_orig` is the PD at the first observation of each loan, used as a proxy for
PD at initial recognition. Thresholds are assumptions. Sensitivity of the Stage 2 share to the thresholds
(relative 1.5 to 3.0, absolute 0.25 to 1.0 pp): `reports/tables/sicr_sensitivity.csv`.
Chosen thresholds and justification: TBD. Stage shares at the reporting date: TBD.

## 6. ECL calculation
ECL = LGD x sum over months t of marginal PD(t) x EAD(t) x discount factor(t).
Horizon: 12 months for Stage 1, remaining contractual term for Stage 2. Stage 3: ECL = LGD x EAD.
PD: lifetime hazard model (section 3.7) under three macro scenarios. EAD: amortisation schedule with 6% CPR
(section 4.2). LGD: model of section 4.1, evaluated with each scenario's macro values 12 months ahead.
Discount rate: contractual rate of each loan (proxy for the effective interest rate).
Probability weights: base 60%, upside 10%, downside 30% (assumption). Sensitivity of total ECL to the weights:
`reports/tables/ecl_weight_sensitivity.csv`.
Results (EAD, ECL and coverage by stage, per scenario and weighted): `reports/tables/ecl_coverage_by_stage.csv`. TBD.
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
Results: `reports/tables/capital_by_scenario.csv`, `el_vs_ecl.csv`, `stress_test.csv`, `capital_sensitivity.csv`. TBD.
Limitations: no downturn LGD or regulatory LGD floor, no defaulted-exposure RWA, no other risk types, no
off-balance-sheet exposures, parameters must be checked against the current Basel / CRR text.

## 8. Validation and monitoring
Out-of-time testing, backtesting, PSI, stability, sensitivity analysis.

## 9. Limitations and assumptions
List every assumption, known weakness, and what you would do with more data or time.

## 10. Appendix
Variable dictionary, additional tables and charts.
