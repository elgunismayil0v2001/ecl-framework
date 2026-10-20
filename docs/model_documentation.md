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
section 6.

## 4. LGD and EAD models
Methodology, assumptions, limitations.

## 5. Staging
Stage 1, 2, 3 criteria, SICR thresholds, justification and sensitivity.

## 6. ECL calculation
Formula, discounting, horizons, macro scenarios and weights, results by stage and scenario.

## 7. Capital adequacy
IRB RWA formula and parameters, assumed capital, capital ratio, stress test, comparison of IRB expected loss with IFRS 9 ECL.

## 8. Validation and monitoring
Out-of-time testing, backtesting, PSI, stability, sensitivity analysis.

## 9. Limitations and assumptions
List every assumption, known weakness, and what you would do with more data or time.

## 10. Appendix
Variable dictionary, additional tables and charts.
