# Model documentation: IFRS 9 ECL and capital adequacy framework

Write this like a bank model development document. Fill each section as you build.

## 1. Purpose and scope
Portfolio, products, intended use (provisioning and illustrative capital), what is out of scope.

## 2. Data
Sources, sample period, exclusions, data quality checks, default definition (90+ DPD), observation and performance windows.
Observations in the last 12 months of the data are excluded because their 12-month outcome window is right-censored.

## 3. PD model
Methodology (WoE scorecard, ML benchmark), variable selection, monotonicity, performance (Gini, KS, calibration), lifetime PD term structure.

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
