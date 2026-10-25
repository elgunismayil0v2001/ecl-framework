# Changelog

## 0.1.0

First release.

- Data: Freddie Mac loader, FRED macro loader, 14 data quality checks, synthetic data for tests and smoke runs.
- PD: WoE scorecard (monotone bins, free-shape seasoning bins for loan age, no macro variables) with sign checks, LightGBM benchmark with monotonic constraints,
  out-of-time split with gap, validation gates (Gini, KS, calibration, rank order, PSI).
- Lifetime PD: roll-rate (Markov) model over delinquency states, three macro scenarios.
- LGD (fractional logit) and EAD (amortisation with prepayment).
- IFRS 9: staging, 12-month and lifetime ECL by scenario, probability-weighted ECL, sensitivity tables.
- Capital: IRB RWA for residential mortgages, expected loss vs ECL, stress test.
- Monitoring: PSI over time, correlation-adjusted PD backtest on quarter-end snapshots, stage migration,
  PD consistency.
- Reporting: `make report` fills results tables, the data source and the validation split in the README and
  documentation.
- Presentation: README with results, charts and lessons learned; MIT license; `make demo` runs everything.
- CI: lint and tests on Python 3.11 and 3.12; release workflow on version tags.
