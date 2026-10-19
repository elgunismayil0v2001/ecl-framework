"""PD models: logistic scorecard (main) and LightGBM / CatBoost benchmark.

TODO:
- WoE binning and information value
- Logistic regression scorecard with monotonic, explainable features
- Benchmark model with monotonic constraints, SHAP
- Discrete-time hazard version for the lifetime PD term structure
- Write scores (pd_12m, pd_12m_orig) back to DuckDB table `scores`
"""


def fit_scorecard(*args, **kwargs):
    raise NotImplementedError


def fit_benchmark(*args, **kwargs):
    raise NotImplementedError


def lifetime_pd_curve(*args, **kwargs):
    raise NotImplementedError
