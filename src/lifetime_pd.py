"""Lifetime PD: discrete-time hazard model, macro scenarios and PD term structure.

The hazard model is a logistic regression on person-period data (one row per performing loan-month).
It predicts the probability of defaulting in the next month given loan age, loan characteristics and
macro variables. Projecting the hazards forward under a macro scenario and compounding them gives the
lifetime PD curve.

Projection assumptions (document them): loan characteristics stay constant, age increases by one
month per step, delinquency is cured after the first month, and hazards are zero after contractual
maturity. Prepayment is treated as censoring when fitting and is handled in the EAD path.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MACRO_COLS = ["unemployment", "hpi_yoy", "gdp_yoy"]
DEFAULT_FEATURES = ["credit_score", "orig_ltv", "orig_dti", "orig_rate", "dpd_months", *MACRO_COLS]

# Illustrative scenario shocks relative to the last observed macro values. These are assumptions,
# not official forecasts. unemployment in percentage points, growth rates as fractions.
DEFAULT_SHOCKS = {
    "base": {"unemployment": 0.0, "hpi_yoy": 0.0, "gdp_yoy": 0.0},
    "upside": {"unemployment": -1.0, "hpi_yoy": 0.03, "gdp_yoy": 0.01},
    "downside": {"unemployment": 3.0, "hpi_yoy": -0.10, "gdp_yoy": -0.04},
}


class HazardModel:
    def __init__(self, features=None, C=10.0):
        self.features = list(features or DEFAULT_FEATURES)
        self.C = C

    def fit(self, df, target_col="default_next"):
        usable = [f for f in self.features if f in df.columns and df[f].notna().any()]
        self.features_ = [f for f in usable if df[f].nunique() > 1]
        self.fill_ = df[self.features_].median()
        X = self._design(df, fit=True)
        self.model_ = LogisticRegression(C=self.C, max_iter=2000)
        self.model_.fit(X, df[target_col].astype(int).to_numpy())
        names = ["log_age", "age", *self.features_]
        self.coef_ = pd.Series(self.model_.coef_[0], index=names)  # per standard deviation
        return self

    def predict_hazard(self, df):
        return self.model_.predict_proba(self._design(df))[:, 1]

    def _design(self, df, fit=False):
        age = df["months_on_book"].to_numpy(dtype=float)
        cols = [np.log1p(age), age]
        cols += [df[f].fillna(self.fill_[f]).to_numpy(dtype=float) for f in self.features_]
        X = np.column_stack(cols)
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_


def _shock_profile(horizon, ramp, hold, revert):
    """0 at month 0, linear ramp to 1, hold, then linear reversion to 0."""
    k = np.arange(horizon, dtype=float)
    up = k / ramp if ramp > 0 else np.ones(horizon)
    down = 1.0 - (k - ramp - hold) / revert if revert > 0 else np.zeros(horizon)
    profile = np.where(k <= ramp, up, np.where(k <= ramp + hold, 1.0, down))
    return np.clip(profile, 0.0, 1.0)


def make_scenarios(last_macro, horizon, shocks=None, ramp_months=12, hold_months=12, revert_months=24):
    """Macro paths per scenario: DataFrame with one row per month, starting at the last observed values."""
    profile = _shock_profile(horizon, ramp_months, hold_months, revert_months)
    paths = {}
    for name, shock in (shocks or DEFAULT_SHOCKS).items():
        paths[name] = pd.DataFrame({c: last_macro[c] + shock.get(c, 0.0) * profile for c in MACRO_COLS})
    return paths


def project_hazards(model, loans, macro_path, horizon=None):
    """Monthly default hazards (n_loans, horizon). Column k is the PD for month k + 1."""
    horizon = len(macro_path) if horizon is None else horizon
    base = loans.reset_index(drop=True)
    age0 = base["months_on_book"].to_numpy(dtype=float)
    remaining = base["term_months"].to_numpy(dtype=float) - age0
    dpd0 = base["dpd_months"].to_numpy(dtype=float)
    hazards = np.zeros((len(base), horizon))
    for k in range(horizon):
        step = base.copy()
        step["months_on_book"] = age0 + k
        step["dpd_months"] = dpd0 if k == 0 else 0.0
        for c in MACRO_COLS:
            step[c] = float(macro_path.iloc[k][c])
        hazards[:, k] = np.where(k < remaining, model.predict_hazard(step), 0.0)
    return hazards


def cumulative_pd(hazards):
    """Cumulative default probability by month: 1 - prod(1 - h)."""
    return 1.0 - np.cumprod(1.0 - np.asarray(hazards, dtype=float), axis=1)


def lifetime_curves(model, loans, scenarios, horizon):
    """Portfolio-average cumulative PD by month and scenario, as a tidy DataFrame."""
    rows = []
    for name, path in scenarios.items():
        cum = cumulative_pd(project_hazards(model, loans, path, horizon)).mean(axis=0)
        rows.append(pd.DataFrame({"month": np.arange(1, horizon + 1), "scenario": name, "cum_pd": cum}))
    return pd.concat(rows, ignore_index=True)


def scenario_ordering_ok(curves):
    """True if, at the end of the horizon, downside PD > base PD > upside PD."""
    last = curves[curves["month"] == curves["month"].max()].set_index("scenario")["cum_pd"]
    return bool(last["downside"] > last["base"] > last["upside"])
