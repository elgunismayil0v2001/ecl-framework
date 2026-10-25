"""Roll-rate (Markov) model for lifetime PD.

States: 0 current, 1 = 30 days late, 2 = 60 days late, 3 = 90+ days late (default, absorbing).
Delinquency moves one step at a time, so a one-month default hazard is zero for a current loan. The
lifetime PD therefore comes from the probability of rolling 0 -> 1 -> 2 -> 3, which this model estimates
with one multinomial logistic regression per starting state (0, 1, 2). Covariates: loan age, loan
characteristics at origination and macro variables. Prepayment is treated as censoring when fitting.

`project` returns, for each loan and month, the probability of reaching state 3 in that month given
that the loan has not defaulted before (the same "hazard" interface used by the ECL engine).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.lifetime_pd import MACRO_COLS

DEFAULT_FEATURES = ["credit_score", "orig_ltv", "orig_dti", "orig_rate", *MACRO_COLS]


class RollRateModel:
    def __init__(self, features=None, C=10.0, max_rows_per_state=500_000, seed=42):
        self.features = list(features or DEFAULT_FEATURES)
        self.C = C
        self.max_rows_per_state = max_rows_per_state
        self.seed = seed

    def fit(self, df, state_col="dpd_months", next_col="next_dpd_months"):
        usable = [f for f in self.features if f in df.columns and df[f].notna().any()]
        self.features_ = [f for f in usable if df[f].nunique() > 1]
        self.fill_ = df[self.features_].median()
        X = self._design(df, fit=True)
        state = np.clip(df[state_col].to_numpy(dtype=float), 0, 2).astype(int)
        nxt = np.clip(df[next_col].to_numpy(dtype=float), 0, 3).astype(int)
        rng = np.random.default_rng(self.seed)

        self.models_ = {}
        for s in (0, 1, 2):
            idx = np.flatnonzero(state == s)
            if len(idx) == 0:
                self.models_[s] = ("const", s)  # never observed: assume the state persists
                continue
            y, w = nxt[idx], np.ones(len(idx))
            stay = np.flatnonzero(y == s)
            if len(stay) > self.max_rows_per_state:  # thin out the very common "no change" rows
                keep = rng.choice(stay, self.max_rows_per_state, replace=False)
                mask = np.ones(len(idx), dtype=bool)
                mask[stay] = False
                mask[keep] = True
                w[keep] = len(stay) / len(keep)
                idx, y, w = idx[mask], y[mask], w[mask]
            classes = np.unique(y)
            if len(classes) == 1:
                self.models_[s] = ("const", int(classes[0]))
                continue
            model = LogisticRegression(C=self.C, max_iter=2000).fit(X[idx], y, sample_weight=w)
            self.models_[s] = ("lr", model)

        self.coef_ = self._worse_state_coefficients(0)
        self.coef_by_state_ = {s: self._worse_state_coefficients(s) for s in (0, 1, 2)}
        return self

    def project(self, loans, macro_path, horizon=None):
        """Monthly default hazards (n_loans, horizon). Column k is the PD for month k + 1."""
        horizon = len(macro_path) if horizon is None else horizon
        base = loans.reset_index(drop=True)
        n = len(base)
        age0 = base["months_on_book"].to_numpy(dtype=float)
        remaining = base["term_months"].to_numpy(dtype=float) - age0
        current = np.clip(base["dpd_months"].to_numpy(dtype=float), 0, 2).astype(int)
        p = np.zeros((n, 4))
        p[np.arange(n), current] = 1.0

        static = [f for f in self.features_ if f not in MACRO_COLS]
        step = base[["months_on_book", *static]].copy()
        hazards = np.zeros((n, horizon))
        for k in range(horizon):
            step["months_on_book"] = age0 + k
            for c in MACRO_COLS:
                if c in self.features_:
                    step[c] = float(macro_path.iloc[k][c])
            X = self._design(step)
            new = p[:, [3]] * np.array([[0.0, 0.0, 0.0, 1.0]])
            for s in (0, 1, 2):
                new = new + p[:, [s]] * self._transition(s, X)
            new = np.where((k < remaining)[:, None], new, p)  # no transitions after contractual maturity
            d_before, d_after = p[:, 3], new[:, 3]
            alive = 1.0 - d_before
            hazards[:, k] = np.where(alive > 1e-12, (d_after - d_before) / np.maximum(alive, 1e-12), 0.0)
            p = new
        return np.clip(hazards, 0.0, 1.0)

    def _transition(self, state, X):
        out = np.zeros((X.shape[0], 4))
        kind, model = self.models_[state]
        if kind == "const":
            out[:, model] = 1.0
        else:
            out[:, model.classes_] = model.predict_proba(X)
        return out

    def _worse_state_coefficients(self, state):
        names = ["log_age", "age", *self.features_]
        kind, model = self.models_[state]
        if kind == "const":
            return pd.Series(0.0, index=names)
        classes = list(model.classes_)
        if len(classes) == 2:
            row = model.coef_[0]  # effect on the higher class
        else:
            row = model.coef_[classes.index(state + 1)] if (state + 1) in classes else model.coef_[-1]
        return pd.Series(row, index=names)

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
