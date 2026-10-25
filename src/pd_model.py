"""PD scorecard: WoE-binned logistic regression, the main PD model.

Steps in `Scorecard.fit`:
1. WoE binning per feature (monotonic bad rate)
2. drop features with information value below `iv_min`
3. logistic regression on WoE values
4. drop the feature with the most negative coefficient and refit until all signs are positive
   (a negative coefficient means the feature works against its own risk direction)
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

from src.woe import WoEBinner


class Scorecard:
    def __init__(
        self,
        features,
        max_bins=10,
        min_bin_share=0.01,
        iv_min=0.02,
        C=1.0,
        monotonic=True,
        non_monotonic_features=(),
    ):
        self.features = list(features)
        self.non_monotonic_features = set(non_monotonic_features)
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share
        self.iv_min = iv_min
        self.C = C
        self.monotonic = monotonic

    def fit(self, df, target_col="default_12m"):
        y = df[target_col].astype(int).to_numpy()
        self.binners_, self.dropped_ = {}, {}
        for f in self.features:
            monotonic = self.monotonic and f not in self.non_monotonic_features
            binner = WoEBinner(self.max_bins, self.min_bin_share, monotonic).fit(df[f], y)
            if binner.iv_ < self.iv_min:
                self.dropped_[f] = f"IV {binner.iv_:.3f} below {self.iv_min}"
            else:
                self.binners_[f] = binner

        selected = list(self.binners_)
        while selected:
            model = LogisticRegression(C=self.C, max_iter=1000)
            model.fit(self._woe_matrix(df, selected), y)
            coef = model.coef_[0]
            if coef.min() > 0:
                break
            worst = selected[int(np.argmin(coef))]
            self.dropped_[worst] = "negative coefficient"
            selected.remove(worst)
        if not selected:
            raise ValueError("No usable features left after IV and sign checks")

        self.selected_ = selected
        self.model_ = model
        self.coef_ = pd.Series(model.coef_[0], index=selected)
        self.intercept_ = float(model.intercept_[0])
        return self

    def predict_proba(self, df):
        return self.model_.predict_proba(self._woe_matrix(df, self.selected_))[:, 1]

    def iv_table(self):
        rows = []
        for f in self.features:
            if f in self.binners_:
                selected = f in self.selected_
                note = "" if selected else self.dropped_.get(f, "")
                coef = float(self.coef_[f]) if selected else np.nan
                rows.append((f, self.binners_[f].iv_, selected, coef, note))
            else:
                rows.append((f, np.nan, False, np.nan, self.dropped_.get(f, "")))
        cols = ["feature", "iv", "selected", "coefficient", "note"]
        return pd.DataFrame(rows, columns=cols).sort_values("iv", ascending=False, ignore_index=True)

    def bin_tables(self):
        parts = []
        for f, binner in self.binners_.items():
            t = binner.table_.copy()
            t.insert(0, "feature", f)
            parts.append(t)
        return pd.concat(parts, ignore_index=True)

    def _woe_matrix(self, df, features):
        return np.column_stack([self.binners_[f].transform(df[f]) for f in features])


def points(p, base_score=600.0, base_odds=50.0, pdo=20.0):
    """Scorecard points. Odds are good:bad. `pdo` points double the odds; `base_odds` give `base_score`."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log((1.0 - p) / p)


def calibration_shift(p, target_rate):
    """Logit shift that makes the average predicted PD equal `target_rate`.

    Estimate it on a calibration window (for example the last months of the training data),
    never on the out-of-time sample you validate on.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    logit = np.log(p / (1.0 - p))

    def gap(delta):
        return float(np.mean(1.0 / (1.0 + np.exp(-(logit + delta)))) - target_rate)

    return float(brentq(gap, -10.0, 10.0))


def apply_logit_shift(p, delta):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1.0 - p)) + delta)))
