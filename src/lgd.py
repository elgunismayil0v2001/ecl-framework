"""LGD model: fractional logit on realised LGD of resolved defaults.

Realised LGD is clipped to [0, 1]. The fractional logit is fitted as a weighted logistic regression
(each observation appears twice, as y=1 with weight LGD and y=0 with weight 1 - LGD), which gives the
same maximum likelihood estimate as a GLM with a binomial family and logit link.
Predictions are always between 0 and 1.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.eda import band_labels

DEFAULT_FEATURES = [
    "orig_ltv",
    "credit_score",
    "months_on_book",
    "hpi_yoy",
    "unemployment",
    "orig_dti",
    "orig_rate",
]


class LGDModel:
    def __init__(self, features=None, C=100.0):
        self.features = list(features or DEFAULT_FEATURES)
        self.C = C

    def fit(self, df, target_col="lgd"):
        usable = [f for f in self.features if f in df.columns and df[f].notna().any()]
        self.features_ = [f for f in usable if df[f].nunique() > 1]
        if not self.features_:
            raise ValueError("No usable LGD features")
        self.fill_ = df[self.features_].median()
        X = self._matrix(df[self.features_], fit=True)
        y = np.clip(df[target_col].to_numpy(dtype=float), 0.0, 1.0)
        X2 = np.vstack([X, X])
        y2 = np.concatenate([np.ones(len(y)), np.zeros(len(y))])
        w2 = np.concatenate([y, 1.0 - y])
        self.model_ = LogisticRegression(C=self.C, max_iter=2000).fit(X2, y2, sample_weight=w2)
        self.coef_ = pd.Series(self.model_.coef_[0], index=self.features_)
        return self

    def predict(self, df, floor=0.0):
        X = self._matrix(df[self.features_])
        return np.maximum(self.model_.predict_proba(X)[:, 1], floor)

    def _matrix(self, frame, fit=False):
        X = frame.fillna(self.fill_).to_numpy(dtype=float)
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_


def lgd_by_band(model, df, column, edges, target_col="lgd"):
    """Average realised and predicted LGD by band of `column`."""
    labels = band_labels(edges)
    idx = np.digitize(df[column].to_numpy(dtype=float), edges)
    table = pd.DataFrame(
        {"band_idx": idx, "realised": df[target_col].clip(0, 1).to_numpy(), "predicted": model.predict(df)}
    )
    out = table.groupby("band_idx").agg(n=("realised", "size"), realised=("realised", "mean"),
                                         predicted=("predicted", "mean")).reset_index()
    out.insert(1, "band", [labels[int(i)] for i in out["band_idx"]])
    return out.drop(columns="band_idx")
