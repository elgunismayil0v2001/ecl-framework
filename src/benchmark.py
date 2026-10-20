"""LightGBM benchmark PD model with monotonic constraints.

Constraints encode the expected risk direction: +1 means PD must not decrease when the feature rises,
-1 means PD must not increase, 0 means unconstrained.
"""
import numpy as np
import pandas as pd

DEFAULT_MONOTONE = {
    "credit_score": -1,
    "orig_ltv": 1,
    "orig_dti": 1,
    "orig_rate": 1,
    "dpd_months": 1,
    "max_dpd_last_12m": 1,
    "unemployment": 1,
    "hpi_yoy": -1,
    "gdp_yoy": -1,
}


class BenchmarkModel:
    def __init__(self, features, monotone=None, **params):
        self.features = list(features)
        self.monotone = DEFAULT_MONOTONE if monotone is None else monotone
        self.params = params

    def fit(self, df, target_col="default_12m"):
        from lightgbm import LGBMClassifier  # imported here so the rest of the code works without it

        settings = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 100,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
        settings.update(self.params)
        constraints = [int(self.monotone.get(f, 0)) for f in self.features]
        self.model_ = LGBMClassifier(monotone_constraints=constraints, **settings)
        self.model_.fit(df[self.features], df[target_col].astype(int))
        return self

    def predict_proba(self, df):
        return self.model_.predict_proba(df[self.features])[:, 1]

    def importance(self, df):
        """Mean absolute SHAP contribution (log-odds) per feature, from LightGBM's own TreeSHAP."""
        contrib = self.model_.booster_.predict(df[self.features], pred_contrib=True)
        mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
        table = pd.DataFrame({"feature": self.features, "mean_abs_shap": mean_abs})
        return table.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def plot_importance(table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(t["feature"], t["mean_abs_shap"])
    ax.set_title("Benchmark PD model: mean absolute SHAP value")
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
