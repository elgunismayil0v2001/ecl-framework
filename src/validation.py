"""Model validation metrics."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def gini(y_true, score):
    from sklearn.metrics import roc_auc_score

    return 2.0 * roc_auc_score(y_true, score) - 1.0


def ks(y_true, score):
    y_true = np.asarray(y_true)
    order = np.argsort(score)
    y = y_true[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def psi(expected, actual, bins=10, eps=1e-6):
    """Population stability index. Rule of thumb: <0.1 stable, 0.1 to 0.25 moderate, >0.25 large."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual, cuts)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y_true, p, bins=10):
    """Mean predicted PD vs observed default rate in equal-count score bins."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_pd=("p", "mean"), default_rate=("y", "mean"))
        .reset_index()
    )


def calibration_in_the_large(y_true, p):
    """Relative error of the average predicted PD against the observed default rate."""
    return float(np.mean(p) / np.mean(y_true) - 1.0)


def rank_order_corr(y_true, p, bins=10):
    """Spearman correlation between bin mean PD and bin default rate (1 = perfect rank order)."""
    table = calibration_table(y_true, p, bins)
    return float(spearmanr(table["mean_pd"], table["default_rate"])[0])
