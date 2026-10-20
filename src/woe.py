"""Weight of evidence (WoE) binning with optional monotonic bad rate.

Convention: WoE = ln(share of bads / share of goods). A higher WoE means higher risk, so a logistic
regression on WoE features should have positive coefficients.
Counts are smoothed with `eps` so empty bins do not give infinite WoE.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class WoEBinner:
    def __init__(self, max_bins=10, min_bin_share=0.05, monotonic=True, eps=0.5):
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share
        self.monotonic = monotonic
        self.eps = eps

    def fit(self, x, y):
        x = pd.Series(x).reset_index(drop=True).astype(float)
        y = pd.Series(y).reset_index(drop=True).astype(int)
        missing = x.isna()
        xv = x[~missing].to_numpy()
        yv = y[~missing].to_numpy()

        edges = self._initial_edges(xv)
        idx = np.digitize(xv, edges)
        n_bins = len(edges) + 1
        n = list(np.bincount(idx, minlength=n_bins).astype(float))
        bad = list(np.bincount(idx, weights=yv, minlength=n_bins))
        edges = list(edges)

        self._merge_small_bins(n, bad, edges, total=len(xv))
        if self.monotonic:
            self._merge_non_monotonic(n, bad, edges)

        n_miss, bad_miss = float(missing.sum()), float(y[missing].sum())
        self.has_missing_ = n_miss > 0
        self.edges_ = np.array(edges, dtype=float)
        self._compute_woe(np.array(n), np.array(bad), n_miss, bad_miss)
        return self

    def transform(self, x):
        arr = pd.Series(x).astype(float).to_numpy()
        nan = np.isnan(arr)
        idx = np.digitize(np.where(nan, 0.0, arr), self.edges_)
        return np.where(nan, self.missing_woe_, self.woe_[idx])

    # --- internals -------------------------------------------------------------------------
    def _initial_edges(self, xv):
        if len(xv) == 0:
            return np.array([])
        probs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
        return np.unique(np.quantile(xv, probs))

    def _merge_small_bins(self, n, bad, edges, total):
        while total > 0 and len(n) > 1:
            too_small = len(n) > 2 and min(n) / total < self.min_bin_share
            if min(n) > 0 and not too_small:
                break
            j = int(np.argmin(n))
            rates = [b / c if c > 0 else 0.0 for b, c in zip(bad, n, strict=True)]
            left = abs(rates[j] - rates[j - 1]) if j > 0 else np.inf
            right = abs(rates[j] - rates[j + 1]) if j < len(n) - 1 else np.inf
            self._merge(n, bad, edges, j - 1 if left <= right else j)

    def _merge_non_monotonic(self, n, bad, edges):
        if len(n) < 3:
            return
        rates = [b / c if c > 0 else 0.0 for b, c in zip(bad, n, strict=True)]
        corr = spearmanr(np.arange(len(rates)), rates)[0]
        direction = 1 if np.isnan(corr) or corr >= 0 else -1
        while len(n) > 1:
            rates = [b / c if c > 0 else 0.0 for b, c in zip(bad, n, strict=True)]
            violation = next(
                (i for i in range(len(rates) - 1) if direction * (rates[i + 1] - rates[i]) < 0),
                None,
            )
            if violation is None:
                break
            self._merge(n, bad, edges, violation)

    @staticmethod
    def _merge(n, bad, edges, i):
        """Merge bin i with bin i + 1 and drop the edge between them."""
        n[i] += n[i + 1]
        bad[i] += bad[i + 1]
        del n[i + 1], bad[i + 1], edges[i]

    def _compute_woe(self, n, bad, n_miss, bad_miss):
        good = n - bad
        good_miss = n_miss - bad_miss
        k = len(n) + (1 if self.has_missing_ else 0)
        total_bad = bad.sum() + bad_miss
        total_good = good.sum() + good_miss
        denom_bad = total_bad + self.eps * k
        denom_good = total_good + self.eps * k

        pct_bad = (bad + self.eps) / denom_bad
        pct_good = (good + self.eps) / denom_good
        self.woe_ = np.log(pct_bad / pct_good)
        iv_bins = (pct_bad - pct_good) * self.woe_

        lower = np.concatenate(([-np.inf], self.edges_))
        upper = np.concatenate((self.edges_, [np.inf]))
        table = pd.DataFrame(
            {
                "bin": [f"[{lo:.4g}, {up:.4g})" for lo, up in zip(lower, upper, strict=True)],
                "n": n,
                "bad": bad,
                "bad_rate": bad / np.maximum(n, 1),
                "woe": self.woe_,
                "iv": iv_bins,
            }
        )
        self.missing_woe_ = 0.0
        if self.has_missing_:
            pb = (bad_miss + self.eps) / denom_bad
            pg = (good_miss + self.eps) / denom_good
            self.missing_woe_ = float(np.log(pb / pg))
            row = {
                "bin": "missing",
                "n": n_miss,
                "bad": bad_miss,
                "bad_rate": bad_miss / max(n_miss, 1),
                "woe": self.missing_woe_,
                "iv": (pb - pg) * self.missing_woe_,
            }
            table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        self.table_ = table
        self.iv_ = float(table["iv"].sum())
