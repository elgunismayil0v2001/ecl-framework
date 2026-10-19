"""ECL engine (single-loan arithmetic). Portfolio runs loop over loans and scenarios.

ECL = sum over t of marginal_PD(t) * LGD * EAD(t) * discount_factor(t)
Stage 1: horizon = 12 months. Stage 2 and 3: horizon = remaining lifetime.
"""
import numpy as np


def marginal_pd(monthly_pd):
    """Unconditional default probability in each month from conditional monthly PDs."""
    h = np.asarray(monthly_pd, dtype=float)
    survival_before = np.concatenate(([1.0], np.cumprod(1.0 - h)[:-1]))
    return survival_before * h


def stage_horizon(stage, remaining_months):
    return min(12, remaining_months) if stage == 1 else remaining_months


def compute_ecl(monthly_pd, lgd, ead, eir=0.0, horizon=None):
    h = np.asarray(monthly_pd, dtype=float)
    n = len(h) if horizon is None else min(int(horizon), len(h))
    h = h[:n]
    ead_t = np.full(n, float(ead)) if np.ndim(ead) == 0 else np.asarray(ead, dtype=float)[:n]
    discount = (1.0 + eir / 12.0) ** (-np.arange(1, n + 1))
    return float(np.sum(marginal_pd(h) * lgd * ead_t * discount))


def weighted_ecl(ecl_by_scenario, weights):
    """Probability-weighted ECL. Weights must sum to 1."""
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Scenario weights must sum to 1")
    return float(sum(ecl_by_scenario[s] * w for s, w in weights.items()))
