"""Basel IRB risk-weighted assets for retail exposures (residential mortgage defaults shown).

Verify parameters (asset correlation, PD floor, confidence level, scaling) against the current
Basel / CRR text before reporting numbers. Defaulted exposures (PD = 1) need separate treatment.
"""
import numpy as np
from scipy.stats import norm

MORTGAGE_CORRELATION = 0.15
CONFIDENCE = 0.999
PD_FLOOR = 0.0005


def capital_requirement(pd_, lgd, r=MORTGAGE_CORRELATION, conf=CONFIDENCE, pd_floor=PD_FLOOR):
    """K = LGD * N[(G(PD) + sqrt(R) * G(conf)) / sqrt(1 - R)] - PD * LGD (no maturity adjustment)."""
    pd_ = np.clip(np.asarray(pd_, dtype=float), pd_floor, 0.9999)
    lgd = np.asarray(lgd, dtype=float)
    z = (norm.ppf(pd_) + np.sqrt(r) * norm.ppf(conf)) / np.sqrt(1.0 - r)
    return lgd * norm.cdf(z) - pd_ * lgd


def rwa(pd_, lgd, ead, scaling=1.0, **kwargs):
    """RWA = K * 12.5 * EAD * scaling."""
    return capital_requirement(pd_, lgd, **kwargs) * 12.5 * np.asarray(ead, dtype=float) * scaling


def expected_loss(pd_, lgd, ead):
    return np.asarray(pd_, dtype=float) * np.asarray(lgd, dtype=float) * np.asarray(ead, dtype=float)


def capital_ratio(capital, total_rwa):
    return capital / total_rwa
