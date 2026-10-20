"""EAD: expected balance path from the amortisation schedule with a prepayment assumption.

EAD at month t is the balance at the start of month t (the exposure if the loan defaults during
that month). Prepayment is a constant annual rate (CPR) applied as a monthly rate (SMM). The PD
hazards are not adjusted for prepayment as a competing risk (simplification, document it).
"""
import numpy as np


def annuity_payment(balance, annual_rate, months):
    """Level monthly payment that repays `balance` over `months`. Pays off the balance if months <= 0."""
    balance = np.asarray(balance, dtype=float)
    r = np.asarray(annual_rate, dtype=float) / 12.0
    n = np.asarray(months, dtype=float)
    n_safe = np.maximum(n, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        payment = np.where(
            np.abs(r) < 1e-12,
            balance / n_safe,
            balance * r / (1.0 - (1.0 + r) ** (-n_safe)),
        )
    return np.where(n <= 0, balance, payment)


def ead_paths(balance, annual_rate, remaining_months, horizon, cpr=0.0):
    """Matrix (n_loans, horizon) of expected balances at the start of months 1..horizon."""
    b = np.atleast_1d(np.asarray(balance, dtype=float)).copy()
    rate = np.broadcast_to(np.asarray(annual_rate, dtype=float), b.shape)
    rem = np.broadcast_to(np.asarray(remaining_months, dtype=float), b.shape)
    smm = 1.0 - (1.0 - cpr) ** (1.0 / 12.0)
    out = np.zeros((b.shape[0], horizon))
    for t in range(horizon):
        out[:, t] = b
        remaining = np.maximum(rem - t, 0.0)
        payment = annuity_payment(b, rate, remaining)
        principal = np.where(remaining <= 0, b, np.minimum(payment - b * rate / 12.0, b))
        b = np.maximum((b - principal) * (1.0 - smm), 0.0)
    return out
