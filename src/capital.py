"""Capital adequacy from model output: IRB RWA, expected loss vs ECL, stress test.

Simplifications (document them):
- IRB formula for residential mortgages (src/rwa.py); PD = 12-month PD of each scenario, LGD = model LGD.
- Defaulted exposures (Stage 3) are excluded from RWA. Their treatment is a separate regulatory formula.
- Capital is an assumption (a starting capital ratio on base RWA), not the real capital of a bank.
- Stress test: extra provisions (downside ECL minus base ECL) reduce capital one for one. No profit,
  tax, dividends, management actions or RWA mitigation.
"""
import numpy as np
import pandas as pd

from src.rwa import PD_FLOOR, expected_loss, rwa


def scenario_metrics(loan_ecl, scenario, lgd_floor=0.0, pd_floor=PD_FLOOR):
    d = loan_ecl[loan_ecl["scenario"] == scenario]
    perf = d[d["stage"] < 3]
    lgd = np.maximum(perf["lgd"].to_numpy(dtype=float), lgd_floor)
    ead = perf["ead"].to_numpy(dtype=float)
    pd_ = perf["pd_12m"].to_numpy(dtype=float)
    total_rwa = float(rwa(pd_, lgd, ead, pd_floor=pd_floor).sum())
    el = float(expected_loss(np.maximum(pd_, pd_floor), lgd, ead).sum())
    return {
        "scenario": scenario,
        "ead_total": float(d["ead"].sum()),
        "ead_performing": float(ead.sum()),
        "rwa": total_rwa,
        "rwa_density": total_rwa / ead.sum() if ead.sum() > 0 else np.nan,
        "el_irb": el,
        "ecl_total": float(d["ecl"].sum()),
        "ecl_performing": float(perf["ecl"].sum()),
    }


def el_vs_ecl(metrics):
    """Compare IRB expected loss with IFRS 9 ECL on performing exposures.

    Positive shortfall: EL above provisions. Under the Basel framework a shortfall is deducted from
    capital; an excess can count towards Tier 2 within limits (check the current rules).
    """
    return {
        "scenario": metrics["scenario"],
        "el_irb": metrics["el_irb"],
        "ecl_performing": metrics["ecl_performing"],
        "shortfall": metrics["el_irb"] - metrics["ecl_performing"],
        "ecl_to_el": metrics["ecl_performing"] / metrics["el_irb"] if metrics["el_irb"] > 0 else np.nan,
    }


def stress_test(base, stressed, capital_ratio):
    """Capital ratio before and after the stress for an assumed starting ratio on base RWA."""
    capital = capital_ratio * base["rwa"]
    extra_provisions = stressed["ecl_total"] - base["ecl_total"]
    capital_after = capital - extra_provisions
    ratio_after = capital_after / stressed["rwa"]
    return {
        "capital_ratio_start": capital_ratio,
        "capital": capital,
        "extra_provisions": extra_provisions,
        "rwa_base": base["rwa"],
        "rwa_stressed": stressed["rwa"],
        "capital_after": capital_after,
        "capital_ratio_after": ratio_after,
        "change_pp": (ratio_after - capital_ratio) * 100.0,
    }


def capital_sensitivity(base, stressed, ratios=(0.08, 0.10, 0.12, 0.15)):
    return pd.DataFrame([stress_test(base, stressed, r) for r in ratios])
