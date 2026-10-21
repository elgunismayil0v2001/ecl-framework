"""Portfolio ECL: lifetime and 12-month ECL by scenario, stage 3 ECL, probability weighting.

For performing loans (Stage 1 and 2):
    ECL = LGD * sum over months t of marginal_PD(t) * EAD(t) * discount(t)
    with a 12-month horizon for Stage 1 and the remaining contractual term for Stage 2.
For defaulted loans (Stage 3): ECL = LGD * EAD (PD = 1, no discounting, simplification).
The discount rate is the loan's contractual rate (proxy for the effective interest rate).
LGD is evaluated per scenario with the macro values `lgd_macro_month` months ahead.
"""
import numpy as np
import pandas as pd

from src.ead import ead_paths
from src.lifetime_pd import project_hazards


def marginal_pd_matrix(hazards):
    """Unconditional default probability per month (rows are loans) from conditional hazards."""
    h = np.asarray(hazards, dtype=float)
    survival_before = np.hstack([np.ones((h.shape[0], 1)), np.cumprod(1.0 - h, axis=1)[:, :-1]])
    return survival_before * h


def _month_mask(horizon_months, n_cols):
    return np.arange(1, n_cols + 1)[None, :] <= np.asarray(horizon_months, dtype=float)[:, None]


def portfolio_ecl(hazards, lgd, ead_matrix, eir, horizon_months):
    """Vectorised ECL per loan. Equals `src.ecl.compute_ecl` loan by loan."""
    h = np.asarray(hazards, dtype=float)
    k = np.arange(1, h.shape[1] + 1)
    discount = (1.0 + np.asarray(eir, dtype=float)[:, None] / 12.0) ** (-k[None, :])
    mask = _month_mask(horizon_months, h.shape[1])
    return (marginal_pd_matrix(h) * ead_matrix * discount * mask).sum(axis=1) * np.asarray(lgd, dtype=float)


def cumulative_pd_over(hazards, horizon_months):
    """Cumulative PD over each loan's own horizon."""
    h = np.asarray(hazards, dtype=float)
    mask = _month_mask(horizon_months, h.shape[1])
    return 1.0 - np.prod(1.0 - h * mask, axis=1)


def lifetime_horizon(loans, cap=360):
    """Longest remaining term among performing loans, capped."""
    performing = loans[loans["stage"] < 3]
    if len(performing) == 0:
        return 1
    remaining = performing["term_months"] - performing["months_on_book"]
    return int(max(1, min(cap, remaining.max())))


def _scenario_lgd(lgd_model, loans, path, default_lgd, macro_month):
    if lgd_model is None:
        return np.full(len(loans), float(default_lgd))
    row = path.iloc[min(macro_month, len(path) - 1)]
    frame = loans.assign(hpi_yoy=row["hpi_yoy"], unemployment=row["unemployment"])
    return lgd_model.predict(frame)


def build_loan_ecl(loans, hazard_model, lgd_model, scenarios, cpr=0.06, default_lgd=0.25, lgd_macro_month=12):
    """Loan-level ECL for every scenario as a long DataFrame.

    `loans` needs: loan_id, report_month, months_on_book, dpd_months, term_months, current_upb,
    orig_rate, stage and the hazard / LGD features. `scenarios` maps name to a macro path with one
    row per month; all paths must have the same length (the projection horizon).
    """
    loans = loans.reset_index(drop=True)
    horizon = len(next(iter(scenarios.values())))
    stage = loans["stage"].to_numpy(dtype=int)
    n = len(loans)
    performing = np.flatnonzero(stage < 3)
    defaulted = np.flatnonzero(stage == 3)

    ead0 = loans["current_upb"].to_numpy(dtype=float)
    eir = loans["orig_rate"].to_numpy(dtype=float)
    remaining = np.maximum((loans["term_months"] - loans["months_on_book"]).to_numpy(dtype=float), 0.0)
    perf = loans.iloc[performing]
    if len(performing):
        ead_matrix = ead_paths(ead0[performing], eir[performing], remaining[performing], horizon, cpr)
        rem_p = remaining[performing]
        horizon_months = np.where(stage[performing] == 1, np.minimum(12.0, rem_p), rem_p)
        horizon_months = np.minimum(horizon_months, horizon)

    frames = []
    for name, path in scenarios.items():
        lgd = _scenario_lgd(lgd_model, loans, path, default_lgd, lgd_macro_month)
        ecl = np.zeros(n)
        pd_12m = np.zeros(n)
        pd_horizon = np.zeros(n)
        if len(performing):
            hazards = project_hazards(hazard_model, perf, path, horizon)
            ecl[performing] = portfolio_ecl(
                hazards, lgd[performing], ead_matrix, eir[performing], horizon_months
            )
            pd_12m[performing] = cumulative_pd_over(hazards, np.minimum(12.0, rem_p))
            pd_horizon[performing] = cumulative_pd_over(hazards, horizon_months)
        if len(defaulted):
            ecl[defaulted] = lgd[defaulted] * ead0[defaulted]
            pd_12m[defaulted] = 1.0
            pd_horizon[defaulted] = 1.0
        frames.append(
            pd.DataFrame(
                {
                    "loan_id": loans["loan_id"],
                    "report_month": loans["report_month"],
                    "scenario": name,
                    "stage": stage,
                    "ead": ead0,
                    "ecl": ecl,
                    "lgd": lgd,
                    "pd_12m": pd_12m,
                    "pd_horizon": pd_horizon,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def coverage_by_stage(loan_ecl, weights=None):
    """EAD, ECL and coverage (ECL / EAD) by stage, per scenario and probability-weighted."""
    g = loan_ecl.groupby(["scenario", "stage"], as_index=False).agg(ead=("ead", "sum"), ecl=("ecl", "sum"))
    g["coverage"] = g["ecl"] / g["ead"]
    if weights:
        w = g["scenario"].map(weights)
        weighted = (
            g.assign(ead=g["ead"] * w, ecl=g["ecl"] * w)
            .groupby("stage", as_index=False)[["ead", "ecl"]]
            .sum()
        )
        weighted["coverage"] = weighted["ecl"] / weighted["ead"]
        weighted.insert(0, "scenario", "weighted")
        g = pd.concat([g, weighted], ignore_index=True)
    return g


def weighted_total(loan_ecl, weights):
    totals = loan_ecl.groupby("scenario")["ecl"].sum()
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Scenario weights must sum to 1")
    return float(sum(totals[s] * w for s, w in weights.items()))


def weight_sensitivity(loan_ecl, weight_sets):
    """Total ECL under alternative scenario weight sets."""
    totals = loan_ecl.groupby("scenario")["ecl"].sum()
    ead = float(loan_ecl[loan_ecl["scenario"] == next(iter(totals.index))]["ead"].sum())
    rows = []
    for name, weights in weight_sets.items():
        total = weighted_total(loan_ecl, weights)
        row = {"weights": name, **{f"w_{k}": v for k, v in weights.items()}}
        rows.append({**row, "ecl": total, "coverage": total / ead})
    return pd.DataFrame(rows)
