"""Simulate a Freddie Mac style mortgage dataset (origination file, performance file, FRED macro CSVs).

For use when the real data cannot be downloaded. The files have the real layout, so the whole pipeline
(loader, SQL, models, ECL, capital, monitoring) runs on them unchanged. THE DATA IS SIMULATED: results
describe the simulation, not the mortgage market. Say so wherever you show results.

The simulation has a known structure: default risk depends on credit score, LTV, DTI, loan age,
unemployment and house prices; loss severity depends on current LTV; prepayment depends on the gap
between the loan rate and the market rate. Macro paths contain a recession episode in 2020.

    python -m src.simulate --outdir data/sim --loans 15000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.freddie_schema import LAYOUTS, internal_names
from src.load_macro import build_macro_table

FIRST_MONTH = pd.Timestamp("2015-01-01")  # macro history starts here (needs 12 months for yoy)
OBS_START = 24  # month index of 2017-01: first loans are originated here
STATES = ["CA", "TX", "FL", "NY", "IL", "OH", "AZ", "WA", "GA", "NC"]


def _months(start, n):
    return pd.date_range(start, periods=n, freq="MS")


def macro_paths(n_months=120, seed=0):
    """Monthly unemployment (%), house price index, market mortgage rate (%), quarterly real GDP."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    # key dates by month index from 2015-01
    anchors_u = {0: 5.6, 24: 4.6, 60: 3.6, 63: 3.5, 64: 14.0, 66: 11.0, 72: 6.5, 84: 3.9, 96: 3.7, 119: 4.1}
    u = np.interp(t, list(anchors_u), list(anchors_u.values()))
    u = u + rng.normal(0, 0.08, n_months)
    anchors_g = {0: 0.05, 36: 0.06, 60: 0.05, 72: 0.10, 84: 0.17, 90: 0.12, 100: 0.03, 119: 0.04}
    annual_growth = np.interp(t, list(anchors_g), list(anchors_g.values()))
    hpi = 175.0 * np.cumprod((1.0 + annual_growth) ** (1 / 12))
    anchors_r = {0: 3.9, 24: 4.0, 48: 4.6, 66: 3.1, 72: 2.9, 84: 3.1, 96: 6.4, 108: 6.8, 119: 6.9}
    market_rate = np.interp(t, list(anchors_r), list(anchors_r.values()))
    quarters = _months(FIRST_MONTH, n_months)[::3]
    q_growth = np.full(len(quarters), 0.005)
    q_growth[(quarters >= "2020-01-01") & (quarters < "2020-04-01")] = -0.012
    q_growth[(quarters >= "2020-04-01") & (quarters < "2020-07-01")] = -0.085
    q_growth[(quarters >= "2020-07-01") & (quarters < "2020-10-01")] = 0.075
    q_growth[(quarters >= "2020-10-01") & (quarters < "2021-04-01")] = 0.012
    gdp = 16000.0 * np.cumprod(1.0 + q_growth + rng.normal(0, 0.001, len(quarters)))
    return u, hpi, market_rate, quarters, gdp


def write_fred_csvs(outdir, n_months=120, seed=0):
    u, hpi, _, quarters, gdp = macro_paths(n_months, seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    months = _months(FIRST_MONTH, n_months)
    pd.DataFrame({"observation_date": months, "UNRATE": u.round(2)}).to_csv(out / "UNRATE.csv", index=False)
    pd.DataFrame({"observation_date": months, "CSUSHPINSA": hpi.round(3)}).to_csv(
        out / "CSUSHPINSA.csv", index=False
    )
    pd.DataFrame({"observation_date": quarters, "GDPC1": gdp.round(1)}).to_csv(out / "GDPC1.csv", index=False)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(n_loans=15000, n_months=120, seed=0):
    """Return (loans, performance) as DataFrames with internal column names (not yet in file layout)."""
    rng = np.random.default_rng(seed)
    unemp, hpi, market, _, _ = macro_paths(n_months, seed)
    hpi_growth = np.r_[np.full(12, np.nan), hpi[12:] / hpi[:-12] - 1.0]

    n = n_loans
    orig = rng.integers(OBS_START, OBS_START + 36, n)  # originated 2017-01 .. 2019-12
    score = np.clip(rng.normal(740, 50, n), 620, 830).round()
    ltv_pool = np.where(rng.random(n) < 0.35, rng.choice([80, 90, 95], n), rng.uniform(45, 85, n))
    ltv = np.clip(ltv_pool, 40, 97).round()
    dti = rng.uniform(15, 50, n).round()
    term = rng.choice([360, 180, 240], n, p=[0.8, 0.15, 0.05])
    upb0 = (rng.uniform(100_000, 450_000, n) / 1000).round() * 1000
    rate = np.clip(market[orig] + 0.5 + 0.01 * (740 - score) / 5 + rng.normal(0, 0.15, n), 2.5, 9.0)
    r_m = rate / 100.0 / 12.0
    state = rng.choice(STATES, n)
    first_pay = orig + 1

    alive = np.ones(n, dtype=bool)
    dpd = np.zeros(n, dtype=int)
    def_months = np.zeros(n, dtype=int)
    liq_after = np.zeros(n, dtype=int)
    rows = []

    for m in range(n_months):
        live = alive & (first_pay <= m)
        if not live.any():
            continue
        idx = np.flatnonzero(live)
        age = m - orig[idx]
        growth = np.nan_to_num(hpi_growth[m], nan=0.04)
        u_m = unemp[m]
        # scheduled balance (closed form annuity)
        rr, nn = r_m[idx], term[idx].astype(float)
        factor = (1 + rr) ** nn
        bal = upb0[idx] * (factor - (1 + rr) ** age) / (factor - 1)
        bal = np.maximum(bal, 0.0)

        d = dpd[idx]
        u = rng.random(len(idx))
        # start of delinquency
        logit = (
            -7.6
            + 1.1 * np.log1p(age) - 0.02 * age
            - 0.012 * (score[idx] - 740)
            + 0.025 * (ltv[idx] - 80)
            + 0.02 * (dti[idx] - 35)
            + 0.25 * (u_m - 4.5)
            - 6.0 * (growth - 0.04)
        )
        h01 = _sigmoid(logit)
        new_d = d.copy()
        new_d[(d == 0) & (u < h01)] = 1
        m1 = d == 1
        new_d[m1 & (u < 0.50)] = 0
        new_d[m1 & (u >= 0.50) & (u < 0.85)] = 2
        m2 = d == 2
        new_d[m2 & (u < 0.25)] = 0
        new_d[m2 & (u >= 0.25) & (u < 0.80)] = 3
        in_def = d >= 3
        new_d[in_def] = d[in_def] + 1
        just_def = (d < 3) & (new_d >= 3)
        extra = 3 if u_m > 7.0 else 0
        liq_after[idx[just_def]] = 6 + rng.integers(0, 10, just_def.sum()) + extra
        def_months[idx[in_def | just_def]] += 1
        dpd[idx] = new_d

        zero = np.full(len(idx), "", dtype=object)
        removal = np.full(len(idx), np.nan)
        loss = np.full(len(idx), np.nan)
        upb_rep = bal.copy()
        gone = np.zeros(len(idx), dtype=bool)

        # liquidation of defaulted loans
        liq = (new_d >= 3) & (def_months[idx] >= liq_after[idx])
        if liq.any():
            li = np.flatnonzero(liq)
            ltv_cur = ltv[idx[li]] * (bal[li] / upb0[idx[li]]) / (hpi[m] / hpi[orig[idx[li]]])
            lgd_mean = _sigmoid(
                -0.9 + 0.035 * (ltv_cur - 80) - 0.004 * (score[idx[li]] - 740) + 0.15 * (u_m - 4.5)
            )
            lgd = np.clip(lgd_mean + rng.normal(0, 0.15, len(li)), 0.0, 1.0)
            zero[li] = rng.choice(["09", "03", "02"], len(li), p=[0.6, 0.25, 0.15])
            removal[li] = bal[li]
            loss[li] = lgd * bal[li]
            gone[li] = True
        # prepayment of loans that are current
        prep_p = np.clip(0.003 + 0.018 * np.maximum(0.0, rate[idx] - market[m]), 0.0, 0.04)
        prep = (new_d == 0) & (rng.random(len(idx)) < prep_p) & ~gone
        # loans at maturity are repaid
        mature = (age >= term[idx]) & ~gone & ~prep
        end = prep | mature
        if end.any():
            ei = np.flatnonzero(end)
            zero[ei] = "01"
            removal[ei] = bal[ei]
            gone[ei] = True
        upb_rep[gone] = np.nan
        alive[idx[gone]] = False

        rows.append(
            pd.DataFrame(
                {
                    "i": idx,
                    "month": m,
                    "upb": upb_rep,
                    "dpd": new_d,
                    "age": age,
                    "zero": zero,
                    "removal": removal,
                    "loss": loss,
                }
            )
        )

    perf = pd.concat(rows, ignore_index=True)
    loans = pd.DataFrame(
        {
            "i": np.arange(n),
            "orig": orig,
            "score": score.astype(int),
            "ltv": ltv.astype(int),
            "dti": dti.astype(int),
            "upb0": upb0.astype(int),
            "rate": rate.round(3),
            "term": term,
            "state": state,
        }
    )
    return loans, perf


def _loan_ids(loans):
    months = _months(FIRST_MONTH, int(loans["orig"].max()) + 2)
    dates = months[loans["orig"].to_numpy()]
    return [
        f"F{x.year % 100:02d}Q{(x.month - 1) // 3 + 1}{i + 1:07d}"
        for i, x in zip(loans["i"], dates, strict=True)
    ]


def to_freddie_files(loans, perf, outdir, layout="v2", n_months=120):
    """Write origination and performance files in the chosen Freddie Mac layout."""
    cfg = LAYOUTS[layout]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    months = _months(FIRST_MONTH, n_months + 2)
    ids = np.array(_loan_ids(loans))

    def ym(m):
        return months[m].strftime("%Y%m")

    terms = loans["term"].to_numpy()
    rates = loans["rate"].to_numpy()
    loan_idx = perf["i"].to_numpy()

    o = pd.DataFrame(
        {
            "credit_score": loans["score"].astype(str),
            "first_payment_date": [ym(m + 1) for m in loans["orig"]],
            "maturity_date": [
                ym(m + 1 + t) if m + 1 + t < len(months) else ""
                for m, t in zip(loans["orig"], loans["term"], strict=True)
            ],
            "number_of_units": "1",
            "occupancy_status": "P",
            "original_cltv": loans["ltv"].astype(str),
            "original_dti": loans["dti"].astype(str),
            "original_upb": loans["upb0"].astype(str),
            "original_ltv": loans["ltv"].astype(str),
            "original_interest_rate": loans["rate"].map("{:.3f}".format),
            "channel": "R",
            "amortization_type": "FRM",
            "property_state": loans["state"],
            "property_type": "SF",
            "loan_sequence_number": ids,
            "loan_purpose": "P",
            "original_loan_term": loans["term"].astype(str),
            "number_of_borrowers": "1",
            "seller_name": "OTHER",
            "servicer_name": "OTHER",
            "harp_indicator": "N",
            "property_valuation_method": "2",
        }
    )
    cols = internal_names(cfg["orig"])
    o_out = pd.DataFrame({c: o[c] if c in o else "" for c in cols})
    o_out.to_csv(out / "orig_SIM.txt", sep="|", header=False, index=False)

    status = perf["dpd"].map(lambda x: f"{x:02d}" if layout == "v2" else str(x))
    sign = cfg["loss_sign"]
    p = pd.DataFrame(
        {
            "loan_sequence_number": ids[loan_idx],
            "monthly_reporting_period": [ym(m) for m in perf["month"]],
            "current_actual_upb": perf["upb"].map(lambda x: "" if np.isnan(x) else f"{x:.2f}"),
            "current_loan_delinquency_status": status,
            "loan_age": perf["age"].astype(str),
            "remaining_months_to_legal_maturity": (terms[loan_idx] - perf["age"]).astype(str),
            "zero_balance_code": perf["zero"],
            "zero_balance_effective_date": [
                ym(m) if z else "" for m, z in zip(perf["month"], perf["zero"], strict=True)
            ],
            "current_interest_rate": rates[loan_idx].round(3).astype(str),
            "zero_balance_removal_upb": perf["removal"].map(lambda x: "" if np.isnan(x) else f"{x:.2f}"),
            "actual_loss_calculation": perf["loss"].map(lambda x: "" if np.isnan(x) else f"{x * sign:.2f}"),
            "servicer_name": "OTHER",
        }
    )
    cols = internal_names(cfg["perf"])
    p_out = pd.DataFrame({c: p[c] if c in p else "" for c in cols})
    p_out.to_csv(out / "perf_SIM.txt", sep="|", header=False, index=False)
    return out / "orig_SIM.txt", out / "perf_SIM.txt"


def make_person_period(n_loans=3000, seed=0, n_months=120):
    """Loan-month rows of live, non-defaulted loans with the next month's delinquency state.

    Same content as the `hazard_obs` table (plus `default_12m`), built without DuckDB for tests.
    """
    loans, perf = simulate(n_loans, n_months, seed)
    u, hpi, _, quarters, gdp = macro_paths(n_months, seed)
    months = _months(FIRST_MONTH, n_months)
    macro = build_macro_table(
        pd.Series(u, index=months), pd.Series(hpi, index=months), pd.Series(gdp, index=quarters)
    ).set_index("month")

    perf = perf.sort_values(["i", "month"]).reset_index(drop=True)
    first_default = perf[perf["dpd"] >= 3].groupby("i")["month"].min()
    nxt = perf.groupby("i")["dpd"].shift(-1)
    has_next = perf.groupby("i")["month"].shift(-1) == perf["month"] + 1
    rows = perf[has_next & (perf["dpd"] < 3) & (perf["zero"] == "")].copy()
    rows["next_dpd_months"] = nxt[rows.index].clip(upper=3).astype(int)
    rows = rows.merge(loans, on="i")
    m = macro.reindex(months[rows["month"].to_numpy()])
    fd = rows["i"].map(first_default)
    out = pd.DataFrame(
        {
            "loan_id": rows["i"].to_numpy(),
            "month": rows["month"].to_numpy(),
            "months_on_book": rows["age"].to_numpy(),
            "dpd_months": rows["dpd"].to_numpy(),
            "next_dpd_months": rows["next_dpd_months"].to_numpy(),
            "credit_score": rows["score"].to_numpy(),
            "orig_ltv": rows["ltv"].to_numpy(),
            "orig_dti": rows["dti"].to_numpy(),
            "orig_rate": rows["rate"].to_numpy() / 100.0,
            "term_months": rows["term"].to_numpy(),
            "current_upb": rows["upb"].to_numpy(),
            "unemployment": m["unemployment"].to_numpy(),
            "hpi_yoy": m["hpi_yoy"].to_numpy(),
            "gdp_yoy": m["gdp_yoy"].to_numpy(),
            "default_12m": ((fd > rows["month"]) & (fd <= rows["month"] + 12)).astype(int).to_numpy(),
        }
    )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulate Freddie Mac style files and FRED macro CSVs.")
    ap.add_argument("--outdir", default="data/sim")
    ap.add_argument("--loans", type=int, default=15000)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="v2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    loans, perf = simulate(args.loans, seed=args.seed)
    orig_path, perf_path = to_freddie_files(loans, perf, args.outdir, args.layout)
    write_fred_csvs(Path(args.outdir) / "fred", seed=args.seed)
    print("*" * 70)
    print("SIMULATED DATA: results describe the simulation, not the mortgage market.")
    print("*" * 70)
    print(f"{len(loans)} loans, {len(perf)} performance rows, layout {args.layout}")
    print(orig_path, perf_path, Path(args.outdir) / "fred", sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
