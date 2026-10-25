"""Week 4 pipeline: staging, ECL by scenario, weighted ECL, RWA, stress test.

Run after `make data-freddie`, `make pd` (writes scores) and with hazard_obs / lgd_data built:
    python -m src.ecl_pipeline --db ifrs9.duckdb

The reporting date is the last month in the data. Portfolio = all loans with a balance at that date.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.capital import capital_sensitivity, el_vs_ecl, scenario_metrics, stress_test
from src.data import DB_PATH, connect, run_sql
from src.ecl_portfolio import (
    build_loan_ecl,
    coverage_by_stage,
    lifetime_horizon,
    weight_sensitivity,
    weighted_total,
)
from src.lgd import LGDModel
from src.lifetime_pd import MACRO_COLS, make_scenarios
from src.lifetime_pipeline import MIN_LGD_ROWS, load_hazard_sample, load_last_macro
from src.rollrate import RollRateModel
from src.staging import sicr_sensitivity

PERFORMING_SQL = """
SELECT s.loan_id, s.report_month, s.months_on_book, s.dpd_months, s.credit_score, s.orig_ltv,
       s.orig_dti, s.orig_rate, l.term_months, p.current_upb,
       st.stage, st.pd_12m, st.pd_12m_orig
FROM features_scoring s
JOIN loans l ON l.loan_id = s.loan_id
JOIN panel p ON p.loan_id = s.loan_id AND p.report_month = s.report_month
JOIN staging st ON st.loan_id = s.loan_id AND st.report_month = s.report_month
WHERE s.report_month = (SELECT MAX(report_month) FROM panel) AND p.current_upb > 0
"""

DEFAULTED_SQL = """
SELECT p.loan_id, p.report_month, p.months_on_book, p.dpd_months, l.credit_score, l.orig_ltv,
       l.orig_dti, l.orig_rate, l.term_months, p.current_upb,
       3 AS stage, CAST(NULL AS DOUBLE) AS pd_12m, CAST(NULL AS DOUBLE) AS pd_12m_orig
FROM panel p
JOIN loans l ON l.loan_id = p.loan_id
WHERE p.report_month = (SELECT MAX(report_month) FROM panel) AND p.in_default = 1 AND p.current_upb > 0
"""

WEIGHT_SETS = {
    "base only": {"base": 1.0, "upside": 0.0, "downside": 0.0},
    "60/10/30 (default)": None,
    "50/10/40": {"base": 0.5, "upside": 0.1, "downside": 0.4},
    "40/10/50": {"base": 0.4, "upside": 0.1, "downside": 0.5},
}


def _table_exists(con, name):
    sql = f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}'"
    return con.execute(sql).fetchone()[0] > 0


def load_portfolio(con):
    """Loans with a balance at the reporting date, with their stage."""
    if not _table_exists(con, "scores"):
        raise RuntimeError("Table `scores` not found. Run `make pd` first: it writes the PD scores.")
    run_sql(con, "04_staging.sql")
    performing = con.execute(PERFORMING_SQL).df()
    defaulted = con.execute(DEFAULTED_SQL).df()
    return pd.concat([performing, defaulted], ignore_index=True)


def load_weights(con):
    return dict(con.execute("SELECT scenario, weight FROM scenario_weights").fetchall())


def run_checks(loan_ecl, weights, metrics=None):
    """Sanity checks on the results. Each value is True when the check passes."""
    cov = coverage_by_stage(loan_ecl, weights)
    w = cov[cov["scenario"] == "weighted"].set_index("stage")["coverage"]
    totals = loan_ecl.groupby("scenario")["ecl"].sum()
    checks = {
        "weights_sum_to_one": abs(sum(weights.values()) - 1.0) < 1e-9,
        "ecl_between_zero_and_lgd_times_ead": bool(
            ((loan_ecl["ecl"] >= -1e-9) & (loan_ecl["ecl"] <= loan_ecl["lgd"] * loan_ecl["ead"] + 1e-6)).all()
        ),
        "all_three_stages_present": {1, 2, 3} <= set(w.index),
    }
    if {"downside", "base", "upside"} <= set(totals.index):
        checks["scenario_order_downside_base_upside"] = bool(
            totals["downside"] > totals["base"] > totals["upside"]
        )
    if {1, 2, 3} <= set(w.index):
        checks["coverage_stage3_gt_stage2_gt_stage1"] = bool(w[3] > w[2] > w[1])
    if metrics is not None and "base" in metrics:
        density = metrics["base"]["rwa_density"]
        checks["rwa_density_between_3_and_80_percent"] = bool(0.03 <= density <= 0.80)
    return checks


def run_ecl(
    con,
    outdir="reports",
    max_horizon=360,
    max_rows=1_000_000,
    max_loans=50_000,
    cpr=0.06,
    default_lgd=0.25,
    capital_ratio=0.12,
    seed=42,
):
    out = Path(outdir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    notes = []

    loans = load_portfolio(con)
    n_total = len(loans)
    if n_total > max_loans:
        loans = loans.sample(max_loans, random_state=seed).reset_index(drop=True)
        notes.append(f"Portfolio sampled: {len(loans)} of {n_total} loans (ratios are unaffected)")
    if len(loans) == 0:
        raise RuntimeError("No loans with a balance at the reporting date")

    hazard_model = RollRateModel().fit(load_hazard_sample(con, max_rows, seed))
    lgd_df = con.execute("SELECT * FROM lgd_data").df()
    if len(lgd_df) >= MIN_LGD_ROWS:
        lgd_model = LGDModel().fit(lgd_df)
    else:
        lgd_model = None
        notes.append(f"Only {len(lgd_df)} resolved defaults: constant LGD of {default_lgd:.0%} is used")

    last_macro = load_last_macro(con)
    if last_macro is None:
        notes.append("No macro data: all scenarios use the same flat macro values")
        last_macro = dict.fromkeys(MACRO_COLS, 0.0)
    horizon = lifetime_horizon(loans, max_horizon)
    scenarios = make_scenarios(last_macro, horizon)

    loan_ecl = build_loan_ecl(loans, hazard_model, lgd_model, scenarios, cpr, default_lgd)
    con.register("df_loan_ecl", loan_ecl)
    con.execute(
        """CREATE OR REPLACE TABLE loan_ecl AS
           SELECT loan_id, CAST(report_month AS DATE) AS report_month, scenario,
                  CAST(stage AS INTEGER) AS stage, ead, ecl, lgd, pd_12m, pd_horizon
           FROM df_loan_ecl"""
    )
    con.unregister("df_loan_ecl")
    run_sql(con, "05_ecl_aggregation.sql")

    weights = load_weights(con)
    coverage = coverage_by_stage(loan_ecl, weights)
    coverage.to_csv(out / "ecl_coverage_by_stage.csv", index=False)

    sets = {k: (weights if v is None else v) for k, v in WEIGHT_SETS.items()}
    weight_sensitivity(loan_ecl, sets).to_csv(out / "ecl_weight_sensitivity.csv", index=False)

    performing = loans[loans["stage"] < 3].assign(ead=lambda d: d["current_upb"])
    sicr_sensitivity(performing).to_csv(out / "sicr_sensitivity.csv", index=False)

    metrics = {s: scenario_metrics(loan_ecl, s) for s in scenarios}
    pd.DataFrame(metrics.values()).to_csv(out / "capital_by_scenario.csv", index=False)
    pd.DataFrame([el_vs_ecl(m) for m in metrics.values()]).to_csv(out / "el_vs_ecl.csv", index=False)
    stress = stress_test(metrics["base"], metrics["downside"], capital_ratio)
    pd.DataFrame([stress]).to_csv(out / "stress_test.csv", index=False)
    capital_sensitivity(metrics["base"], metrics["downside"]).to_csv(
        out / "capital_sensitivity.csv", index=False
    )

    return {
        "loan_ecl": loan_ecl,
        "coverage": coverage,
        "weighted_ecl": weighted_total(loan_ecl, weights),
        "metrics": metrics,
        "stress": stress,
        "checks": run_checks(loan_ecl, weights, metrics),
        "weights": weights,
        "notes": notes,
        "n_loans": len(loans),
        "horizon": horizon,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Staging, ECL by scenario, RWA and stress test.")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--cpr", type=float, default=0.06)
    ap.add_argument("--default-lgd", type=float, default=0.25)
    ap.add_argument("--capital-ratio", type=float, default=0.12, help="assumed starting capital / RWA")
    ap.add_argument("--max-loans", type=int, default=50_000)
    args = ap.parse_args(argv)

    r = run_ecl(
        connect(args.db),
        args.outdir,
        cpr=args.cpr,
        default_lgd=args.default_lgd,
        capital_ratio=args.capital_ratio,
        max_loans=args.max_loans,
    )
    print(f"Loans: {r['n_loans']}, horizon: {r['horizon']} months")
    print(r["coverage"].round(4).to_string(index=False))
    print(f"Probability-weighted ECL: {r['weighted_ecl']:,.0f}")
    s = r["stress"]
    print(
        f"Stress test: capital ratio {s['capital_ratio_start']:.1%} -> {s['capital_ratio_after']:.1%} "
        f"({s['change_pp']:+.2f} pp)"
    )
    for name, ok in r["checks"].items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for note in r["notes"]:
        print("NOTE:", note)
    return 0 if all(r["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
