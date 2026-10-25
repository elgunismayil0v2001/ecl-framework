"""Week 3 pipeline: lifetime PD curves by scenario, LGD model and EAD path.

Run after `python -m src.data ...` (which also builds hazard_obs and lgd_data):
    python -m src.lifetime_pipeline --db ifrs9.duckdb
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DB_PATH, connect
from src.ead import ead_paths
from src.lgd import LGDModel, lgd_by_band
from src.lifetime_pd import (
    MACRO_COLS,
    lifetime_curves,
    make_scenarios,
    scenario_ordering_ok,
)
from src.rollrate import RollRateModel

LOANS_NOW_SQL = """
SELECT s.loan_id, s.report_month, s.months_on_book, s.dpd_months, s.credit_score, s.orig_ltv,
       s.orig_dti, s.orig_rate, l.term_months, p.current_upb
FROM features_scoring s
JOIN loans l ON l.loan_id = s.loan_id
JOIN panel p ON p.loan_id = s.loan_id AND p.report_month = s.report_month
WHERE s.report_month = (SELECT MAX(report_month) FROM panel) AND p.current_upb > 0
"""

LAST_MACRO_SQL = """
SELECT unemployment, hpi_yoy, gdp_yoy FROM macro
WHERE scenario = 'base' AND unemployment IS NOT NULL AND hpi_yoy IS NOT NULL AND gdp_yoy IS NOT NULL
ORDER BY month DESC LIMIT 1
"""

LTV_EDGES = [60, 70, 80, 90, 95]
MIN_LGD_ROWS = 200


def load_hazard_sample(con, max_rows=1_000_000, seed=42):
    """Hazard observations, sampled with a reservoir if there are more than `max_rows`."""
    n_obs = con.execute("SELECT COUNT(*) FROM hazard_obs").fetchone()[0]
    if n_obs > max_rows:
        sample = f"reservoir({int(max_rows)} ROWS) REPEATABLE ({int(seed)})"
        sql = f"SELECT * FROM hazard_obs USING SAMPLE {sample}"
    else:
        sql = "SELECT * FROM hazard_obs"
    return con.execute(sql).df()


def load_last_macro(con):
    """Last month with all three macro variables, or None if the macro table is empty."""
    row = con.execute(LAST_MACRO_SQL).fetchone()
    return None if row is None else dict(zip(MACRO_COLS, row, strict=True))


def _plot_curves(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in curves.groupby("scenario"):
        ax.plot(g["month"], g["cum_pd"] * 100, label=name)
    ax.set_title("Lifetime PD by scenario (portfolio average)")
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("Cumulative PD (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_lifetime(con, outdir="reports", horizon=120, max_rows=1_000_000, max_loans=20_000, cpr=0.06, seed=42):
    out = Path(outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    notes = []

    hazard_df = load_hazard_sample(con, max_rows, seed)
    model = RollRateModel().fit(hazard_df)
    loans = con.execute(LOANS_NOW_SQL).df()
    if len(loans) > max_loans:
        loans = loans.sample(max_loans, random_state=seed)

    last_macro = load_last_macro(con)
    if last_macro is None:
        notes.append("No macro data: all scenarios use the same flat macro values")
        last_macro = dict.fromkeys(MACRO_COLS, 0.0)
    if not any(c in model.features_ for c in MACRO_COLS):
        notes.append("The lifetime PD model has no macro features, so scenario curves will be identical")

    curves = lifetime_curves(model, loans, make_scenarios(last_macro, horizon), horizon)
    curves.to_csv(out / "tables" / "lifetime_pd_curves.csv", index=False)
    _plot_curves(curves, out / "figures" / "lifetime_pd_curves.png")
    by_state = pd.DataFrame({f"from_state_{k}": v for k, v in model.coef_by_state_.items()})
    by_state.to_csv(out / "tables" / "hazard_coefficients.csv")

    remaining = loans["term_months"] - loans["months_on_book"]
    ead = ead_paths(loans["current_upb"], loans["orig_rate"], remaining, horizon, cpr)
    ead_share = ead.sum(axis=0) / ead[:, 0].sum()
    ead_curve = pd.DataFrame({"month": np.arange(1, horizon + 1), "ead_share": ead_share})
    ead_curve.to_csv(out / "tables" / "ead_portfolio_path.csv", index=False)

    lgd_df = con.execute("SELECT * FROM lgd_data").df()
    lgd_model = None
    if len(lgd_df) >= MIN_LGD_ROWS:
        lgd_model = LGDModel().fit(lgd_df)
        lgd_by_band(lgd_model, lgd_df, "orig_ltv", LTV_EDGES).to_csv(
            out / "tables" / "lgd_by_ltv_band.csv", index=False
        )
    else:
        notes.append(
            f"Only {len(lgd_df)} resolved defaults with losses (need {MIN_LGD_ROWS}): LGD model skipped"
        )

    return {
        "hazard_model": model,
        "curves": curves,
        "ordering_ok": scenario_ordering_ok(curves),
        "lgd_model": lgd_model,
        "n_loans": len(loans),
        "notes": notes,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lifetime PD curves, LGD model and EAD path.")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--cpr", type=float, default=0.06, help="annual prepayment rate assumption")
    ap.add_argument("--max-rows", type=int, default=1_000_000)
    args = ap.parse_args(argv)

    result = run_lifetime(connect(args.db), args.outdir, args.horizon, args.max_rows, cpr=args.cpr)
    print("Roll-rate model coefficients, current -> worse (per standard deviation):")
    print(result["hazard_model"].coef_.round(3).to_string())
    last = result["curves"][result["curves"]["month"] == args.horizon].set_index("scenario")["cum_pd"]
    print(f"Cumulative PD at {args.horizon} months:", (last * 100).round(2).to_dict())
    print("Scenario ordering (downside > base > upside):", "OK" if result["ordering_ok"] else "FAILED")
    if result["lgd_model"] is not None:
        print("LGD coefficients (per standard deviation):")
        print(result["lgd_model"].coef_.round(3).to_string())
    for note in result["notes"]:
        print("NOTE:", note)
    return 0 if result["ordering_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
