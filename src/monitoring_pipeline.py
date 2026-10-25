"""Monitoring pipeline: reads the scores written by `make pd` and the staging view.

Run:  python -m src.monitoring_pipeline --db ifrs9.duckdb
"""
import argparse
from pathlib import Path

import pandas as pd

from src.data import DB_PATH, connect, run_sql
from src.monitoring import (
    add_quarter,
    backtest_by_period,
    feature_psi,
    migration_matrix,
    psi_over_time,
)
from src.pd_pipeline import DEFAULT_FEATURES

SCORES_SQL = """
SELECT s.report_month, s.pd_12m, f.default_12m
FROM scores s
JOIN features f ON f.loan_id = s.loan_id AND f.report_month = s.report_month
"""

CONSISTENCY_SQL = """
SELECT AVG(s.pd_12m) AS scorecard_pd, AVG(l.pd_12m) AS hazard_pd,
       CORR(s.pd_12m, l.pd_12m) AS correlation, COUNT(*) AS n
FROM loan_ecl l
JOIN scores s ON s.loan_id = l.loan_id AND s.report_month = l.report_month
WHERE l.scenario = 'base' AND l.stage < 3
"""


def _exists(con, name):
    sql = f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}'"
    return con.execute(sql).fetchone()[0] > 0


def _plot_psi(table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pd.to_datetime(table["period"]), table["psi"], marker="o")
    ax.axhline(0.10, linestyle="--", linewidth=1)
    ax.axhline(0.25, linestyle="--", linewidth=1)
    ax.set_title("PD score PSI by month vs reference window")
    ax.set_ylabel("PSI")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_backtest(table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["quarter"], table["mean_pd"] * 100, marker="o", label="predicted")
    ax.plot(table["quarter"], table["observed_rate"] * 100, marker="s", label="observed")
    ax.set_title("12-month PD: predicted vs observed by quarter")
    ax.set_ylabel("Default rate (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_monitoring(con, outdir="reports", ref_periods=12, recent_periods=6, max_feature_rows=2_000_000):
    if not _exists(con, "scores"):
        raise RuntimeError("Table `scores` not found. Run `make pd` first.")
    out = Path(outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    notes = []

    scores = con.execute(SCORES_SQL).df()
    psi_table = psi_over_time(scores, "pd_12m", "report_month", ref_periods)
    backtest = backtest_by_period(add_quarter(scores), "pd_12m", "default_12m", "quarter")
    psi_table.to_csv(out / "tables" / "monitoring_psi_by_month.csv", index=False)
    backtest.to_csv(out / "tables" / "monitoring_backtest_by_quarter.csv", index=False)
    if len(psi_table):
        _plot_psi(psi_table, out / "figures" / "monitoring_psi.png")
    _plot_backtest(backtest, out / "figures" / "monitoring_backtest.png")

    cols = ", ".join(["report_month", *DEFAULT_FEATURES])
    n = con.execute("SELECT COUNT(*) FROM features_scoring").fetchone()[0]
    sample = ""
    if n > max_feature_rows:
        sample = f" USING SAMPLE reservoir({max_feature_rows} ROWS) REPEATABLE (42)"
    feats = con.execute(f"SELECT {cols} FROM features_scoring{sample}").df()
    usable = [f for f in DEFAULT_FEATURES if feats[f].notna().any()]
    feature_table = feature_psi(feats, usable, "report_month", ref_periods, recent_periods)
    feature_table.to_csv(out / "tables" / "monitoring_feature_psi.csv", index=False)

    run_sql(con, "04_staging.sql")
    run_sql(con, "06_monitoring.sql")
    counts = con.execute("SELECT stage, next_stage, n FROM stage_migration").df()
    migration_counts, migration_probs = migration_matrix(counts)
    migration_probs.to_csv(out / "tables" / "monitoring_stage_migration.csv")

    consistency = None
    if _exists(con, "loan_ecl"):
        consistency = con.execute(CONSISTENCY_SQL).df().iloc[0].to_dict()
        pd.DataFrame([consistency]).to_csv(out / "tables" / "monitoring_pd_consistency.csv", index=False)
    else:
        notes.append("loan_ecl not found: run `make ecl` to compare the scorecard PD with the ECL-engine PD")

    return {
        "psi_by_month": psi_table,
        "backtest": backtest,
        "feature_psi": feature_table,
        "migration_counts": migration_counts,
        "migration_probs": migration_probs,
        "consistency": consistency,
        "summary": {
            "psi_red_months": int((psi_table["flag"] == "red").sum()),
            "backtest_red_quarters": int((backtest["flag"] == "red").sum()),
            "feature_psi_red": int((feature_table["flag"] == "red").sum()),
        },
        "notes": notes,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="PSI over time, PD backtest, stage migration.")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--ref-periods", type=int, default=12)
    ap.add_argument("--recent-periods", type=int, default=6)
    args = ap.parse_args(argv)

    r = run_monitoring(connect(args.db), args.outdir, args.ref_periods, args.recent_periods)
    print("Backtest by quarter:")
    print(r["backtest"].round(4).to_string(index=False))
    print("Feature PSI:")
    print(r["feature_psi"].round(4).to_string(index=False))
    print("Stage migration probabilities:")
    print(r["migration_probs"].round(4).to_string())
    if r["consistency"]:
        c = r["consistency"]
        print(
            f"PD consistency: scorecard {c['scorecard_pd']:.4f} vs roll-rate model {c['hazard_pd']:.4f}, "
            f"correlation {c['correlation']:.3f}"
        )
    print("Summary:", r["summary"])
    for note in r["notes"]:
        print("NOTE:", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
