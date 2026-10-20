"""PD pipeline: split, fit scorecard and benchmark, run validation gates, write scores.

Run after `python -m src.data ...`:
    python -m src.pd_pipeline --db ifrs9.duckdb
"""
import argparse
from pathlib import Path

import pandas as pd

from src.benchmark import BenchmarkModel, plot_importance
from src.data import DB_PATH, connect
from src.gates import evaluate_pd_model
from src.pd_model import Scorecard
from src.splits import time_based_split

DEFAULT_FEATURES = [
    "credit_score",
    "orig_ltv",
    "orig_dti",
    "orig_rate",
    "balance_ratio",
    "months_on_book",
    "dpd_months",
    "max_dpd_last_12m",
    "unemployment",
    "hpi_yoy",
    "gdp_yoy",
]


def usable_features(df, features):
    """Features that exist and have at least one non-missing value (macro may be absent)."""
    return [f for f in features if f in df.columns and df[f].notna().any()]


def run_pd_models(
    df,
    features=None,
    time_col="report_month",
    target_col="default_12m",
    oot_periods=6,
    gap_periods=12,
    use_benchmark=True,
    thresholds=None,
    scorecard_params=None,
):
    train, oot = time_based_split(df, time_col, oot_periods, gap_periods)
    feats = usable_features(train, features or DEFAULT_FEATURES)

    scorecard = Scorecard(feats, **(scorecard_params or {})).fit(train, target_col)
    result = {
        "split": {
            "n_train": len(train),
            "n_oot": len(oot),
            "train_default_rate": float(train[target_col].mean()),
            "oot_default_rate": float(oot[target_col].mean()),
            "train_last_period": str(train[time_col].max()),
            "oot_first_period": str(oot[time_col].min()),
        },
        "features": feats,
        "scorecard": scorecard,
        "benchmark": None,
        "reports": {},
    }
    result["reports"]["scorecard"] = evaluate_pd_model(
        oot[target_col].to_numpy(),
        scorecard.predict_proba(oot),
        scorecard.predict_proba(train),
        thresholds,
    )

    if use_benchmark:
        try:
            benchmark = BenchmarkModel(feats).fit(train, target_col)
        except ImportError:
            result["benchmark_note"] = "lightgbm is not installed, benchmark skipped"
        else:
            result["benchmark"] = benchmark
            result["reports"]["benchmark"] = evaluate_pd_model(
                oot[target_col].to_numpy(),
                benchmark.predict_proba(oot),
                benchmark.predict_proba(train),
                thresholds,
            )
    return result


def load_training_frame(con):
    return con.execute("SELECT * FROM features").df()


def write_scores(con, model, table="scores"):
    """Score every performing loan-month and write `scores` (pd_12m, pd_12m_orig).

    pd_12m_orig is the PD at the first observation of each loan (proxy for PD at initial recognition).
    """
    df = con.execute("SELECT * FROM features_scoring").df()
    scored = pd.DataFrame(
        {"loan_id": df["loan_id"], "report_month": df["report_month"], "pd_12m": model.predict_proba(df)}
    )
    con.execute("DROP VIEW IF EXISTS staging")
    con.register("df_scores_raw", scored)
    con.execute(
        """CREATE OR REPLACE TABLE scores_raw AS
           SELECT loan_id, CAST(report_month AS DATE) AS report_month, pd_12m FROM df_scores_raw"""
    )
    con.unregister("df_scores_raw")
    con.execute(
        f"""CREATE OR REPLACE TABLE {table} AS
            SELECT loan_id, report_month, pd_12m,
                   FIRST_VALUE(pd_12m) OVER (PARTITION BY loan_id ORDER BY report_month) AS pd_12m_orig
            FROM scores_raw"""
    )
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def format_gate_report(name, report):
    lines = [f"{name}:"]
    for metric, r in report.items():
        if metric == "all_passed":
            continue
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"  [{status}] {metric}: {r['value']:.4f} (needs {r['rule']})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit PD models, validate them and write scores.")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--oot-periods", type=int, default=6)
    ap.add_argument("--gap-periods", type=int, default=12)
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--strict", action="store_true", help="exit with an error if a gate fails")
    args = ap.parse_args(argv)

    con = connect(args.db)
    df = load_training_frame(con)
    result = run_pd_models(
        df,
        oot_periods=args.oot_periods,
        gap_periods=args.gap_periods,
        use_benchmark=not args.no_benchmark,
    )
    print("Split:", result["split"])
    print("Features used:", result["features"])

    scorecard = result["scorecard"]
    out = Path(args.outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    scorecard.iv_table().to_csv(out / "tables" / "pd_iv_table.csv", index=False)
    scorecard.bin_tables().to_csv(out / "tables" / "pd_woe_bins.csv", index=False)
    print(scorecard.iv_table().to_string(index=False))

    for name, report in result["reports"].items():
        print(format_gate_report(name, report))
    if result["benchmark"] is not None:
        scoring = result["benchmark"].importance(df)
        plot_importance(scoring, out / "figures" / "pd_benchmark_importance.png")
    elif "benchmark_note" in result:
        print(result["benchmark_note"])

    print("scores written:", write_scores(con, scorecard))
    ok = all(r["all_passed"] for r in result["reports"].values())
    return 1 if (args.strict and not ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
