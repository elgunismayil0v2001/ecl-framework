"""Build the DuckDB database: schema, data load, panel and features, quality report.

Examples:
    python -m src.data --synthetic
    python -m src.data --orig "data/raw/sample_orig_*.txt" --perf "data/raw/sample_svcg_*.txt" \
        --fred-dir data/raw/fred
"""
import argparse
from pathlib import Path

from src.load_freddie import load_freddie
from src.load_macro import load_macro
from src.synthetic import make_loan_panel

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
DB_PATH = Path(__file__).resolve().parents[1] / "ifrs9.duckdb"


def connect(path=DB_PATH):
    import duckdb  # imported here so the pure-Python helpers work without DuckDB installed

    return duckdb.connect(str(path))


def run_sql(con, filename):
    con.execute((SQL_DIR / filename).read_text())


def apply_schema(con):
    run_sql(con, "01_schema.sql")


def load_synthetic(con, n_loans=300, months=36, seed=11):
    loans, perf = make_loan_panel(n_loans=n_loans, months=months, seed=seed)
    con.register("df_loans", loans)
    con.register("df_perf", perf)
    con.execute(
        """INSERT INTO loans
           SELECT loan_id, CAST(orig_date AS DATE), orig_upb, orig_ltv, orig_dti,
                  credit_score, orig_rate, term_months, property_state
           FROM df_loans"""
    )
    con.execute(
        """INSERT INTO performance
           SELECT loan_id, CAST(report_month AS DATE), current_upb, dpd_months, zero_bal_code
           FROM df_perf"""
    )
    con.unregister("df_loans")
    con.unregister("df_perf")


def build_panel(con):
    run_sql(con, "02_panel.sql")
    run_sql(con, "03_features.sql")


def main(argv=None):
    from src.data_quality import format_report, run_checks

    ap = argparse.ArgumentParser(description="Build the loan database.")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke test)")
    ap.add_argument("--orig", help="glob for Freddie Mac origination files")
    ap.add_argument("--perf", help="glob for Freddie Mac performance files")
    ap.add_argument("--fred-dir", help="folder with UNRATE.csv, CSUSHPINSA.csv, GDPC1.csv")
    ap.add_argument("--lag-months", type=int, default=0, help="macro lag to avoid look-ahead")
    args = ap.parse_args(argv)

    con = connect(args.db)
    apply_schema(con)
    if args.synthetic:
        load_synthetic(con)
    elif args.orig and args.perf:
        print(load_freddie(con, args.orig, args.perf))
    else:
        ap.error("use --synthetic, or both --orig and --perf")
    if args.fred_dir:
        print("macro rows:", load_macro(con, args.fred_dir, lag_months=args.lag_months))
    build_panel(con)

    report = run_checks(con)
    print(format_report(report))
    return 0 if all(c["passed"] for c in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
