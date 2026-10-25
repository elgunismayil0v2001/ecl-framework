"""Build the DuckDB database: schema, data load, panel and features, quality report.

Examples:
    python -m src.data --synthetic
    python -m src.data --orig "data/raw/sample_orig_*.txt" --perf "data/raw/sample_svcg_*.txt" \
        --fred-dir data/raw/fred
"""
import argparse
from pathlib import Path

from src.load_freddie import load_freddie, validate_inputs
from src.load_macro import check_fred_files, load_macro
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
        """INSERT INTO performance (loan_id, report_month, current_upb, dpd_months, zero_bal_code)
           SELECT loan_id, CAST(report_month AS DATE), current_upb, dpd_months, zero_bal_code
           FROM df_perf"""
    )
    con.unregister("df_loans")
    con.unregister("df_perf")


def build_panel(con):
    run_sql(con, "02_panel.sql")
    run_sql(con, "03_features.sql")


def build_lgd_data(con):
    """LGD sample. Needs loans, performance and macro only."""
    run_sql(con, "07_lgd_data.sql")


def build_hazard_data(con):
    """Hazard model sample. Needs the panel, so run build_panel first."""
    run_sql(con, "08_hazard_data.sql")


def build_lifetime_data(con):
    """Tables for the LGD model and the lifetime PD (hazard) model. Run build_panel first."""
    build_lgd_data(con)
    build_hazard_data(con)


def write_data_source(con, label, outdir="reports"):
    """Record what data the results come from, so `make report` can state it next to the results."""
    import pandas as pd

    loans, rows, first, last = con.execute(
        """SELECT (SELECT COUNT(*) FROM loans), COUNT(*), MIN(report_month), MAX(report_month)
           FROM performance"""
    ).fetchone()
    table = pd.DataFrame(
        [{"source": label, "loans": loans, "loan_months": rows, "first_month": first, "last_month": last}]
    )
    path = Path(outdir) / "tables"
    path.mkdir(parents=True, exist_ok=True)
    table.to_csv(path / "data_source.csv", index=False)
    return table


def main(argv=None):
    from src.data_quality import format_report, run_checks

    ap = argparse.ArgumentParser(description="Build the loan database.")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke test)")
    ap.add_argument("--orig", help="glob(s), comma separated, for Freddie Mac origination files")
    ap.add_argument("--perf", help="glob(s), comma separated, for Freddie Mac performance files")
    ap.add_argument("--fred-dir", help="folder with UNRATE.csv, CSUSHPINSA.csv, GDPC1.csv")
    ap.add_argument("--lag-months", type=int, default=0, help="macro lag to avoid look-ahead")
    ap.add_argument("--source", default="unspecified", help="label of the data source, shown in the reports")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args(argv)

    if not args.synthetic:
        if not (args.orig and args.perf):
            ap.error("use --synthetic, or both --orig and --perf")
        *_, layout = validate_inputs(args.orig, args.perf)  # fail before the database is touched
        print("Freddie Mac file layout:", layout)
    if args.fred_dir:
        check_fred_files(args.fred_dir)

    con = connect(args.db)
    apply_schema(con)
    if args.synthetic:
        load_synthetic(con)
    else:
        print(load_freddie(con, args.orig, args.perf))
    if args.fred_dir:
        print("macro rows:", load_macro(con, args.fred_dir, lag_months=args.lag_months))
    build_panel(con)
    build_lifetime_data(con)
    write_data_source(con, args.source, args.outdir)

    report = run_checks(con)
    print(format_report(report))
    return 0 if all(c["passed"] for c in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
