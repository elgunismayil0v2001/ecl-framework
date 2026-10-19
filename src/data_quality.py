"""Data quality checks for the loan database. Run after loading and building the panel."""
import argparse

from src.data import DB_PATH, connect


def _scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def _table_exists(con, name):
    sql = f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}'"
    return _scalar(con, sql) > 0


def run_checks(con, max_missing_share=0.05):
    checks = []

    def add(name, value, passed, detail=""):
        checks.append({"check": name, "value": value, "passed": bool(passed), "detail": detail})

    n_loans = _scalar(con, "SELECT COUNT(*) FROM loans")
    n_perf = _scalar(con, "SELECT COUNT(*) FROM performance")
    add("loans_not_empty", n_loans, n_loans > 0)
    add("performance_not_empty", n_perf, n_perf > 0)

    dup_loans = _scalar(con, "SELECT COUNT(*) - COUNT(DISTINCT loan_id) FROM loans")
    add("loan_id_unique", dup_loans, dup_loans == 0, "duplicate loan ids")

    dup_perf = _scalar(
        con,
        """SELECT COUNT(*) FROM (
               SELECT loan_id, report_month FROM performance
               GROUP BY loan_id, report_month HAVING COUNT(*) > 1)""",
    )
    add("performance_key_unique", dup_perf, dup_perf == 0, "duplicate loan-month rows")

    orphans = _scalar(
        con,
        """SELECT COUNT(*) FROM performance p
           LEFT JOIN loans l ON p.loan_id = l.loan_id WHERE l.loan_id IS NULL""",
    )
    add("no_orphan_performance_rows", orphans, orphans == 0, "performance rows without a loan")

    bad_dates = _scalar(
        con,
        """SELECT COUNT(*) FROM loans
           WHERE orig_date < DATE '1990-01-01' OR orig_date > current_date""",
    )
    add("orig_date_in_range", bad_dates, bad_dates == 0)

    for col in ("credit_score", "orig_ltv", "orig_dti"):
        share = _scalar(con, f"SELECT AVG(CASE WHEN {col} IS NULL THEN 1.0 ELSE 0.0 END) FROM loans")
        share = 0.0 if share is None else float(share)
        add(
            f"missing_share_{col}",
            round(share, 4),
            share <= max_missing_share,
            f"max {max_missing_share}",
        )

    neg_dpd = _scalar(con, "SELECT COUNT(*) FROM performance WHERE dpd_months < 0")
    add("dpd_non_negative", neg_dpd, neg_dpd == 0)

    if _table_exists(con, "panel_target") and _table_exists(con, "features"):
        neg_mob = _scalar(con, "SELECT COUNT(*) FROM panel WHERE months_on_book < 0")
        add("months_on_book_non_negative", neg_mob, neg_mob == 0, "report month before origination")

        defaulted = _scalar(con, "SELECT COUNT(*) FROM panel_target WHERE in_default = 1")
        add("sample_excludes_defaulted_rows", defaulted, defaulted == 0)

        censored = _scalar(
            con,
            """SELECT COUNT(*) FROM features
               WHERE report_month >
                     CAST((SELECT MAX(report_month) FROM panel) - INTERVAL 12 MONTH AS DATE)""",
        )
        add("no_right_censored_observations", censored, censored == 0)

        dup_feat = _scalar(
            con,
            """SELECT COUNT(*) FROM (
                   SELECT loan_id, report_month FROM features
                   GROUP BY loan_id, report_month HAVING COUNT(*) > 1)""",
        )
        add("features_key_unique", dup_feat, dup_feat == 0)

        rate = _scalar(con, "SELECT AVG(default_12m) FROM features")
        rate = None if rate is None else float(rate)
        add(
            "default_rate_plausible",
            None if rate is None else round(rate, 4),
            rate is not None and 0.0 < rate < 0.20,
            "12-month default rate should be between 0% and 20%",
        )

        if _table_exists(con, "macro") and _scalar(con, "SELECT COUNT(*) FROM macro") > 0:
            cover = _scalar(
                con, "SELECT AVG(CASE WHEN unemployment IS NOT NULL THEN 1.0 ELSE 0.0 END) FROM features"
            )
            cover = 0.0 if cover is None else float(cover)
            add("macro_coverage", round(cover, 4), cover >= 0.90, "share of features rows with macro data")

    return checks


def format_report(checks):
    lines = []
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        extra = f"  ({c['detail']})" if c["detail"] and not c["passed"] else ""
        lines.append(f"[{status}] {c['check']}: {c['value']}{extra}")
    failed = sum(1 for c in checks if not c["passed"])
    lines.append(f"{len(checks) - failed}/{len(checks)} checks passed")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run data quality checks on the DuckDB database.")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args(argv)
    con = connect(args.db)
    report = run_checks(con)
    print(format_report(report))
    return 0 if all(c["passed"] for c in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
