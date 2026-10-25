"""Week 1 pipeline tests: loaders, data quality checks, EDA. Skipped if DuckDB is not installed."""
import datetime as dt
import os
import tempfile

import pytest

duckdb = pytest.importorskip("duckdb")

from src import data  # noqa: E402
from src.data_quality import run_checks  # noqa: E402
from src.eda import default_rate_by_band, make_figures, vintage_default_rate  # noqa: E402
from src.freddie_schema import (  # noqa: E402
    ORIG_COLUMNS,
    ORIG_COLUMNS_V2,
    PERF_COLUMNS,
    PERF_COLUMNS_V2,
)
from src.load_freddie import load_freddie  # noqa: E402
from src.load_macro import load_macro  # noqa: E402


def _synthetic_con():
    con = duckdb.connect(":memory:")
    data.apply_schema(con)
    data.load_synthetic(con)
    data.build_panel(con)
    return con


def _row(columns, **values):
    row = dict.fromkeys(columns, "")
    row.update(values)
    return "|".join(row[c] for c in columns)


def _orig(**values):
    base = {
        "credit_score": "740",
        "first_payment_date": "201803",
        "original_dti": "35",
        "original_upb": "200000",
        "original_ltv": "80",
        "original_interest_rate": "4.5",
        "property_state": "CA",
        "loan_sequence_number": "F1",
        "original_loan_term": "360",
    }
    base.update(values)
    return _row(ORIG_COLUMNS, **base)


def _perf(loan, period, status, upb="199000", zero=""):
    return _row(
        PERF_COLUMNS,
        loan_sequence_number=loan,
        monthly_reporting_period=period,
        current_actual_upb=upb,
        current_loan_delinquency_status=status,
        zero_balance_code=zero,
    )


def test_quality_checks_pass_on_synthetic_data():
    report = run_checks(_synthetic_con())
    failed = [c for c in report if not c["passed"]]
    assert not failed, failed


def test_quality_checks_detect_duplicate_loan():
    con = _synthetic_con()
    con.execute("INSERT INTO loans SELECT * FROM loans LIMIT 1")
    report = {c["check"]: c for c in run_checks(con)}
    assert not report["loan_id_unique"]["passed"]


def test_no_features_in_right_censored_window():
    con = _synthetic_con()
    n = con.execute(
        """SELECT COUNT(*) FROM features
           WHERE report_month > CAST((SELECT MAX(report_month) FROM panel) - INTERVAL 12 MONTH AS DATE)"""
    ).fetchone()[0]
    assert n == 0


def test_eda_tables_and_figures():
    con = _synthetic_con()
    assert len(vintage_default_rate(con)) > 0
    bands = default_rate_by_band(con, "credit_score", [620, 660, 700, 740, 780])
    assert bands["default_rate_12m"].between(0, 1).all()
    with tempfile.TemporaryDirectory() as d:
        paths = make_figures(con, d)
        assert len(paths) == 3
        assert all(os.path.exists(p) for p in paths)


def test_freddie_loader_cleans_fake_files():
    with tempfile.TemporaryDirectory() as d:
        orig_path = os.path.join(d, "sample_orig_2018.txt")
        perf_path = os.path.join(d, "sample_svcg_2018.txt")
        with open(orig_path, "w") as f:
            f.write(_orig() + "\n")
            f.write(
                _orig(
                    loan_sequence_number="F2",
                    credit_score="9999",
                    original_ltv="999",
                    original_dti="999",
                )
                + "\n"
            )
        with open(perf_path, "w") as f:
            f.write(_perf("F1", "201804", "0") + "\n")
            f.write(_perf("F1", "201805", "1") + "\n")
            f.write(_perf("F1", "201806", "3") + "\n")
            f.write(_perf("F1", "201807", "RA", zero="09") + "\n")
            f.write(_perf("F2", "201804", "XX") + "\n")
            f.write(_perf("F2", "201805", "0") + "\n")
        con = duckdb.connect(":memory:")
        data.apply_schema(con)
        orig_glob = os.path.join(d, "sample_orig_*.txt")
        perf_glob = os.path.join(d, "sample_svcg_*.txt")
        counts = load_freddie(con, orig_glob, perf_glob)

    assert counts == {"loans": 2, "performance_rows": 5, "layout": "v1"}
    f1 = con.execute("SELECT * FROM loans WHERE loan_id = 'F1'").fetchone()
    assert f1[1] == dt.date(2018, 2, 1)  # first payment 2018-03 minus one month
    assert f1[5] == 740
    assert abs(f1[6] - 0.045) < 1e-9
    f2 = con.execute(
        "SELECT credit_score, orig_ltv, orig_dti FROM loans WHERE loan_id = 'F2'"
    ).fetchone()
    assert f2 == (None, None, None)
    dpd = dict(
        con.execute("SELECT report_month, dpd_months FROM performance WHERE loan_id = 'F1'").fetchall()
    )
    assert dpd[dt.date(2018, 7, 1)] == 3  # RA mapped to default
    assert dpd[dt.date(2018, 5, 1)] == 1
    n_f2 = con.execute("SELECT COUNT(*) FROM performance WHERE loan_id = 'F2'").fetchone()[0]
    assert n_f2 == 1  # the XX row is dropped


def test_freddie_loader_rejects_wrong_column_count():
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "sample_orig_2018.txt")
        with open(bad, "w") as f:
            f.write("a|b|c\n")
        con = duckdb.connect(":memory:")
        data.apply_schema(con)
        with pytest.raises(ValueError):
            load_freddie(con, bad, bad)


def test_macro_loader_fills_table_and_joins_into_features():
    with tempfile.TemporaryDirectory() as d:
        months = [f"{y}-{m:02d}-01" for y in (2017, 2018, 2019, 2020, 2021) for m in range(1, 13)]
        quarters = [f"{y}-{m:02d}-01" for y in (2017, 2018, 2019, 2020, 2021) for m in (1, 4, 7, 10)]
        with open(os.path.join(d, "UNRATE.csv"), "w") as f:
            f.write("observation_date,UNRATE\n" + "\n".join(f"{m},4.0" for m in months) + "\n")
        with open(os.path.join(d, "CSUSHPINSA.csv"), "w") as f:
            rows = "\n".join(f"{m},{100 * 1.005**i}" for i, m in enumerate(months))
            f.write("observation_date,CSUSHPINSA\n" + rows + "\n")
        with open(os.path.join(d, "GDPC1.csv"), "w") as f:
            rows = "\n".join(f"{q},{100 * 1.01**i}" for i, q in enumerate(quarters))
            f.write("observation_date,GDPC1\n" + rows + "\n")
        con = duckdb.connect(":memory:")
        data.apply_schema(con)
        data.load_synthetic(con)
        n = load_macro(con, d)
        data.build_panel(con)
    assert n > 0
    share = con.execute(
        "SELECT AVG(CASE WHEN unemployment IS NOT NULL THEN 1.0 ELSE 0.0 END) FROM features"
    ).fetchone()[0]
    assert share > 0.99
    rows = {c["check"]: c for c in run_checks(con)}
    assert rows["macro_coverage"]["passed"]


def test_freddie_loader_reads_the_release_47_layout():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "orig_2019Q1.txt"), "w") as f:
            f.write(
                _row(
                    ORIG_COLUMNS_V2,
                    credit_score="740",
                    first_payment_date="201903",
                    original_dti="35",
                    original_upb="200000",
                    original_ltv="80",
                    original_interest_rate="4.5",
                    property_state="CA",
                    loan_identifier="F119Q1000001",
                    original_loan_term="360",
                )
                + "\n"
            )

        def perf(period, status, zero="", removal="", loss=""):
            return _row(
                PERF_COLUMNS_V2,
                loan_identifier="F119Q1000001",
                period=period,
                current_actual_upb="190000",
                current_loan_delinquency_status=status,
                zero_balance_code=zero,
                zero_balance_removal_upb=removal,
                actual_loss=loss,
            )

        rows = [
            perf("201904", "00"),
            perf("201905", "01"),
            perf("201906", "03"),
            perf("201907", "XX"),
            perf("201908", "RA", zero="09", removal="100000", loss="20000"),
        ]
        with open(os.path.join(d, "perf_2019Q1.txt"), "w") as f:
            f.write("\n".join(rows) + "\n")
        con = duckdb.connect(":memory:")
        data.apply_schema(con)
        counts = load_freddie(con, os.path.join(d, "orig_*.txt"), os.path.join(d, "perf_*.txt"))
        data.build_lgd_data(con)
    assert counts == {"loans": 1, "performance_rows": 4, "layout": "v2"}
    dpd = dict(con.execute("SELECT report_month, dpd_months FROM performance").fetchall())
    assert dpd[dt.date(2019, 4, 1)] == 0 and dpd[dt.date(2019, 5, 1)] == 1
    assert dpd[dt.date(2019, 6, 1)] == 3 and dpd[dt.date(2019, 8, 1)] == 3
    loan = con.execute("SELECT loan_id, orig_date FROM loans").fetchone()
    assert loan == ("F119Q1000001", dt.date(2019, 2, 1))
    lgd = con.execute("SELECT lgd FROM lgd_data").fetchone()[0]
    assert abs(lgd - 0.2) < 1e-9  # in this layout a positive loss is a loss
