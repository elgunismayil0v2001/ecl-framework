"""SQL pipeline tests on synthetic data. Skipped if DuckDB is not installed."""
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from src.synthetic import make_loan_panel  # noqa: E402

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def _build(upto="03_features.sql"):
    con = duckdb.connect(":memory:")
    loans, perf = make_loan_panel(n_loans=300, months=36, seed=11)
    for f in sorted(SQL_DIR.glob("*.sql")):
        if f.name == "01_schema.sql":
            con.execute(f.read_text())
            con.register("df_loans", loans)
            con.register("df_perf", perf)
            con.execute("INSERT INTO loans SELECT * FROM df_loans")
            con.execute("INSERT INTO performance SELECT * FROM df_perf")
        elif f.name <= upto:
            con.execute(f.read_text())
    return con, loans, perf


def test_panel_keeps_every_performance_row():
    con, _, perf = _build()
    assert con.execute("SELECT COUNT(*) FROM panel").fetchone()[0] == len(perf)


def test_modelling_sample_has_no_defaulted_observations():
    con, _, _ = _build()
    assert con.execute("SELECT COUNT(*) FROM panel_target WHERE in_default = 1").fetchone()[0] == 0


def test_target_is_binary_and_consistent_with_default_month():
    con, _, _ = _build()
    values = {r[0] for r in con.execute("SELECT DISTINCT default_12m FROM panel_target").fetchall()}
    assert values <= {0, 1}
    bad = con.execute(
        """SELECT COUNT(*) FROM panel_target
           WHERE default_12m = 1
             AND date_diff('month', report_month, first_default_month) NOT BETWEEN 1 AND 12"""
    ).fetchone()[0]
    assert bad == 0


def test_features_are_unique_per_loan_month_and_match_sample():
    con, _, _ = _build()
    n_sample = con.execute("SELECT COUNT(*) FROM panel_target").fetchone()[0]
    n_feat = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    n_unique = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT loan_id, report_month FROM features)"
    ).fetchone()[0]
    assert n_feat == n_sample == n_unique


def test_staging_assigns_stage_3_to_all_defaulted_rows():
    con, _, _ = _build()
    con.execute(
        """CREATE TABLE scores AS
           SELECT loan_id, report_month, 0.02 AS pd_12m, 0.02 AS pd_12m_orig FROM features"""
    )
    con.execute((SQL_DIR / "04_staging.sql").read_text())
    stages = {r[0] for r in con.execute("SELECT DISTINCT stage FROM staging").fetchall()}
    assert stages <= {1, 2, 3}
    mismatch = con.execute(
        """SELECT COUNT(*) FROM staging s JOIN panel p USING (loan_id, report_month)
           WHERE (p.in_default = 1) <> (s.stage = 3)"""
    ).fetchone()[0]
    assert mismatch == 0
