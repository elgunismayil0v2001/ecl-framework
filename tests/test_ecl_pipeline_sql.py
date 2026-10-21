"""Week 4 SQL and pipeline tests on synthetic data. Skipped if DuckDB is not installed."""
import os
import tempfile

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from src import data  # noqa: E402
from src.ecl_pipeline import load_portfolio, run_ecl  # noqa: E402
from src.pd_pipeline import write_scores  # noqa: E402
from src.staging import assign_stage  # noqa: E402

TABLES = (
    "ecl_coverage_by_stage.csv",
    "ecl_weight_sensitivity.csv",
    "sicr_sensitivity.csv",
    "capital_by_scenario.csv",
    "el_vs_ecl.csv",
    "stress_test.csv",
    "capital_sensitivity.csv",
)


class AgeModel:
    """Stand-in PD model: PD rises with loan age, so many loans trigger the SICR rule."""

    def predict_proba(self, df):
        age = df["months_on_book"].to_numpy(dtype=float)
        dpd = df["dpd_months"].to_numpy(dtype=float)
        return np.clip(0.01 + 0.002 * age + 0.001 * dpd, 0.0, 0.99)


def _base_con():
    con = duckdb.connect(":memory:")
    data.apply_schema(con)
    data.load_synthetic(con)
    data.build_panel(con)
    data.build_lifetime_data(con)
    return con


def _con():
    con = _base_con()
    write_scores(con, AgeModel())
    return con


def test_staging_view_matches_the_python_stage_function():
    con = _con()
    data.run_sql(con, "04_staging.sql")
    df = con.execute(
        "SELECT loan_id, report_month, dpd_months, pd_12m, pd_12m_orig, stage FROM staging"
    ).df()
    assert (assign_stage(df) == df["stage"].to_numpy()).all()
    assert {1, 2, 3} <= set(df["stage"])


def test_load_portfolio_returns_loans_at_the_reporting_date_with_stages():
    con = _con()
    loans = load_portfolio(con)
    last = con.execute("SELECT MAX(report_month) FROM panel").fetchone()[0]
    assert len(loans) > 0
    assert set(loans["stage"]) <= {1, 2, 3}
    assert (loans["current_upb"] > 0).all()
    assert (loans["report_month"].dt.date == last).all()
    for col in ("loan_id", "term_months", "orig_rate", "credit_score", "dpd_months"):
        assert col in loans.columns


def test_load_portfolio_needs_scores():
    with pytest.raises(RuntimeError):
        load_portfolio(_base_con())


def test_run_ecl_writes_tables_and_outputs_on_synthetic_data():
    con = _con()
    with tempfile.TemporaryDirectory() as d:
        result = run_ecl(con, outdir=d, max_horizon=36)
        for name in TABLES:
            assert os.path.exists(os.path.join(d, "tables", name)), name
    n = result["n_loans"]
    assert n > 0
    assert con.execute("SELECT COUNT(*) FROM loan_ecl").fetchone()[0] == 3 * n
    assert abs(con.execute("SELECT SUM(weight) FROM scenario_weights").fetchone()[0] - 1.0) < 1e-9
    weighted = con.execute("SELECT SUM(ecl_weighted) FROM ecl_weighted").fetchone()[0]
    assert np.isclose(weighted, result["weighted_ecl"])
    ead_by_scenario = con.execute(
        "SELECT scenario, SUM(ead) FROM ecl_by_scenario GROUP BY scenario"
    ).fetchall()
    assert len({round(v, 6) for _, v in ead_by_scenario}) == 1
    assert result["checks"]["weights_sum_to_one"]
    assert result["checks"]["ecl_between_zero_and_lgd_times_ead"]
    assert any("constant LGD" in note for note in result["notes"])
