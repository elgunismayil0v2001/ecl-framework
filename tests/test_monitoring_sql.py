"""Monitoring pipeline tests on synthetic data. Skipped if DuckDB is not installed."""
import os
import tempfile

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from src import data  # noqa: E402
from src.ecl_pipeline import run_ecl  # noqa: E402
from src.monitoring_pipeline import run_monitoring  # noqa: E402
from src.pd_pipeline import write_scores  # noqa: E402

TABLES = (
    "monitoring_psi_by_month.csv",
    "monitoring_backtest_by_quarter.csv",
    "monitoring_feature_psi.csv",
    "monitoring_stage_migration.csv",
)


class AgeModel:
    """Stand-in PD model: PD rises with loan age."""

    def predict_proba(self, df):
        age = df["months_on_book"].to_numpy(dtype=float)
        return np.clip(0.01 + 0.002 * age + 0.001 * df["dpd_months"].to_numpy(dtype=float), 0.0, 0.99)


def _con():
    con = duckdb.connect(":memory:")
    data.apply_schema(con)
    data.load_synthetic(con)
    data.build_panel(con)
    data.build_lifetime_data(con)
    write_scores(con, AgeModel())
    return con


def test_monitoring_writes_tables_and_figures():
    con = _con()
    with tempfile.TemporaryDirectory() as d:
        result = run_monitoring(con, outdir=d, ref_periods=6, recent_periods=3)
        for name in TABLES:
            assert os.path.exists(os.path.join(d, "tables", name)), name
        assert os.path.exists(os.path.join(d, "figures", "monitoring_backtest.png"))
    assert len(result["backtest"]) > 0
    assert set(result["summary"]) == {"psi_red_months", "backtest_red_quarters", "feature_psi_red"}
    assert result["consistency"] is None
    assert any("loan_ecl" in note for note in result["notes"])


def test_stage_migration_probabilities_are_valid():
    con = _con()
    with tempfile.TemporaryDirectory() as d:
        result = run_monitoring(con, outdir=d, ref_periods=6, recent_periods=3)
    probs = result["migration_probs"]
    assert list(probs.index) == [1, 2, 3]
    assert np.allclose(probs.sum(axis=1).dropna(), 1.0)
    assert result["migration_counts"].to_numpy().sum() > 0


def test_pd_consistency_is_reported_once_ecl_has_been_run():
    con = _con()
    with tempfile.TemporaryDirectory() as d:
        run_ecl(con, outdir=d, max_horizon=36)
        result = run_monitoring(con, outdir=d, ref_periods=6, recent_periods=3)
        assert os.path.exists(os.path.join(d, "tables", "monitoring_pd_consistency.csv"))
    c = result["consistency"]
    assert c["n"] > 0
    assert c["scorecard_pd"] > 0 and c["hazard_pd"] > 0


def test_monitoring_needs_scores():
    con = duckdb.connect(":memory:")
    data.apply_schema(con)
    with pytest.raises(RuntimeError):
        run_monitoring(con)
