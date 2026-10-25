"""Scoring and staging integration on synthetic data. Skipped if DuckDB is not installed."""
from pathlib import Path

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from src import data  # noqa: E402
from src.pd_pipeline import load_training_frame, write_scores  # noqa: E402

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


class ConstantModel:
    """Stand-in PD model: a base PD plus a small amount per month in arrears."""

    def predict_proba(self, df):
        return np.full(len(df), 0.02) + 0.001 * df["dpd_months"].to_numpy()


def _con():
    con = duckdb.connect(":memory:")
    data.apply_schema(con)
    data.load_synthetic(con)
    data.build_panel(con)
    return con


def test_training_frame_has_target_and_features():
    df = load_training_frame(_con())
    for col in ("loan_id", "report_month", "default_12m", "credit_score", "orig_ltv", "max_dpd_last_12m"):
        assert col in df.columns


def test_scoring_features_cover_all_performing_months_including_the_censored_tail():
    con = _con()
    n_performing = con.execute("SELECT COUNT(*) FROM panel WHERE in_default = 0").fetchone()[0]
    n_scoring = con.execute("SELECT COUNT(*) FROM features_scoring").fetchone()[0]
    n_training = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    assert n_scoring == n_performing
    assert n_scoring > n_training


def test_training_and_scoring_features_agree_on_shared_rows():
    con = _con()
    mismatch = con.execute(
        """SELECT COUNT(*) FROM features f JOIN features_scoring s USING (loan_id, report_month)
           WHERE f.max_dpd_last_12m IS DISTINCT FROM s.max_dpd_last_12m
              OR f.balance_ratio IS DISTINCT FROM s.balance_ratio"""
    ).fetchone()[0]
    assert mismatch == 0


def test_write_scores_gives_the_pd_now_and_the_pd_with_a_clean_payment_history():
    con = _con()
    n = write_scores(con, ConstantModel())
    assert n == con.execute("SELECT COUNT(*) FROM features_scoring").fetchone()[0]
    # ConstantModel: PD = 0.02 + 0.001 * months in arrears, so the clean-history PD is always 0.02
    wrong = "SELECT COUNT(*) FROM scores WHERE ABS(pd_12m_orig - 0.02) > 1e-12"
    assert con.execute(wrong).fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM scores WHERE pd_12m < pd_12m_orig - 1e-12").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM scores WHERE pd_12m > pd_12m_orig + 1e-12").fetchone()[0] > 0


def test_staging_works_on_written_scores_and_rewriting_scores_is_safe():
    con = _con()
    write_scores(con, ConstantModel())
    con.execute((SQL_DIR / "04_staging.sql").read_text())
    missing_pd = con.execute("SELECT COUNT(*) FROM staging WHERE stage <> 3 AND pd_12m IS NULL").fetchone()[0]
    assert missing_pd == 0
    n_stage3 = con.execute("SELECT COUNT(*) FROM staging WHERE stage = 3").fetchone()[0]
    n_default = con.execute("SELECT COUNT(*) FROM panel WHERE in_default = 1").fetchone()[0]
    assert n_stage3 == n_default
    write_scores(con, ConstantModel())  # staging view exists; must not raise
