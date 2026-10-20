"""Week 3 SQL and pipeline tests on synthetic data. Skipped if DuckDB is not installed."""
import os
import tempfile

import pytest

duckdb = pytest.importorskip("duckdb")

from src import data  # noqa: E402
from src.freddie_schema import ORIG_COLUMNS, PERF_COLUMNS  # noqa: E402
from src.lifetime_pipeline import run_lifetime  # noqa: E402


def _synthetic_con():
    con = duckdb.connect(":memory:")
    data.apply_schema(con)
    data.load_synthetic(con)
    data.build_panel(con)
    data.build_lifetime_data(con)
    return con


def _row(columns, **values):
    row = dict.fromkeys(columns, "")
    row.update(values)
    return "|".join(row[c] for c in columns)


def _orig(loan):
    return _row(
        ORIG_COLUMNS,
        credit_score="720",
        first_payment_date="201803",
        original_dti="35",
        original_upb="200000",
        original_ltv="80",
        original_interest_rate="4.5",
        property_state="CA",
        loan_sequence_number=loan,
        original_loan_term="360",
    )


def _perf(loan, period, status="0", zero="", removal_upb="", loss=""):
    return _row(
        PERF_COLUMNS,
        loan_sequence_number=loan,
        monthly_reporting_period=period,
        current_actual_upb="190000",
        current_loan_delinquency_status=status,
        zero_balance_code=zero,
        zero_balance_removal_upb=removal_upb,
        actual_loss_calculation=loss,
    )


def test_lgd_data_uses_resolved_defaults_and_clips_to_unit_interval():
    from src.load_freddie import load_freddie

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "sample_orig_2018.txt"), "w") as f:
            f.write("\n".join(_orig(x) for x in ("A", "B", "C", "D", "E")) + "\n")
        rows = [
            _perf("A", "201901", "3", "09", "100000", "20000"),  # LGD 0.2
            _perf("B", "201901", "3", "03", "100000", "150000"),  # loss above UPB: clipped to 1
            _perf("C", "201901", "3", "02", "100000", "-5000"),  # gain: clipped to 0
            _perf("D", "201901", "0", "01", "100000", "0"),  # prepaid, not a credit event
            _perf("E", "201901", "3", "09", "100000", ""),  # no loss reported
        ]
        with open(os.path.join(d, "sample_svcg_2018.txt"), "w") as f:
            f.write("\n".join(rows) + "\n")
        con = duckdb.connect(":memory:")
        data.apply_schema(con)
        load_freddie(con, os.path.join(d, "sample_orig_*.txt"), os.path.join(d, "sample_svcg_*.txt"))
        data.build_lifetime_data(con)
    lgd = dict(con.execute("SELECT loan_id, lgd FROM lgd_data").fetchall())
    assert set(lgd) == {"A", "B", "C"}
    assert abs(lgd["A"] - 0.2) < 1e-9
    assert lgd["B"] == 1.0
    assert lgd["C"] == 0.0


def test_hazard_obs_target_is_consistent_with_next_month_status():
    con = _synthetic_con()
    values = {r[0] for r in con.execute("SELECT DISTINCT default_next FROM hazard_obs").fetchall()}
    assert values <= {0, 1}
    bad = con.execute(
        """SELECT COUNT(*) FROM hazard_obs h
           JOIN panel n ON n.loan_id = h.loan_id
                       AND n.report_month = CAST(h.report_month + INTERVAL 1 MONTH AS DATE)
           WHERE (h.default_next = 1) <> (n.in_default = 1)"""
    ).fetchone()[0]
    assert bad == 0
    n_obs = con.execute("SELECT COUNT(*) FROM hazard_obs").fetchone()[0]
    n_performing = con.execute("SELECT COUNT(*) FROM panel WHERE in_default = 0").fetchone()[0]
    assert 0 < n_obs <= n_performing
    assert con.execute("SELECT SUM(default_next) FROM hazard_obs").fetchone()[0] > 0


def test_each_loan_defaults_at_most_once_in_hazard_data():
    con = _synthetic_con()
    worst = con.execute(
        "SELECT MAX(c) FROM (SELECT SUM(default_next) AS c FROM hazard_obs GROUP BY loan_id)"
    ).fetchone()[0]
    assert worst == 1


def test_run_lifetime_pipeline_writes_outputs_on_synthetic_data():
    con = _synthetic_con()
    with tempfile.TemporaryDirectory() as d:
        result = run_lifetime(con, outdir=d, horizon=24)
        assert len(result["curves"]) == 3 * 24
        assert result["n_loans"] > 0
        for name in ("lifetime_pd_curves.csv", "hazard_coefficients.csv", "ead_portfolio_path.csv"):
            assert os.path.exists(os.path.join(d, "tables", name))
        assert os.path.exists(os.path.join(d, "figures", "lifetime_pd_curves.png"))
    assert any("macro" in n.lower() for n in result["notes"])
    assert result["lgd_model"] is None
    assert any("LGD model skipped" in n for n in result["notes"])
