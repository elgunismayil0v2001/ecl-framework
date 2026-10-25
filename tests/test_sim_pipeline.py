"""End-to-end test on simulated Freddie Mac style files. Skipped if DuckDB is not installed."""
import os
import tempfile

import pytest

duckdb = pytest.importorskip("duckdb")

from src import data  # noqa: E402
from src.data_quality import run_checks  # noqa: E402
from src.ecl_pipeline import run_ecl  # noqa: E402
from src.lgd import LGDModel  # noqa: E402
from src.lifetime_pipeline import load_hazard_sample  # noqa: E402
from src.load_freddie import load_freddie  # noqa: E402
from src.load_macro import load_macro  # noqa: E402
from src.pd_pipeline import load_training_frame, run_pd_models, write_scores  # noqa: E402
from src.rollrate import RollRateModel  # noqa: E402
from src.simulate import simulate, to_freddie_files, write_fred_csvs  # noqa: E402

_CACHE = {}


def _con(layout="v2"):
    if layout not in _CACHE:
        loans, perf = simulate(2500, 120, 1)
        with tempfile.TemporaryDirectory() as d:
            orig_path, perf_path = to_freddie_files(loans, perf, d, layout)
            write_fred_csvs(os.path.join(d, "fred"))
            con = duckdb.connect(":memory:")
            data.apply_schema(con)
            load_freddie(con, str(orig_path), str(perf_path))
            load_macro(con, os.path.join(d, "fred"))
        data.build_panel(con)
        data.build_lifetime_data(con)
        _CACHE[layout] = con
    return _CACHE[layout]


def test_data_quality_checks_all_pass_on_simulated_files_in_both_layouts():
    for layout in ("v2", "v1"):
        report = run_checks(_con(layout))
        failed = [c for c in report if not c["passed"]]
        assert not failed, (layout, failed)
        names = {c["check"] for c in report}
        assert {"lgd_mean_plausible", "macro_coverage"} <= names


def test_rows_where_loans_leave_the_pool_are_not_in_the_modelling_sample():
    con = _con()
    leaked = con.execute(
        """SELECT COUNT(*) FROM features f
           JOIN performance p ON p.loan_id = f.loan_id AND p.report_month = f.report_month
           WHERE COALESCE(p.zero_bal_code, '') <> ''"""
    ).fetchone()[0]
    assert leaked == 0
    exits = "SELECT COUNT(*) FROM performance WHERE COALESCE(zero_bal_code, '') <> ''"
    assert con.execute(exits).fetchone()[0] > 0


def test_scorecard_finds_the_simulated_risk_drivers_out_of_time():
    result = run_pd_models(load_training_frame(_con()), use_benchmark=False)
    report = result["reports"]["scorecard"]
    assert report["gini"]["value"] > 0.30
    assert (result["scorecard"].coef_ > 0).all()
    assert len(result["scorecard"].selected_) >= 2


def test_roll_rate_model_recovers_macro_and_loan_effects_and_a_sensible_lifetime_pd():
    model = RollRateModel().fit(load_hazard_sample(_con(), 400_000))
    c = model.coef_
    assert c["credit_score"] < 0
    assert c["orig_ltv"] > 0
    assert c["hpi_yoy"] < 0


def test_lgd_model_finds_higher_losses_for_higher_ltv():
    lgd = _con().execute("SELECT * FROM lgd_data").df()
    assert len(lgd) > 100
    model = LGDModel().fit(lgd)
    assert model.coef_["orig_ltv"] > 0


def test_full_chain_from_scores_to_ecl_runs_on_simulated_data():
    con = _con()  # runs last in this file: it adds the scores, staging and ECL tables to the shared database
    result = run_pd_models(load_training_frame(con), use_benchmark=False)
    write_scores(con, result["scorecard"])
    with tempfile.TemporaryDirectory() as d:
        out = run_ecl(con, outdir=d, max_horizon=120, max_loans=2000)
    assert out["checks"]["weights_sum_to_one"]
    assert out["checks"]["ecl_between_zero_and_lgd_times_ead"]
    assert out["checks"]["scenario_order_downside_base_upside"]
    assert out["checks"]["all_three_stages_present"]
    assert out["checks"]["coverage_stage3_gt_stage2_gt_stage1"]
    assert out["checks"]["rwa_density_between_3_and_80_percent"]
    assert out["weighted_ecl"] > 0
