import os
import tempfile

import numpy as np
import pandas as pd

from src.report import (
    PLACEHOLDER,
    make_charts,
    markdown_table,
    render_data_source,
    render_results_table,
    render_split,
    replace_block,
    update_file,
)


def _write_tables(d):
    pd.DataFrame(
        [
            {"model": "scorecard", "metric": "gini", "value": 0.512, "rule": ">= 0.3", "passed": True},
            {"model": "benchmark", "metric": "gini", "value": 0.534, "rule": ">= 0.3", "passed": True},
        ]
    ).to_csv(os.path.join(d, "pd_validation.csv"), index=False)
    rows = []
    for scenario, mult in (("base", 1.0), ("downside", 1.5), ("upside", 0.8), ("weighted", 1.1)):
        for stage, ead, cov in ((1, 800.0, 0.002), (2, 150.0, 0.03), (3, 50.0, 0.30)):
            rows.append({"scenario": scenario, "stage": stage, "ead": ead, "ecl": ead * cov * mult,
                         "coverage": cov * mult})
    pd.DataFrame(rows).to_csv(os.path.join(d, "ecl_coverage_by_stage.csv"), index=False)
    capital = {"scenario": "base", "ead_total": 1000.0, "rwa": 120.0, "rwa_density": 0.12,
               "el_irb": 3.0, "ecl_total": 6.0}
    pd.DataFrame([capital]).to_csv(os.path.join(d, "capital_by_scenario.csv"), index=False)
    pd.DataFrame([{"capital_ratio_start": 0.12, "capital_ratio_after": 0.10, "change_pp": -2.0}]).to_csv(
        os.path.join(d, "stress_test.csv"), index=False)
    pd.DataFrame({"capital_ratio_start": [0.08, 0.12], "capital_ratio_after": [0.05, 0.10],
                  "change_pp": [-3.0, -2.0]}).to_csv(os.path.join(d, "capital_sensitivity.csv"), index=False)


def test_markdown_table_formats_numbers_and_missing_values():
    t = markdown_table(pd.DataFrame({"a": [1.23456, np.nan], "b": ["x", "y"]}), fmt="{:.2f}")
    lines = t.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1.23 | x |"
    assert lines[3] == "|  | y |"


def test_replace_block_replaces_only_the_marked_block_and_reports_missing_markers():
    text = "before\n<!-- X:START -->\nold\n<!-- X:END -->\nafter"
    new, found = replace_block(text, "X", "new")
    assert found and new == "before\n<!-- X:START -->\nnew\n<!-- X:END -->\nafter"
    same, found = replace_block(text, "Y", "new")
    assert not found and same == text


def test_results_table_uses_placeholders_when_nothing_has_been_generated():
    with tempfile.TemporaryDirectory() as d:
        t = render_results_table(d)
    assert t.count("TBD") == 7


def test_results_table_fills_values_from_tables():
    with tempfile.TemporaryDirectory() as d:
        _write_tables(d)
        t = render_results_table(d)
    assert "0.512" in t and "0.534" in t
    assert "80.00% / 15.00% / 5.00%" in t
    assert "12.00%" in t  # RWA density
    assert "12.0% to 10.0% (-2.00 pp)" in t


def test_update_file_fills_blocks_and_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        _write_tables(d)
        path = os.path.join(d, "doc.md")
        with open(path, "w") as f:
            f.write(f"# T\n<!-- RESULTS:START -->\n{PLACEHOLDER}\n<!-- RESULTS:END -->\n"
                    f"<!-- ECL_COVERAGE:START -->\n{PLACEHOLDER}\n<!-- ECL_COVERAGE:END -->\n"
                    f"<!-- LIFETIME:START -->\n{PLACEHOLDER}\n<!-- LIFETIME:END -->\n")
        assert update_file(path, d) == ["RESULTS", "LIFETIME", "ECL_COVERAGE"]
        first = open(path).read()
        update_file(path, d)
        second = open(path).read()
    assert first == second
    assert "0.512" in first
    assert "Stage 3" in first  # coverage table
    assert PLACEHOLDER in first  # lifetime curves were not generated, so its block keeps the placeholder


def test_make_charts_creates_files_only_for_available_tables():
    with tempfile.TemporaryDirectory() as d:
        figs = os.path.join(d, "figs")
        assert make_charts(d, figs) == []
        _write_tables(d)
        paths = make_charts(d, figs)
        assert len(paths) == 2
        assert all(os.path.exists(p) for p in paths)


def test_data_source_block_labels_simulated_results():
    with tempfile.TemporaryDirectory() as d:
        assert render_data_source(d) == PLACEHOLDER
        pd.DataFrame(
            [{"source": "simulated data (make sim)", "loans": 15000, "loan_months": 603209,
              "first_month": "2017-02-01", "last_month": "2024-12-01"}]
        ).to_csv(os.path.join(d, "data_source.csv"), index=False)
        text = render_data_source(d)
        assert "simulated data (make sim)" in text and "15,000 loans" in text
        assert "2017-02 to 2024-12" in text
        assert "simulation, not the mortgage market" in text
        real = {"source": "Freddie Mac Single-Family Loan-Level Dataset", "loans": 50000,
                "loan_months": 3000000, "first_month": "2017-02-01", "last_month": "2025-09-01"}
        pd.DataFrame([real]).to_csv(os.path.join(d, "data_source.csv"), index=False)
        assert "simulation" not in render_data_source(d)


def test_split_block_shows_periods_counts_and_rates():
    with tempfile.TemporaryDirectory() as d:
        assert render_split(d) == PLACEHOLDER
        pd.DataFrame(
            [{"n_train": 450081, "n_oot": 24836, "train_default_rate": 0.0380, "oot_default_rate": 0.0353,
              "train_first_period": "2017-02-01 00:00:00", "train_last_period": "2022-06-01 00:00:00",
              "oot_first_period": "2023-07-01 00:00:00", "oot_last_period": "2023-12-01 00:00:00",
              "gap_periods": 12, "features": "credit_score, orig_ltv"}]
        ).to_csv(os.path.join(d, "pd_split.csv"), index=False)
        text = render_split(d)
    assert "2017-02 to 2022-06" in text and "2023-07 to 2023-12" in text
    assert "450,081 / 24,836" in text and "3.80% / 3.53%" in text
    assert "credit_score, orig_ltv" in text


def _more_tables(d):
    pd.DataFrame(
        {
            "feature": ["dpd_months", "orig_dti"],
            "iv": [0.681, 0.042],
            "selected": [True, False],
            "coefficient": [0.82, np.nan],
            "note": [np.nan, "IV 0.015 below 0.02"],
        }
    ).to_csv(os.path.join(d, "pd_iv_table.csv"), index=False)
    pd.DataFrame(
        {"band": ["<60", "60-69"], "n": [120, 340], "realised": [0.12, 0.18], "predicted": [0.13, 0.17]}
    ).to_csv(os.path.join(d, "lgd_by_ltv_band.csv"), index=False)
    pd.DataFrame(
        {
            "rel_threshold": [2.0, 2.0, 3.0, 3.0],
            "abs_threshold": [0.005, 0.01, 0.005, 0.01],
            "stage2_share_count": [0.03, 0.02, 0.02, 0.01],
            "stage2_share_ead": [0.031, 0.021, 0.019, 0.012],
        }
    ).to_csv(os.path.join(d, "sicr_sensitivity.csv"), index=False)
    pd.DataFrame(
        {"quarter": ["2020Q1", "2020Q2", "2020Q3"], "flag": ["amber", "green", "green"]}
    ).to_csv(os.path.join(d, "monitoring_backtest_by_quarter.csv"), index=False)
    pd.DataFrame({"feature": ["credit_score"], "psi": [0.04], "flag": ["green"]}).to_csv(
        os.path.join(d, "monitoring_feature_psi.csv"), index=False
    )
    migration = {"stage": [1, 2, 3], "1": [0.99, 0.4, 0.0], "2": [0.01, 0.4, 0.0], "3": [0.0, 0.2, 1.0]}
    pd.DataFrame(migration).to_csv(os.path.join(d, "monitoring_stage_migration.csv"), index=False)


def test_additional_result_blocks_render_from_tables():
    from src.report import (
        render_iv_table,
        render_lgd_bands,
        render_monitoring,
        render_sicr_sensitivity,
        render_stage_shares,
    )

    with tempfile.TemporaryDirectory() as d:
        for render in (render_iv_table, render_lgd_bands, render_stage_shares, render_sicr_sensitivity,
                       render_monitoring):
            assert render(d) == PLACEHOLDER
        _write_tables(d)
        _more_tables(d)
        iv = render_iv_table(d)
        assert "dpd_months" in iv and "yes" in iv and "IV 0.015 below 0.02" in iv
        lgd = render_lgd_bands(d)
        assert "60-69" in lgd and "| 340 |" in lgd
        shares = render_stage_shares(d)
        assert "80.00%" in shares and "5.00%" in shares
        sicr = render_sicr_sensitivity(d)
        assert "2.0x" in sicr and "+0.50 pp" in sicr and "3.10" in sicr
        mon = render_monitoring(d)
        assert "2 green, 1 amber, 0 red out of 3 quarters" in mon
        assert "credit_score" in mon and "from stage" in mon
