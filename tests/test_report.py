import os
import tempfile

import numpy as np
import pandas as pd

from src.report import (
    PLACEHOLDER,
    make_charts,
    markdown_table,
    render_results_table,
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
