import os
import tempfile

from src import data
from src.freddie_schema import (
    LAYOUTS,
    ORIG_COLUMNS,
    ORIG_COLUMNS_V2,
    PERF_COLUMNS,
    PERF_COLUMNS_V2,
    count_fields,
    detect_layout,
    internal_names,
)
from src.load_freddie import check_files, expand, validate_inputs


def _expect(exc, fn):
    """Return the exception raised by fn(), or fail if it does not raise `exc`."""
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError(f"{exc.__name__} was not raised")


def test_column_counts_and_uniqueness():
    assert len(ORIG_COLUMNS) == 32
    assert len(PERF_COLUMNS) == 32
    assert len(set(ORIG_COLUMNS)) == 32
    assert len(set(PERF_COLUMNS)) == 32


def test_key_columns_present():
    for col in ("credit_score", "first_payment_date", "original_ltv", "original_dti",
                "original_upb", "loan_sequence_number", "original_loan_term"):
        assert col in ORIG_COLUMNS
    for col in ("loan_sequence_number", "monthly_reporting_period", "current_actual_upb",
                "current_loan_delinquency_status", "zero_balance_code"):
        assert col in PERF_COLUMNS


def test_count_fields_reads_first_non_empty_line():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.txt")
        with open(path, "w") as f:
            f.write("\n" + "|".join(["a"] * 5) + "\n" + "|".join(["b"] * 9) + "\n")
        assert count_fields(path) == 5
        empty = os.path.join(d, "empty.txt")
        open(empty, "w").close()
        assert count_fields(empty) == 0


def test_check_files_explains_what_is_missing():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "other_file.txt"), "w").close()
        e = _expect(FileNotFoundError, lambda: check_files(os.path.join(d, "sample_orig_*.txt"), 32))
    assert "No files match" in str(e)
    assert "other_file.txt" in str(e)  # shows what is in the folder


def test_check_files_rejects_wrong_column_count_and_accepts_right_one():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sample_orig_2019.txt")
        with open(path, "w") as f:
            f.write("|".join(["x"] * 32) + "\n")
        assert check_files(os.path.join(d, "sample_orig_*.txt"), 32) == [path]
        _expect(ValueError, lambda: check_files(os.path.join(d, "sample_orig_*.txt"), 31))


def test_failed_file_check_does_not_create_or_touch_the_database():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.duckdb")
        args = ["--db", db, "--orig", os.path.join(d, "none_*.txt"), "--perf", os.path.join(d, "none_*.txt")]
        _expect(FileNotFoundError, lambda: data.main(args))
        assert not os.path.exists(db)


def test_missing_fred_files_are_reported_before_the_database_is_touched():
    with tempfile.TemporaryDirectory() as d:
        orig = os.path.join(d, "sample_orig_2019.txt")
        perf = os.path.join(d, "sample_svcg_2019.txt")
        with open(orig, "w") as f:
            f.write("|".join(["x"] * 32) + "\n")
        with open(perf, "w") as f:
            f.write("|".join(["x"] * 32) + "\n")
        db = os.path.join(d, "test.duckdb")
        fred = os.path.join(d, "fred")
        argv = ["--db", db, "--orig", orig, "--perf", perf, "--fred-dir", fred]
        e = _expect(FileNotFoundError, lambda: data.main(argv))
        assert "UNRATE.csv" in str(e)
        assert not os.path.exists(db)


USED_BY_LOADER = {
    "orig": ["credit_score", "first_payment_date", "original_ltv", "original_dti", "original_upb",
             "original_interest_rate", "original_loan_term", "property_state", "loan_sequence_number"],
    "perf": ["loan_sequence_number", "monthly_reporting_period", "current_actual_upb",
             "current_loan_delinquency_status", "zero_balance_code", "zero_balance_removal_upb",
             "actual_loss_calculation"],
}


def test_both_layouts_have_the_documented_field_counts_and_unique_names():
    assert (len(LAYOUTS["v1"]["orig"]), len(LAYOUTS["v1"]["perf"])) == (32, 32)
    assert (len(LAYOUTS["v2"]["orig"]), len(LAYOUTS["v2"]["perf"])) == (31, 35)
    for layout in LAYOUTS.values():
        assert len(set(layout["orig"])) == len(layout["orig"])
        assert len(set(layout["perf"])) == len(layout["perf"])
    assert ORIG_COLUMNS_V2[19] == "loan_identifier" and PERF_COLUMNS_V2[21] == "actual_loss"


def test_every_field_used_by_the_loader_exists_in_both_layouts_after_renaming():
    for name, layout in LAYOUTS.items():
        for kind in ("orig", "perf"):
            names = set(internal_names(layout[kind]))
            for col in USED_BY_LOADER[kind]:
                assert col in names, (name, kind, col)


def test_loss_sign_convention_differs_between_layouts():
    assert LAYOUTS["v2"]["loss_sign"] == 1.0
    assert LAYOUTS["v1"]["loss_sign"] == -1.0


def test_detect_layout():
    assert detect_layout(32, 32) == "v1"
    assert detect_layout(31, 35) == "v2"
    e = _expect(ValueError, lambda: detect_layout(33, 36))
    assert "33 origination" in str(e) and "trailing pipe" in str(e)


def _write_fields(path, n_fields):
    with open(path, "w") as f:
        f.write("|".join(["x"] * n_fields) + "\n")


def test_expand_accepts_comma_separated_patterns_without_duplicates():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "sample_orig_2019.txt")
        b = os.path.join(d, "orig_2020Q1.txt")
        _write_fields(a, 3)
        _write_fields(b, 3)
        files = expand(f"{d}/sample_orig_*.txt, {d}/orig_*.txt, {d}/sample_orig_*.txt")
        assert files == sorted([a, b])


def test_validate_inputs_detects_each_layout_and_rejects_mixing():
    with tempfile.TemporaryDirectory() as d:
        _write_fields(os.path.join(d, "orig_2020Q1.txt"), 31)
        _write_fields(os.path.join(d, "perf_2020Q1.txt"), 35)
        _, _, layout = validate_inputs(f"{d}/orig_*.txt", f"{d}/perf_*.txt")
        assert layout == "v2"
        _write_fields(os.path.join(d, "orig_2020Q2.txt"), 32)  # an old-layout file in the same load
        e = _expect(ValueError, lambda: validate_inputs(f"{d}/orig_*.txt", f"{d}/perf_*.txt"))
        assert "Do not mix" in str(e)
    with tempfile.TemporaryDirectory() as d:
        _write_fields(os.path.join(d, "sample_orig_2019.txt"), 32)
        _write_fields(os.path.join(d, "sample_svcg_2019.txt"), 32)
        assert validate_inputs(f"{d}/sample_orig_*.txt", f"{d}/sample_svcg_*.txt")[2] == "v1"
