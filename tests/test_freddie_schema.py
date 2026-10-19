import os
import tempfile

from src.freddie_schema import ORIG_COLUMNS, PERF_COLUMNS, count_fields


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
