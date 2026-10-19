from src.eda import band_case_sql, band_labels


def test_band_labels():
    assert band_labels([620, 660, 700]) == ["<620", "620-659", "660-699", "700+"]


def test_band_case_sql_has_one_branch_per_edge_and_handles_nulls():
    sql = band_case_sql("credit_score", [620, 660])
    assert "WHEN credit_score < 620 THEN 0" in sql
    assert "WHEN credit_score < 660 THEN 1" in sql
    assert "ELSE 2" in sql
    assert "credit_score IS NULL THEN NULL" in sql
