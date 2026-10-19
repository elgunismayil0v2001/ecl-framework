from src.synthetic import make_loan_panel, make_pd_dataset


def test_pd_dataset_shape_and_default_rate():
    df = make_pd_dataset(n=5000, seed=1)
    assert len(df) == 5000
    assert 0.01 < df["default"].mean() < 0.10
    assert df["p_true"].between(0, 1).all()


def test_loan_panel_is_reproducible_and_complete():
    loans, perf = make_loan_panel(n_loans=50, months=24, seed=7)
    loans2, perf2 = make_loan_panel(n_loans=50, months=24, seed=7)
    assert loans.equals(loans2) and perf.equals(perf2)
    assert len(perf) == 50 * 24
    assert loans["loan_id"].is_unique


def test_default_state_is_absorbing():
    _, perf = make_loan_panel(n_loans=200, months=36, seed=3)
    for _, g in perf.sort_values("report_month").groupby("loan_id"):
        d = g["dpd_months"].to_numpy()
        hit = (d >= 3).argmax() if (d >= 3).any() else None
        if hit is not None:
            assert (d[hit:] >= 3).all()
