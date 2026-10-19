import numpy as np
from sklearn.linear_model import LogisticRegression

from src.gates import assert_gates, evaluate_pd_model
from src.synthetic import make_pd_dataset

FEATURES = ["credit_score", "ltv", "dti"]


def _fit_and_predict():
    df = make_pd_dataset(n=20000, seed=0)
    train, test = df[df["vintage"] < 6], df[df["vintage"] >= 6]  # time-based split
    mu, sd = train[FEATURES].mean(), train[FEATURES].std()
    model = LogisticRegression(max_iter=1000)
    model.fit((train[FEATURES] - mu) / sd, train["default"])
    p_train = model.predict_proba((train[FEATURES] - mu) / sd)[:, 1]
    p_test = model.predict_proba((test[FEATURES] - mu) / sd)[:, 1]
    return train["default"].to_numpy(), test["default"].to_numpy(), p_train, p_test


def test_reference_logistic_model_passes_all_gates():
    _, y_test, p_train, p_test = _fit_and_predict()
    report = evaluate_pd_model(y_test, p_test, p_train)
    assert_gates(report)
    assert report["all_passed"]


def test_random_scores_fail_discrimination_gate():
    _, y_test, _, p_test = _fit_and_predict()
    rng = np.random.default_rng(5)
    report = evaluate_pd_model(y_test, rng.permutation(p_test))
    assert not report["gini"]["passed"]
    assert not report["all_passed"]


def test_miscalibrated_model_fails_calibration_gate_but_keeps_ranking():
    _, y_test, _, p_test = _fit_and_predict()
    report = evaluate_pd_model(y_test, np.clip(p_test * 3.0, 0, 1))
    assert report["gini"]["passed"]
    assert not report["calibration_rel_error"]["passed"]


def test_assert_gates_raises_with_failed_metric_names():
    _, y_test, _, p_test = _fit_and_predict()
    report = evaluate_pd_model(y_test, np.clip(p_test * 3.0, 0, 1))
    try:
        assert_gates(report)
    except AssertionError as e:
        assert "calibration_rel_error" in str(e)
    else:
        raise AssertionError("assert_gates should have raised")


def test_predictions_are_valid_probabilities():
    _, _, p_train, p_test = _fit_and_predict()
    assert np.all((p_train >= 0) & (p_train <= 1))
    assert np.all((p_test >= 0) & (p_test <= 1))
