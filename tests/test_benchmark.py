import numpy as np
import pytest

pytest.importorskip("lightgbm")

from src.benchmark import BenchmarkModel  # noqa: E402
from src.gates import assert_gates, evaluate_pd_model  # noqa: E402
from src.splits import time_based_split  # noqa: E402
from src.synthetic import make_pd_dataset  # noqa: E402

FEATURES = ["credit_score", "ltv", "dti"]
MONOTONE = {"credit_score": -1, "ltv": 1, "dti": 1}


def _fit():
    df = make_pd_dataset(n=40000, seed=0)
    train, oot = time_based_split(df, "vintage", oot_periods=2, gap_periods=0)
    model = BenchmarkModel(FEATURES, monotone=MONOTONE, n_estimators=150).fit(train, "default")
    return model, train, oot


def test_benchmark_passes_validation_gates():
    model, train, oot = _fit()
    report = evaluate_pd_model(
        oot["default"].to_numpy(), model.predict_proba(oot), model.predict_proba(train)
    )
    assert_gates(report)


def test_monotonic_constraints_are_respected():
    model, _, oot = _fit()
    base = oot.iloc[:200].copy()
    grid = np.linspace(520, 820, 25)
    curves = np.array([model.predict_proba(base.assign(credit_score=s)) for s in grid])
    assert np.all(np.diff(curves, axis=0) <= 1e-9)  # PD never rises with a higher credit score
    ltv_grid = np.linspace(40, 97, 25)
    ltv_curves = np.array([model.predict_proba(base.assign(ltv=v)) for v in ltv_grid])
    assert np.all(np.diff(ltv_curves, axis=0) >= -1e-9)


def test_importance_table_covers_all_features_and_ranks_credit_score_first():
    model, _, oot = _fit()
    table = model.importance(oot)
    assert set(table["feature"]) == set(FEATURES)
    assert (table["mean_abs_shap"] >= 0).all()
    assert table.iloc[0]["feature"] == "credit_score"
