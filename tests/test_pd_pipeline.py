import numpy as np
import pandas as pd

from src.pd_pipeline import format_gate_report, run_pd_models, usable_features
from src.synthetic import make_pd_dataset

FEATURES = ["credit_score", "ltv", "dti"]


def test_usable_features_drops_missing_and_all_nan_columns():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    assert usable_features(df, ["a", "b", "c"]) == ["a"]


def test_run_pd_models_returns_split_reports_and_scorecard():
    df = make_pd_dataset(n=40000, seed=0)
    result = run_pd_models(
        df,
        features=FEATURES,
        time_col="vintage",
        target_col="default",
        oot_periods=2,
        gap_periods=0,
        use_benchmark=False,
    )
    assert result["split"]["n_train"] + result["split"]["n_oot"] == len(df[df["vintage"] != 99])
    assert result["features"] == FEATURES
    assert result["benchmark"] is None
    assert result["reports"]["scorecard"]["all_passed"]


def test_gate_report_formatting_lists_each_metric():
    df = make_pd_dataset(n=40000, seed=0)
    result = run_pd_models(
        df, FEATURES, "vintage", "default", oot_periods=2, gap_periods=0, use_benchmark=False
    )
    text = format_gate_report("scorecard", result["reports"]["scorecard"])
    assert "gini" in text and "psi_train_vs_test" in text and "PASS" in text


def test_default_scorecard_features_exclude_macro_variables():
    rng = np.random.default_rng(0)
    n = 6000
    score = rng.normal(700, 50, n)
    df = pd.DataFrame(
        {
            "t": np.repeat(np.arange(12), n // 12),
            "credit_score": score,
            "orig_ltv": rng.uniform(40, 97, n),
            "orig_dti": rng.uniform(10, 50, n),
            "unemployment": rng.normal(5, 1, n),
            "hpi_yoy": rng.normal(0.03, 0.02, n),
            "gdp_yoy": rng.normal(0.02, 0.01, n),
        }
    )
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-(-3.5 - 0.012 * (score - 700)))))
    result = run_pd_models(
        df, time_col="t", target_col="y", oot_periods=2, gap_periods=0, use_benchmark=False
    )
    assert not {"unemployment", "hpi_yoy", "gdp_yoy"} & set(result["features"])
    assert "credit_score" in result["features"]


def test_run_pd_models_treats_seasoning_features_as_non_monotone_by_default():
    rng = np.random.default_rng(3)
    n = 40_000
    age = rng.uniform(0, 100, n)
    p = 1 / (1 + np.exp(-(-4.0 + 0.06 * np.minimum(age, 45) - 0.05 * np.maximum(age - 45, 0))))
    df = pd.DataFrame(
        {
            "t": np.repeat(np.arange(10), n // 10),
            "months_on_book": age,
            "credit_score": rng.normal(700, 50, n),
        }
    )
    df["y"] = rng.binomial(1, p)
    result = run_pd_models(
        df, time_col="t", target_col="y", oot_periods=2, gap_periods=0, use_benchmark=False
    )
    bins = result["scorecard"].bin_tables()
    rates = bins[bins["feature"] == "months_on_book"]["bad_rate"].to_numpy()
    assert rates.argmax() not in (0, len(rates) - 1)
    assert result["split"]["train_first_period"] == "0"
