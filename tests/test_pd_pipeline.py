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
