import numpy as np
from scipy.stats import spearmanr

from src.lgd import LGDModel, lgd_by_band
from src.synthetic import make_lgd_dataset


def _fit():
    df = make_lgd_dataset(n=8000, seed=0)
    train, test = df.iloc[:6000], df.iloc[6000:]
    return LGDModel().fit(train), train, test


def test_predictions_are_between_zero_and_one():
    model, _, test = _fit()
    p = model.predict(test)
    assert np.all((p > 0) & (p < 1))


def test_coefficient_signs_follow_the_risk_direction():
    model, _, _ = _fit()
    assert model.coef_["orig_ltv"] > 0  # higher LTV, higher loss
    assert model.coef_["hpi_yoy"] < 0  # falling house prices, higher loss
    assert model.coef_["credit_score"] < 0


def test_average_prediction_matches_average_realised_lgd():
    model, train, _ = _fit()
    assert abs(model.predict(train).mean() - train["lgd"].mean()) < 0.01


def test_model_ranks_loans_and_beats_a_constant_on_new_data():
    model, train, test = _fit()
    p = model.predict(test)
    assert spearmanr(p, test["true_lgd"])[0] > 0.9
    mse_model = np.mean((p - test["lgd"]) ** 2)
    mse_constant = np.mean((train["lgd"].mean() - test["lgd"]) ** 2)
    assert mse_model < mse_constant


def test_floor_is_applied():
    model, _, test = _fit()
    assert model.predict(test, floor=0.30).min() >= 0.30


def test_missing_values_are_filled_and_constant_features_dropped():
    df = make_lgd_dataset(n=3000, seed=1)
    df["unemployment"] = np.nan
    df["orig_rate"] = 0.05
    model = LGDModel().fit(df)
    assert "unemployment" not in model.features_ and "orig_rate" not in model.features_
    test = df.copy()
    test.loc[test.index[::5], "orig_ltv"] = np.nan
    assert not np.isnan(model.predict(test)).any()


def test_band_summary_shows_higher_lgd_for_higher_ltv():
    model, _, test = _fit()
    table = lgd_by_band(model, test, "orig_ltv", [60, 70, 80, 90])
    assert table["n"].sum() == len(test)
    assert table["realised"].iloc[-1] > table["realised"].iloc[0]
    assert table["predicted"].iloc[-1] > table["predicted"].iloc[0]
