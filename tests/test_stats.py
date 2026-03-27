from __future__ import annotations

from tropical_wave_tools.stats import linear_regression, linear_trend, pearson_correlation


def test_linear_trend_returns_slope_and_intercept(synthetic_wave_data) -> None:
    result = linear_trend(synthetic_wave_data)
    assert "slope" in result
    assert "intercept" in result


def test_correlation_and_regression_are_defined(synthetic_wave_data) -> None:
    predictor = synthetic_wave_data.mean("lon")
    predictand = predictor * 2.0 + 1.0
    correlation = pearson_correlation(predictor, predictand)
    regression = linear_regression(predictor, predictand)
    assert float(correlation.mean()) > 0.99
    assert "slope" in regression
    assert "correlation" in regression
