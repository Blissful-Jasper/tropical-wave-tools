from __future__ import annotations

import numpy as np
import xarray as xr

from tropical_wave_tools.stats import linear_regression, linear_trend, one_sample_ttest, pearson_correlation


def test_linear_trend_returns_slope_and_intercept(synthetic_wave_data) -> None:
    result = linear_trend(synthetic_wave_data)
    assert "slope" in result
    assert "intercept" in result
    assert "pvalue" in result
    assert "stderr" in result


def test_correlation_and_regression_are_defined(synthetic_wave_data) -> None:
    predictor = synthetic_wave_data.mean("lon")
    predictand = predictor * 2.0 + 1.0
    correlation = pearson_correlation(predictor, predictand)
    regression = linear_regression(predictor, predictand)
    assert float(correlation.mean()) > 0.99
    assert "slope" in regression
    assert "correlation" in regression


def test_linear_trend_uses_numeric_dimension_name_for_units() -> None:
    series = xr.DataArray(
        np.array([1.0, 2.0, 3.5, 5.0]),
        dims=("year",),
        coords={"year": [2001, 2002, 2003, 2004]},
        attrs={"units": "K"},
    )
    result = linear_trend(series, dim="year")
    assert result["slope"].attrs["units"] == "K / year"


def test_one_sample_ttest_detects_nonzero_mean() -> None:
    series = xr.DataArray(
        np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
        dims=("time",),
        coords={"time": np.arange(5)},
    )
    result = one_sample_ttest(series)
    assert float(result["mean"]) > 0.0
    assert float(result["pvalue"]) < 0.05
