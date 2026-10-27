"""Statistical utilities for climate diagnostics."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats as scipy_stats

from tropical_wave_tools.io import rename_standard_coordinates, to_dataarray


def _prepare_array(data: xr.DataArray, *, dim: str = "time") -> xr.DataArray:
    array = rename_standard_coordinates(to_dataarray(data))
    if dim not in array.dims:
        raise ValueError(f"Input data must contain the '{dim}' dimension.")
    return array


def variance(data: xr.DataArray, *, dim: str = "time", ddof: int = 0) -> xr.DataArray:
    """Return variance along a dimension."""
    return _prepare_array(data, dim=dim).var(dim=dim, ddof=ddof)


def standard_deviation(data: xr.DataArray, *, dim: str = "time", ddof: int = 0) -> xr.DataArray:
    """Return standard deviation along a dimension."""
    return _prepare_array(data, dim=dim).std(dim=dim, ddof=ddof)


def _time_axis_in_days(data: xr.DataArray, dim: str) -> xr.DataArray:
    coordinate = data[dim]
    if np.issubdtype(coordinate.dtype, np.datetime64):
        delta_days = (coordinate - coordinate.isel({dim: 0})) / np.timedelta64(1, "D")
        return xr.DataArray(delta_days.astype("float64"), dims=(dim,), coords={dim: coordinate})
    return coordinate.astype("float64")


def _linregress_1d(
    x: np.ndarray,
    y: np.ndarray,
    adjust_for_autocorrelation: bool = False,
) -> tuple[float, float, float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    xv, yv = x[valid], y[valid]
    result = scipy_stats.linregress(xv, yv)
    pvalue, stderr = float(result.pvalue), float(result.stderr)

    if adjust_for_autocorrelation and xv.size > 3:
        # Effective sample size n * (1 - r1) / (1 + r1) from the lag-1 autocorrelation of the
        # residuals (Santer et al. 2000); it can only widen the uncertainty.
        residual = yv - (result.intercept + result.slope * xv)
        centred = residual - residual.mean()
        denominator = float(np.sum(centred**2))
        r1 = float(np.sum(centred[1:] * centred[:-1]) / denominator) if denominator > 0 else 0.0
        if r1 > 0.0:
            n_eff = max(xv.size * (1.0 - r1) / (1.0 + r1), 3.0)
            stderr *= float(np.sqrt((xv.size - 2.0) / (n_eff - 2.0)))
            t_stat = result.slope / stderr
            pvalue = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=n_eff - 2.0))

    return (
        float(result.slope),
        float(result.intercept),
        float(result.rvalue),
        pvalue,
        stderr,
    )


def linear_trend(
    data: xr.DataArray,
    *,
    dim: str = "time",
    adjust_for_autocorrelation: bool = False,
) -> xr.Dataset:
    """Estimate a linear trend using least squares.

    By default ``pvalue`` and ``stderr`` assume independent samples, which badly overstates
    significance for autocorrelated series (daily data, AR(1) with rho=0.9: ~60% false
    positives at the nominal 5% level). Set ``adjust_for_autocorrelation=True`` to use an
    effective sample size from the lag-1 autocorrelation of the residuals.
    """
    array = _prepare_array(data, dim=dim)
    coordinate = array[dim]
    axis = _time_axis_in_days(array, dim)
    slope, intercept, rvalue, pvalue, stderr = xr.apply_ufunc(
        _linregress_1d,
        axis,
        array,
        kwargs={"adjust_for_autocorrelation": bool(adjust_for_autocorrelation)},
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, float, float],
    )
    slope = slope.rename("slope")
    intercept = intercept.rename("intercept")
    rvalue = rvalue.rename("rvalue")
    pvalue = pvalue.rename("pvalue")
    stderr = stderr.rename("stderr")
    rate_unit = "day" if np.issubdtype(coordinate.dtype, np.datetime64) else dim
    slope.attrs["units"] = f"{array.attrs.get('units', 'unknown')} / {rate_unit}"
    stderr.attrs["units"] = slope.attrs["units"]
    pvalue.attrs["long_name"] = "two-sided p-value for slope"
    return xr.Dataset(
        {
            "slope": slope,
            "intercept": intercept,
            "rvalue": rvalue,
            "pvalue": pvalue,
            "stderr": stderr,
        }
    )


def _ttest_1samp_1d(values: np.ndarray, popmean: float) -> tuple[float, float, float]:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return (np.nan, np.nan, float(valid.sum()))

    result = scipy_stats.ttest_1samp(values[valid], popmean=popmean, alternative="two-sided")
    return (float(result.statistic), float(result.pvalue), float(valid.sum()))


def one_sample_ttest(
    data: xr.DataArray,
    *,
    dim: str = "time",
    popmean: float = 0.0,
) -> xr.Dataset:
    """Run a two-sided one-sample t-test along one dimension."""
    array = _prepare_array(data, dim=dim)
    statistic, pvalue, sample_size = xr.apply_ufunc(
        _ttest_1samp_1d,
        array,
        kwargs={"popmean": float(popmean)},
        input_core_dims=[[dim]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    return xr.Dataset(
        {
            "statistic": statistic.rename("statistic"),
            "pvalue": pvalue.rename("pvalue"),
            "sample_size": sample_size.rename("sample_size"),
            "mean": array.mean(dim=dim).rename("mean"),
        }
    )


def _align_valid_pairs(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align two arrays and keep only the samples where *both* are finite.

    Means, variances and covariances must be computed over the same samples, otherwise the
    correlation can leave [-1, 1] when the two arrays have different missing values.
    """
    left_aligned, right_aligned = xr.align(
        _prepare_array(left, dim=dim),
        _prepare_array(right, dim=dim),
        join="inner",
    )
    valid = left_aligned.notnull() & right_aligned.notnull()
    return left_aligned.where(valid), right_aligned.where(valid)


def pearson_correlation(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    dim: str = "time",
) -> xr.DataArray:
    """Compute Pearson correlation between two aligned arrays (pairwise-complete samples)."""
    left_aligned, right_aligned = _align_valid_pairs(left, right, dim=dim)
    left_anom = left_aligned - left_aligned.mean(dim=dim)
    right_anom = right_aligned - right_aligned.mean(dim=dim)
    covariance = (left_anom * right_anom).mean(dim=dim)
    denominator = left_anom.std(dim=dim) * right_anom.std(dim=dim)
    return (covariance / denominator).rename("correlation")


def linear_regression(
    predictor: xr.DataArray,
    predictand: xr.DataArray,
    *,
    dim: str = "time",
) -> xr.Dataset:
    """Regress one field onto another and return slope/intercept/correlation."""
    x, y = _align_valid_pairs(predictor, predictand, dim=dim)
    x_anom = x - x.mean(dim=dim)
    y_anom = y - y.mean(dim=dim)

    covariance = (x_anom * y_anom).mean(dim=dim)
    variance_x = (x_anom**2).mean(dim=dim)
    slope = (covariance / variance_x).rename("slope")
    intercept = (y.mean(dim=dim) - slope * x.mean(dim=dim)).rename("intercept")
    correlation = pearson_correlation(x, y, dim=dim)
    return xr.Dataset({"slope": slope, "intercept": intercept, "correlation": correlation})


__all__ = [
    "linear_regression",
    "linear_trend",
    "one_sample_ttest",
    "pearson_correlation",
    "standard_deviation",
    "variance",
]
