"""Statistical utilities for climate diagnostics."""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

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


def linear_trend(data: xr.DataArray, *, dim: str = "time") -> xr.Dataset:
    """Estimate a linear trend using least squares."""
    array = _prepare_array(data, dim=dim)
    axis = _time_axis_in_days(array, dim)
    fitted = array.assign_coords({dim: axis})
    coefficients = fitted.polyfit(dim=dim, deg=1)["polyfit_coefficients"]
    slope = coefficients.sel(degree=1).reset_coords(drop=True).rename("slope")
    intercept = coefficients.sel(degree=0).reset_coords(drop=True).rename("intercept")
    slope.attrs["units"] = f"{array.attrs.get('units', 'unknown')} / day"
    return xr.Dataset({"slope": slope, "intercept": intercept})


def pearson_correlation(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    dim: str = "time",
) -> xr.DataArray:
    """Compute Pearson correlation between two aligned arrays."""
    left_aligned, right_aligned = xr.align(
        _prepare_array(left, dim=dim),
        _prepare_array(right, dim=dim),
        join="inner",
    )
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
    x, y = xr.align(_prepare_array(predictor, dim=dim), _prepare_array(predictand, dim=dim), join="inner")
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
    "pearson_correlation",
    "standard_deviation",
    "variance",
]
