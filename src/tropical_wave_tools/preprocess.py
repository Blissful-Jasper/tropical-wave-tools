"""Common climate preprocessing routines."""

from __future__ import annotations

from typing import Optional, Tuple

import xarray as xr

from tropical_wave_tools.io import rename_standard_coordinates, standardize_data, to_dataarray
from tropical_wave_tools.preprocessing import (
    build_wk_decomposition_layout,
    decompose_symmetric_antisymmetric,
    detrend_with_mean,
    extract_low_harmonics,
    remove_annual_cycle_fft,
    remove_daily_climatology,
    smooth_121,
)


def select_region(
    data: xr.DataArray,
    *,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    lon_target: str = "0_360",
) -> xr.DataArray:
    """Select a latitude-longitude region after coordinate normalization."""
    array = standardize_data(data, lon_target=lon_target)

    if lat_range is not None:
        lat_min, lat_max = sorted(lat_range)
        array = array.sel(lat=slice(lat_min, lat_max))

    if lon_range is not None:
        lon_min, lon_max = lon_range
        if lon_min <= lon_max:
            array = array.sel(lon=slice(lon_min, lon_max))
        else:
            left = array.sel(lon=slice(lon_min, 360.0))
            right = array.sel(lon=slice(0.0, lon_max))
            array = xr.concat([left, right], dim="lon")

    return array


def select_time(
    data: xr.DataArray,
    *,
    time_range: Tuple[str, str],
) -> xr.DataArray:
    """Subset a time range."""
    array = rename_standard_coordinates(to_dataarray(data))
    if "time" not in array.dims:
        raise ValueError("Input data must contain a time-like dimension.")
    return array.sel(time=slice(*time_range))


def compute_climatology(
    data: xr.DataArray,
    *,
    group: str = "dayofyear",
) -> xr.DataArray:
    """Compute a grouped climatology such as daily, monthly, or seasonal mean."""
    array = rename_standard_coordinates(to_dataarray(data))
    if "time" not in array.dims:
        raise ValueError("Input data must contain a time-like dimension.")
    if group == "season":
        return array.groupby("time.season").mean(dim="time")
    if group == "month":
        return array.groupby("time.month").mean(dim="time")
    if group == "dayofyear":
        return array.groupby("time.dayofyear").mean(dim="time")
    raise ValueError("`group` must be one of {'dayofyear', 'month', 'season'}.")


def compute_anomaly(
    data: xr.DataArray,
    *,
    group: str = "dayofyear",
) -> xr.DataArray:
    """Subtract the grouped climatology from the original series."""
    array = rename_standard_coordinates(to_dataarray(data))
    if "time" not in array.dims:
        raise ValueError("Input data must contain a time-like dimension.")
    climatology = compute_climatology(array, group=group)
    if group == "season":
        return array.groupby("time.season") - climatology
    if group == "month":
        return array.groupby("time.month") - climatology
    return array.groupby("time.dayofyear") - climatology


def monthly_mean(data: xr.DataArray) -> xr.DataArray:
    """Compute monthly means."""
    array = rename_standard_coordinates(to_dataarray(data))
    if "time" not in array.dims:
        raise ValueError("Input data must contain a time-like dimension.")
    return array.resample(time="MS").mean()


def seasonal_mean(
    data: xr.DataArray,
    *,
    season: Optional[str] = None,
) -> xr.DataArray:
    """
    Compute seasonal means.

    If ``season`` is provided (e.g. ``"JJA"``), the output is grouped by year.
    """
    array = rename_standard_coordinates(to_dataarray(data))
    if "time" not in array.dims:
        raise ValueError("Input data must contain a time-like dimension.")
    if season is None:
        return array.groupby("time.season").mean(dim="time")
    season = season.upper()
    season_data = array.where(array.time.dt.season == season, drop=True)
    return season_data.groupby("time.year").mean(dim="time")


__all__ = [
    "build_wk_decomposition_layout",
    "compute_anomaly",
    "compute_climatology",
    "decompose_symmetric_antisymmetric",
    "detrend_with_mean",
    "extract_low_harmonics",
    "monthly_mean",
    "remove_annual_cycle_fft",
    "remove_daily_climatology",
    "seasonal_mean",
    "select_region",
    "select_time",
    "smooth_121",
]
