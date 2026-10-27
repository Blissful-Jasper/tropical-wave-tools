"""Monthly RMS, calendar-month climatologies with significance, and variance fractions."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tropical_wave_tools.atlas._config import (
    CASE05_LAT_RANGE,
)
from tropical_wave_tools.diagnostics import (
    area_weighted_mean,
)
from tropical_wave_tools.stats import one_sample_ttest


def compute_monthly_rms(data: xr.DataArray) -> xr.DataArray:
    """Compute monthly root-mean-square amplitude."""
    return np.sqrt((data**2).groupby("time.month").mean("time"))


def compute_yearly_rms(data: xr.DataArray) -> xr.DataArray:
    """Compute yearly root-mean-square amplitude."""
    return np.sqrt((data**2).groupby("time.year").mean("time"))


def compute_monthly_rms_samples(data: xr.DataArray) -> xr.DataArray:
    """Compute monthly RMS samples on the original time axis."""
    return np.sqrt((data**2).resample(time="MS").mean()).rename("monthly_rms")


def _climatology_and_significance(monthly_samples: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Calendar-month climatology of monthly samples and a t-test of each month against the annual mean.

    For every calendar month, ``p`` tests whether that month's anomaly from its own year's mean
    differs from zero across years. Only years with all 12 months enter the annual mean, so a
    partial first or last year cannot bias the anomalies (the climatology still uses every month).
    """
    climatology = monthly_samples.groupby("time.month").mean("time")
    climatology = climatology.reindex(month=np.arange(1, 13, dtype=int))

    months_per_year = monthly_samples["time"].to_series().groupby(monthly_samples["time"].dt.year.values).size()
    complete_years = months_per_year.index[months_per_year.values == 12].to_numpy()
    complete = monthly_samples.sel(time=np.isin(monthly_samples["time"].dt.year.values, complete_years))
    if complete.sizes["time"] > 0:
        annual_mean = complete.groupby("time.year").mean("time")
        monthly_anomaly = complete.groupby("time.year") - annual_mean
    else:
        monthly_anomaly = complete

    pvalues: list[xr.DataArray] = []
    for month in range(1, 13):
        samples = monthly_anomaly.where(monthly_anomaly["time"].dt.month == month, drop=True)
        template = climatology.sel(month=month, drop=True)
        if samples.sizes.get("time", 0) < 2:
            pvalue = xr.full_like(template, np.nan).rename("pvalue")
        else:
            pvalue = one_sample_ttest(samples, dim="time")["pvalue"]
        pvalues.append(pvalue.expand_dims(month=[month]))

    return climatology, xr.concat(pvalues, dim="month")


def compute_monthly_climatology_and_significance(data: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Return monthly RMS climatology and a p-value against each year's annual-mean RMS."""
    return _climatology_and_significance(compute_monthly_rms_samples(data))


def compute_longitude_mean_monthly_rms_climatology_and_significance(
    data: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return monthly RMS climatology after averaging the monthly RMS amplitude over longitude."""
    monthly_samples = compute_monthly_rms_samples(data)
    if "lon" in monthly_samples.dims:
        monthly_samples = monthly_samples.mean("lon")
    return _climatology_and_significance(monthly_samples)


def compute_monthly_variance_fraction_samples(
    raw_anomaly: xr.DataArray,
    filtered: xr.DataArray,
    *,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
) -> xr.DataArray:
    """Return monthly variance fractions (%) for one filtered wave field."""
    raw_monthly_variance = raw_anomaly.resample(time="MS").var("time")
    filtered_monthly_variance = filtered.resample(time="MS").var("time")
    raw_monthly_variance, filtered_monthly_variance = xr.align(
        raw_monthly_variance,
        filtered_monthly_variance,
        join="inner",
    )
    if {"lat", "lon"}.issubset(raw_monthly_variance.dims) and (lat_range is not None or lon_range is not None):
        raw_monthly_variance = area_weighted_mean(
            raw_monthly_variance,
            lat_range=lat_range,
            lon_range=lon_range,
        )
        filtered_monthly_variance = area_weighted_mean(
            filtered_monthly_variance,
            lat_range=lat_range,
            lon_range=lon_range,
        )
    ratio = 100.0 * filtered_monthly_variance / xr.where(raw_monthly_variance > 0.0, raw_monthly_variance, np.nan)
    ratio = ratio.rename("variance_fraction")
    ratio.attrs["units"] = "%"
    ratio.attrs["long_name"] = "Filtered variance fraction"
    return ratio


def summarize_variance_fraction_cycle(
    variance_fraction: xr.DataArray,
    *,
    lat_range: tuple[float, float] = CASE05_LAT_RANGE,
    lon_range: tuple[float, float] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Area-average a monthly variance-fraction field and summarize by calendar month."""
    if {"lat", "lon"}.issubset(variance_fraction.dims):
        regional_series = area_weighted_mean(
            variance_fraction,
            lat_range=lat_range,
            lon_range=lon_range,
        )
    else:
        regional_series = variance_fraction
    monthly_mean = regional_series.groupby("time.month").mean("time")
    monthly_std = regional_series.groupby("time.month").std("time")
    monthly_mean.attrs["units"] = "%"
    monthly_std.attrs["units"] = "%"
    return monthly_mean, monthly_std
