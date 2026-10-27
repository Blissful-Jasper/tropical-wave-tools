"""Lagged-regression Hovmoller diagnostics (Case 10)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from tropical_wave_tools.atlas._config import (
    CASE10_BASE_POINTS,
    CASE10_LAGS,
    CASE10_LON_WINDOWS,
    CASE10_WINDOW_VARIANCE_DAYS,
)
from tropical_wave_tools.atlas.events import (
    _lat_range,
)
from tropical_wave_tools.stats import linear_regression


def _case10_lat_band(base_lat: float, *, meridional_width: float = 5.0) -> tuple[float, float]:
    """Return the 5-degree meridional averaging band used in the paper."""
    half_width = 0.5 * float(meridional_width)
    lower = max(-30.0, float(base_lat) - half_width)
    upper = min(30.0, float(base_lat) + half_width)
    return (lower, upper)


def _case10_activity_mask(reference: xr.DataArray, *, window_days: int) -> xr.DataArray:
    """Return a strong-activity mask following the paper's running-variance rule."""
    reference_anom = reference - reference.mean("time")
    rolling_variance = reference_anom.rolling(
        time=max(int(window_days), 1),
        center=True,
        min_periods=max(3, int(window_days) // 2),
    ).var()
    threshold = float(rolling_variance.mean("time", skipna=True).item())
    if not np.isfinite(threshold):
        threshold = 0.0
    active = xr.where(rolling_variance >= threshold, 1.0, np.nan)
    active.attrs["window_days"] = int(window_days)
    active.attrs["threshold"] = threshold
    return active.rename("active_mask")


def _lagged_regression_slope(
    predictor: xr.DataArray,
    predictand: xr.DataArray,
    *,
    lags: Sequence[int],
    active_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Regress a lagged predictand onto one predictor and stack slopes by lag."""
    patterns: list[xr.DataArray] = []
    predictor_std = predictor - predictor.mean("time")
    predictor_scale = float(predictor_std.std("time", skipna=True).item())
    if np.isfinite(predictor_scale) and predictor_scale > 0.0:
        predictor_std = predictor_std / predictor_scale

    for lag in lags:
        lagged = predictand.shift(time=-int(lag))
        x, y = xr.align(predictor_std, lagged, join="inner")
        if active_mask is not None:
            mask, x, y = xr.align(active_mask, x, y, join="inner")
            x = x.where(mask.notnull())
            y = y.where(mask.notnull())
        slope = linear_regression(x, y, dim="time")["slope"]
        patterns.append(slope.expand_dims(lag=[int(lag)]))

    regression = xr.concat(patterns, dim="lag")
    regression.attrs.update(predictand.attrs)
    regression.attrs["regression_predictor"] = predictor.name or "predictor"
    return regression


def compute_case10_regression_hovmoller(
    raw_olr_anomaly: xr.DataArray,
    filtered_olr: xr.DataArray,
    *,
    wave_name: str,
) -> xr.Dataset:
    """Compute Lubis & Jacobi style lagged-regression Hovmoller diagnostics."""
    wave_key = wave_name.lower()
    if wave_key not in CASE10_BASE_POINTS:
        raise ValueError(f"Case 10 is not configured for wave '{wave_name}'.")

    base_lat, base_lon = CASE10_BASE_POINTS[wave_key]
    lat_band = _case10_lat_band(base_lat)
    lags = CASE10_LAGS[wave_key]
    lon_min, lon_max = CASE10_LON_WINDOWS[wave_key]

    filtered_reference = filtered_olr.sel(lat=base_lat, lon=base_lon, method="nearest").rename("filtered_reference")
    active_mask = _case10_activity_mask(
        filtered_reference,
        window_days=CASE10_WINDOW_VARIANCE_DAYS[wave_key],
    )

    convective_proxy = (-1.0 * raw_olr_anomaly).rename("convective_proxy")
    convective_proxy.attrs["long_name"] = "Convective proxy from negative OLR anomalies"
    convective_proxy.attrs["units"] = raw_olr_anomaly.attrs.get("units", "W m^-2")

    shading_source = _lat_range(convective_proxy, *lat_band).mean("lat")
    contour_source = _lat_range(filtered_olr, *lat_band).mean("lat")

    shading = _lagged_regression_slope(
        filtered_reference,
        shading_source,
        lags=lags,
        active_mask=active_mask,
    ).sel(lon=slice(lon_min, lon_max))
    contours = _lagged_regression_slope(
        filtered_reference,
        contour_source,
        lags=lags,
        active_mask=active_mask,
    ).sel(lon=slice(lon_min, lon_max))

    shading.name = "shading"
    sigma_unit = f"{raw_olr_anomaly.attrs.get('units', 'W m^-2')} per standard deviation"
    shading.attrs["long_name"] = "Regression of the convective proxy (-OLR anomaly) onto the base-point reference"
    shading.attrs["predictor"] = "filtered OLR at the base point, standardised (+1 sigma = suppressed convection)"
    shading.attrs["units"] = sigma_unit
    shading.attrs["lag_convention"] = "positive lag: the field lags the reference"
    shading.attrs["base_lat"] = float(filtered_reference["lat"].item())
    shading.attrs["base_lon"] = float(filtered_reference["lon"].item())
    shading.attrs["lat_band"] = lat_band
    shading.attrs["window_days"] = CASE10_WINDOW_VARIANCE_DAYS[wave_key]
    contours.name = "contours"
    contours.attrs.update(shading.attrs)
    contours.attrs["long_name"] = "Regression of the filtered OLR onto the base-point reference"

    return xr.Dataset(
        {
            "shading": shading,
            "contours": contours,
            "reference": filtered_reference,
            "active_mask": active_mask,
        }
    )
