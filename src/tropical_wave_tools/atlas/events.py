"""Wave projections, event detection, lagged composites and Hovmoller windows."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import xarray as xr
from scipy.signal import find_peaks

from tropical_wave_tools.atlas._config import (
    ANTISYMMETRIC_WAVES,
    CASE03_HOV_LAT_BANDS,
)


def _lat_range(data: xr.DataArray, lower: float, upper: float) -> xr.DataArray:
    """Select ``lower <= lat <= upper`` for ascending *or* descending latitude axes.

    ``data.sel(lat=slice(lower, upper))`` silently returns an empty selection on a descending axis
    (the usual ERA5 layout), which turned every equatorial average into NaN.
    """
    latitude = np.asarray(data["lat"].values)
    if latitude.size > 1 and latitude[0] > latitude[-1]:
        return data.sel(lat=slice(upper, lower))
    return data.sel(lat=slice(lower, upper))


def equatorial_mean(data: xr.DataArray, *, lat_band: tuple[float, float] = (-5.0, 5.0)) -> xr.DataArray:
    """Compute an equatorial-band mean."""
    return _lat_range(data, *lat_band).mean("lat")


def is_antisymmetric_wave(wave_name: str | None) -> bool:
    """Return whether a wave is typically diagnosed with an antisymmetric projection."""
    if wave_name is None:
        return False
    return wave_name.lower() in ANTISYMMETRIC_WAVES


def wave_longitude_projection(
    data: xr.DataArray,
    *,
    wave_name: str | None = None,
    lat_band: tuple[float, float] = (-5.0, 5.0),
    antisymmetric_lat_band: tuple[float, float] = (-10.0, 10.0),
) -> xr.DataArray:
    """Project a wave onto longitude using a symmetry-aware meridional average."""
    if "lat" not in data.dims:
        projection = data.copy()
        projection_name = "preprojected"
    elif not is_antisymmetric_wave(wave_name):
        projection = equatorial_mean(data, lat_band=lat_band)
        projection_name = "equatorial_mean"
    else:
        lower, upper = antisymmetric_lat_band
        north = _lat_range(data, 0.0, upper)
        south = _lat_range(data, lower, 0.0)
        north = north.where(north["lat"] > 0.0, drop=True)
        south = south.where(south["lat"] < 0.0, drop=True)
        if north.sizes.get("lat", 0) == 0 or south.sizes.get("lat", 0) == 0:
            projection = equatorial_mean(data, lat_band=lat_band)
            projection_name = "equatorial_mean_fallback"
        else:
            projection = north.mean("lat") - south.mean("lat")
            projection_name = "antisymmetric_difference"

    projection = projection.rename(data.name or "wave_projection")
    projection.attrs.update(data.attrs)
    projection.attrs["wave_projection"] = projection_name
    if wave_name is not None:
        projection.attrs["wave_name"] = wave_name.lower()
    return projection


def _minimum_distance_samples(reference: xr.DataArray, min_distance_days: float) -> int:
    """Convert a minimum separation in days to samples using the time step of the series.

    ``scipy.signal.find_peaks`` counts samples. Passing ``min_distance_days`` straight through
    meant "7 samples", i.e. 7 hours for hourly data, so events a few days apart were not merged.
    """
    time = reference["time"]
    if time.size > 1 and np.issubdtype(time.dtype, np.datetime64):
        step_days = float(np.median(np.diff(time.values)) / np.timedelta64(1, "D"))
        if step_days > 0.0:
            return max(1, round(min_distance_days / step_days))
    return max(1, int(min_distance_days))


def _find_events(
    reference: xr.DataArray,
    *,
    sign: str,
    threshold_std: float,
    min_distance_days: float,
) -> np.ndarray:
    """Indices of peaks of ``reference`` beyond ``threshold_std`` standard deviations."""
    if sign not in {"negative", "positive", "absolute"}:
        raise ValueError("`sign` must be one of {'negative', 'positive', 'absolute'}.")
    values = np.asarray(reference.values, dtype=float)
    if not np.isfinite(values).any():
        warnings.warn(
            "The reference series has no finite values (check the latitude/longitude selection); "
            "no events were detected.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.array([], dtype=int)
    sigma = float(np.nanstd(values))
    if not np.isfinite(sigma) or sigma == 0.0:
        return np.array([], dtype=int)

    signal = {"negative": -values, "positive": values, "absolute": np.abs(values)}[sign]
    peaks, _ = find_peaks(
        signal,
        height=threshold_std * sigma,
        distance=_minimum_distance_samples(reference, min_distance_days),
    )
    return peaks.astype(int)


def detect_wave_events(
    filtered_olr: xr.DataArray,
    *,
    wave_name: str | None = None,
    lat_band: tuple[float, float] = (-5.0, 5.0),
    antisymmetric_lat_band: tuple[float, float] = (-10.0, 10.0),
    lon_ref: float = 180.0,
    sign: str = "negative",
    threshold_std: float = 1.0,
    min_distance_days: int = 7,
) -> tuple[np.ndarray, xr.DataArray]:
    """Detect strong wave events from equatorial filtered OLR at a reference longitude.

    The reference is the meridional projection at the grid longitude nearest ``lon_ref``. Events
    are peaks of ``-x`` (``sign="negative"``, convective OLR minima), ``x`` or ``|x|`` that exceed
    ``threshold_std`` standard deviations of the whole series; of two peaks closer than
    ``min_distance_days`` days only the larger is kept. Indices refer to the time axis of the input.
    """
    reference = wave_longitude_projection(
        filtered_olr,
        wave_name=wave_name,
        lat_band=lat_band,
        antisymmetric_lat_band=antisymmetric_lat_band,
    ).sel(lon=lon_ref, method="nearest")
    events = _find_events(
        reference, sign=sign, threshold_std=threshold_std, min_distance_days=min_distance_days
    )
    return events, reference


def detect_point_events(
    data: xr.DataArray,
    *,
    base_lat: float,
    base_lon: float,
    sign: str = "negative",
    threshold_std: float = 1.0,
    min_distance_days: int = 7,
) -> tuple[np.ndarray, xr.DataArray]:
    """Detect strong events from a filtered point time series at a literature-style base point."""
    reference = data.sel(lat=base_lat, lon=base_lon, method="nearest").rename(data.name or "point_reference")
    reference.attrs.update(data.attrs)
    reference.attrs["wave_projection"] = "base_point"
    reference.attrs["base_lat"] = float(reference["lat"].item())
    reference.attrs["base_lon"] = float(reference["lon"].item())
    events = _find_events(
        reference, sign=sign, threshold_std=threshold_std, min_distance_days=min_distance_days
    )
    return events, reference


def select_hovmoller_window(
    filtered_olr: xr.DataArray,
    *,
    wave_name: str | None = None,
    lat_band: tuple[float, float] = (-5.0, 5.0),
    antisymmetric_lat_band: tuple[float, float] = (-10.0, 10.0),
    lon_ref: float = 180.0,
    window_days: int = 180,
) -> slice:
    """Select a representative time window around the strongest equatorial event."""
    reference = wave_longitude_projection(
        filtered_olr,
        wave_name=wave_name,
        lat_band=lat_band,
        antisymmetric_lat_band=antisymmetric_lat_band,
    ).sel(lon=lon_ref, method="nearest")
    values = np.asarray(reference.values, dtype=float)
    center_index = int(np.nanargmax(np.abs(values)))
    half_window = window_days // 2
    start = max(0, center_index - half_window)
    stop = min(reference.sizes["time"], start + window_days)
    if stop - start < window_days:
        start = max(0, stop - window_days)
    return slice(reference.time.values[start], reference.time.values[stop - 1])


def _hovmoller_projection(
    data: xr.DataArray,
    *,
    wave_name: str | None = None,
    lat_band: tuple[float, float] = (-5.0, 5.0),
) -> xr.DataArray:
    """Return the longitude projection used for Case 03 diagnostics."""
    wave_key = wave_name.lower() if wave_name is not None else None
    custom_lat_band = CASE03_HOV_LAT_BANDS.get(wave_key)
    if custom_lat_band is None or "lat" not in data.dims:
        return wave_longitude_projection(data, wave_name=wave_name, lat_band=lat_band)

    projection = _lat_range(data, *custom_lat_band).mean("lat").rename(data.name or "wave_projection")
    projection.attrs.update(data.attrs)
    projection.attrs["wave_projection"] = "off_equatorial_mean"
    projection.attrs["wave_name"] = wave_key
    projection.attrs["lat_band"] = custom_lat_band
    return projection


def lagged_composite(
    data: xr.DataArray,
    event_indices: np.ndarray,
    *,
    lags: Sequence[int],
) -> xr.DataArray:
    """Composite a field at fixed lags relative to event indices.

    Events whose shifted index falls outside the record are dropped *for that lag only*, so the
    event set can shrink towards large positive or negative lags. The number of events behind each
    lag is returned as the ``n_events`` coordinate along ``lag``.
    """
    composites: list[xr.DataArray] = []
    counts: list[int] = []
    nt = data.sizes["time"]
    template = data.isel(time=0, drop=True) * np.nan
    events = np.asarray(event_indices, dtype=int)

    for lag in lags:
        lag_indices = events + int(lag)
        lag_indices = lag_indices[(lag_indices >= 0) & (lag_indices < nt)]
        counts.append(int(lag_indices.size))
        if lag_indices.size == 0:
            composite = template.copy()
        else:
            composite = data.isel(time=lag_indices).mean("time")
        composites.append(composite.expand_dims(lag=[int(lag)]))

    result = xr.concat(composites, dim="lag")
    return result.assign_coords(n_events=("lag", counts))
