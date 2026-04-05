"""High-level local-data atlas generation for equatorial wave diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks

from tropical_wave_tools.config import DEFAULT_WAVE_SPECS, SpectralConfig
from tropical_wave_tools.diagnostics import area_weighted_mean, horizontal_divergence, relative_vorticity
from tropical_wave_tools.eof import EOFAnalyzer
from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.io import load_dataarray, save_dataset
from tropical_wave_tools.plotting import (
    plot_case05_regional_variance_cycles,
    plot_case05_seasonal_variance_cycles,
    plot_eof_modes_with_wind,
    plot_horizontal_structure,
    plot_hovmoller_triptych,
    plot_paper_style_hovmoller,
    plot_lag_longitude_evolution,
    plot_lagged_horizontal_structure,
    plot_monthly_cycle,
    plot_monthly_longitude_heatmap,
    plot_wave_annual_trend_comparison,
    plot_wave_monthly_cycle_comparison,
    plot_wave_monthly_longitude_comparison,
    plot_spatial_std_triptych,
    plot_wave_spatial_comparison,
    plot_wind_diagnostics_panel,
    plot_wk_spectrum,
)
from tropical_wave_tools.preprocess import compute_anomaly
from tropical_wave_tools.spectral import analyze_wk_spectrum
from tropical_wave_tools.stats import linear_regression, linear_trend, one_sample_ttest


PathLike = Union[str, Path]

DEFAULT_LOCAL_PATHS = {
    "olr": Path("data/local/olr.day.mean.nc"),
    "gpcp": Path("data/local/GPCP_data_1997-2020-2.5x2.5_stand.nc"),
    "u850": Path("data/local/uwnd_850hPa_1979-2024.nc"),
    "v850": Path("data/local/vwnd_850hPa_1979-2024.nc"),
}

COMPARISON_WAVE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("large_scale", ("kelvin", "er", "mjo")),
    ("westward", ("mrg", "td")),
)

CASE04_WAVES: tuple[str, ...] = ("kelvin", "er", "mjo", "mrg", "td")
CASE04_LAT_RANGE: tuple[float, float] = (-25.0, 25.0)
CASE05_WAVES: tuple[str, ...] = ("kelvin", "er", "mrg", "td")
CASE05_LAT_RANGE: tuple[float, float] = (-20.0, 20.0)
CASE05_REGION_RANGES: tuple[tuple[str, tuple[float, float]], ...] = (
    ("africa", (343.0, 50.0)),
    ("indian_ocean", (51.0, 94.0)),
    ("maritime_continent", (95.0, 150.0)),
    ("west_pacific", (151.0, 210.0)),
    ("east_pacific", (211.0, 270.0)),
    ("america", (271.0, 325.0)),
    ("atlantic", (326.0, 342.0)),
)
CASE05_REGION_LABELS: dict[str, str] = {
    "africa": "Africa",
    "indian_ocean": "Indian Ocean",
    "maritime_continent": "Maritime Continent",
    "west_pacific": "West Pacific",
    "east_pacific": "East Pacific",
    "america": "America",
    "atlantic": "Atlantic",
}
CASE05_PRECIP_VARIABLE = "precipitation"
CASE06_WAVE_QUIVER_STRIDES: dict[str, int] = {
    "kelvin": 3,
    "er": 3,
    "mjo": 4,
    "mrg": 3,
    "td": 2,
}
CASE06_WAVE_QUIVER_SCALES: dict[str, float] = {
    "kelvin": 20.0,
    "er": 22.0,
    "mjo": 26.0,
    "mrg": 18.0,
    "td": 10.0,
}
CASE07_WAVES: tuple[str, ...] = ("kelvin", "er", "mjo", "mrg", "td")
CASE07_WAVE_LAT_RANGES: dict[str, tuple[float, float]] = {
    "kelvin": (-12.5, 12.5),
    "er": (-18.0, 18.0),
    "mjo": (-20.0, 20.0),
    "mrg": (-18.0, 18.0),
    "td": (-22.5, 22.5),
}
CASE07_WAVE_PROJECTIONS: dict[str, str] = {
    "kelvin": "symmetric",
    "er": "symmetric",
    "mjo": "full",
    "mrg": "antisymmetric",
    "td": "full",
}
CASE07_WAVE_QUIVER_STRIDES: dict[str, int] = {
    "kelvin": 3,
    "er": 3,
    "mjo": 4,
    "mrg": 3,
    "td": 3,
}
CASE07_WAVE_QUIVER_SCALES: dict[str, float] = {
    "kelvin": 18.0,
    "er": 18.0,
    "mjo": 17.0,
    "mrg": 15.0,
    "td": 16.0,
}
CASE08_WAVE_LAGS: dict[str, tuple[int, ...]] = {
    "kelvin": (-6, -4, -2, 0, 2, 4, 6),
    "er": (-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12),
    "mjo": (-24, -18, -12, -6, 0, 6, 12, 18, 24),
    "mrg": (-3, -2, -1, 0, 1, 2, 3),
    "td": (-4, -2, 0, 2, 4),
}
CASE08_WAVE_MAP_LAGS: dict[str, tuple[int, ...]] = {
    "kelvin": (-4, -2, 0, 2, 4),
    "er": (-12, -6, 0, 6, 12),
    "mjo": (-24, -12, 0, 12, 24),
    "mrg": (-2, -1, 0, 1, 2),
    "td": (-4, -2, 0, 2, 4),
}
CASE08_WAVE_FOCUS_HALF_WIDTH: dict[str, float] = {
    "kelvin": 80.0,
    "er": 90.0,
    "mjo": 100.0,
    "mrg": 75.0,
    "td": 75.0,
}
CASE08_WAVE_FOCUS_CENTERS: dict[str, float] = {
    "kelvin": 200.0,
    "er": 172.5,
    "mrg": 172.5,
    "td": 133.0,
    "mjo": 150.0,
}
CASE08_EVENT_LAT_BANDS: dict[str, tuple[float, float]] = {
    "td": (5.0, 15.0),
}
CASE08_EVENT_LON_REFS: dict[str, float] = {
    "er": 172.5,
    "mrg": 172.5,
    "td": 133.0,
    "mjo": 150.0,
}
CASE08_EVENT_BASE_POINTS: dict[str, tuple[float, float]] = {
    "kelvin": (7.0, 200.0),
    "er": (7.5, 172.5),
    "mrg": (7.5, 172.5),
    "td": (7.5, 133.0),
}
CASE08_EVENT_MIN_SPACING: dict[str, int] = {
    "kelvin": 3,
    "er": 8,
    "mjo": 12,
    "mrg": 4,
    "td": 4,
}
CASE08_WAVE_QUIVER_STRIDES: dict[str, int] = {
    "kelvin": 2,
    "er": 2,
    "mjo": 3,
    "mrg": 2,
    "td": 2,
}
CASE08_WAVE_QUIVER_SCALES: dict[str, float] = {
    "kelvin": 8.5,
    "er": 10.5,
    "mjo": 8.0,
    "mrg": 7.5,
    "td": 6.5,
}
CASE08_WAVE_QUIVER_WIDTHS: dict[str, float] = {
    "kelvin": 0.0020,
    "er": 0.00195,
    "mjo": 0.00185,
    "mrg": 0.0020,
    "td": 0.00205,
}
CASE08_WAVE_OLR_LEVEL_COUNT: dict[str, int] = {
    "kelvin": 37,
    "er": 37,
    "mjo": 33,
    "mrg": 37,
    "td": 33,
}
CASE08_WAVE_OLR_RANGE_SCALE: dict[str, float] = {
    "kelvin": 1.5,
    "er": 1.45,
    "mjo": 1.25,
    "mrg": 1.5,
    "td": 1.6,
}
CASE08_WAVE_OLR_MIN_LIMIT: dict[str, float] = {
    "kelvin": 12.0,
    "er": 10.0,
    "mjo": 12.0,
    "mrg": 10.0,
    "td": 5.0,
}
CASE09_WAVES: tuple[str, ...] = ("kelvin", "er", "mjo", "mrg", "td")
CASE10_WAVES: tuple[str, ...] = ("kelvin", "er", "mjo", "mrg", "td")
CASE10_BASE_POINTS: dict[str, tuple[float, float]] = {
    "kelvin": (7.0, 200.0),
    "er": (7.5, 172.5),
    "mjo": (0.0, 90.0),
    "mrg": (7.5, 172.5),
    "td": (7.5, 133.0),
}
CASE10_LAGS: dict[str, tuple[int, ...]] = {
    "kelvin": tuple(range(-15, 16)),
    "er": tuple(range(-20, 21)),
    "mjo": tuple(range(-35, 36)),
    "mrg": tuple(range(-10, 11)),
    "td": tuple(range(-8, 9)),
}
CASE10_WINDOW_VARIANCE_DAYS: dict[str, int] = {
    "kelvin": 17,
    "er": 72,
    "mjo": 96,
    "mrg": 10,
    "td": 10,
}
CASE10_LON_WINDOWS: dict[str, tuple[float, float]] = {
    "kelvin": (90.0, 320.0),
    "er": (30.0, 285.0),
    "mjo": (30.0, 270.0),
    "mrg": (70.0, 255.0),
    "td": (75.0, 225.0),
}
CASE10_XTICKS: dict[str, tuple[float, ...]] = {
    "kelvin": (90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
    "er": (30.0, 90.0, 150.0, 210.0, 270.0),
    "mjo": (30.0, 90.0, 150.0, 210.0, 270.0),
    "mrg": (75.0, 120.0, 165.0, 210.0, 255.0),
    "td": (75.0, 112.5, 150.0, 187.5, 225.0),
}
CASE10_SHADING_RANGE_SCALE: dict[str, float] = {
    "kelvin": 1.65,
    "er": 1.7,
    "mjo": 1.35,
    "mrg": 1.65,
    "td": 1.8,
}
CASE10_SHADING_MIN_LIMIT: dict[str, float] = {
    "kelvin": 12.0,
    "er": 10.0,
    "mjo": 10.0,
    "mrg": 10.0,
    "td": 8.0,
}
CASE10_CONTOUR_TARGET_STEPS: dict[str, int] = {
    "kelvin": 6,
    "er": 6,
    "mjo": 7,
    "mrg": 5,
    "td": 5,
}
CASE10_FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "kelvin": (7.4, 5.15),
    "er": (7.8, 5.2),
    "mjo": (8.1, 5.35),
    "mrg": (7.5, 5.0),
    "td": (7.2, 4.95),
}

CASE03_HOV_LAT_BANDS: dict[str, tuple[float, float]] = {
    # Lubis & Jacobi (2015) diagnose MRG time-longitude evolution from an
    # off-equatorial base latitude near 7.5N rather than a north-minus-south
    # projection; this avoids confusing the westward phase with the eastward
    # group envelope in antisymmetric averages.
    "mrg": (5.0, 10.0),
}
CASE03_HOV_LON_REFS: dict[str, float] = {
    "mrg": 172.5,
}

WAVE_QUIVER_SCALES: dict[str, float] = {
    "kelvin": 42.0,
    "er": 45.0,
    "mjo": 28.0,
    "mrg": 38.0,
    "td": 40.0,
}

ANTISYMMETRIC_WAVES: frozenset[str] = frozenset({"mrg", "eig", "eig0", "ig", "wig"})

WAVE_PROPAGATION_DIRECTIONS: dict[str, str] = {
    "kelvin": "eastward",
    "mjo": "eastward",
    "eig": "eastward",
    "eig0": "eastward",
    "ig": "eastward",
    "er": "westward",
    "mrg": "westward",
    "td": "westward",
    "wig": "westward",
}

WAVE_HOVMOLLER_WINDOWS: dict[str, int] = {
    "kelvin": 120,
    "er": 144,
    "mjo": 240,
    "mrg": 96,
    "td": 72,
}


def _case08_wave_figure_title(wave_name: str) -> str:
    """Return a publication-style figure title for Case 08."""
    titles = {
        "kelvin": "Kelvin wave composite phase evolution",
        "er": "Equatorial Rossby composite phase evolution",
        "mjo": "MJO composite phase evolution",
        "mrg": "Mixed Rossby-Gravity composite phase evolution",
        "td": "Tropical disturbance composite phase evolution",
    }
    return titles.get(wave_name.lower(), f"{wave_name.upper()} regional phase evolution")


def _case10_wave_figure_title(wave_name: str) -> str:
    """Return a publication-style Case 10 panel title."""
    titles = {
        "kelvin": "Kelvin wave",
        "er": "Equatorial Rossby wave",
        "mjo": "Madden-Julian Oscillation",
        "mrg": "Mixed Rossby-Gravity wave",
        "td": "Tropical disturbance",
    }
    return titles.get(wave_name.lower(), wave_name.upper())


def _format_lat_label(value: float) -> str:
    """Format a latitude coordinate into a compact N/S string."""
    hemi = "N" if value >= 0.0 else "S"
    magnitude = abs(float(value))
    if np.isclose(magnitude, round(magnitude)):
        return f"{int(round(magnitude))}{hemi}"
    return f"{magnitude:.1f}{hemi}"


def _format_lon_label(value: float) -> str:
    """Format a 0-360 longitude into a compact E/W string."""
    lon = float(value) % 360.0
    if np.isclose(lon, 0.0) or np.isclose(lon, 360.0):
        return "0"
    if np.isclose(lon, 180.0):
        return "180"
    if lon < 180.0:
        return f"{int(round(lon))}E"
    return f"{int(round(360.0 - lon))}W"


def _group_comparison_waves(waves: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Return ordered wave subsets for less crowded multi-panel comparisons."""
    available = [wave.lower() for wave in waves]
    grouped: list[tuple[str, list[str]]] = []
    assigned: set[str] = set()

    for group_name, members in COMPARISON_WAVE_GROUPS:
        selected = [wave for wave in members if wave in available and wave not in assigned]
        if selected:
            grouped.append((group_name, selected))
            assigned.update(selected)

    leftovers = [wave for wave in available if wave not in assigned]
    if leftovers:
        grouped.append(("other", leftovers))

    return grouped


def _comparison_ncols(n_panels: int) -> int:
    """Choose a readable panel count for grouped atlas figures."""
    return max(1, min(3, int(n_panels)))


def fill_missing_with_time_mean(data: xr.DataArray) -> xr.DataArray:
    """Fill missing values with the local temporal mean, then with zero if needed."""
    time_mean = data.mean("time", skipna=True)
    filled = data.where(np.isfinite(data), time_mean)
    return filled.fillna(0.0)


def load_local_wave_fields(
    *,
    olr_path: PathLike = DEFAULT_LOCAL_PATHS["olr"],
    u850_path: PathLike = DEFAULT_LOCAL_PATHS["u850"],
    v850_path: PathLike = DEFAULT_LOCAL_PATHS["v850"],
    time_range: Optional[Tuple[str, str]] = ("1979-01-01", "2014-12-31"),
    lat_range: Tuple[float, float] = (-20.0, 20.0),
    lon_range: Optional[Tuple[float, float]] = None,
) -> xr.Dataset:
    """Load and align local OLR/U850/V850 datasets on a common grid."""
    olr = load_dataarray(olr_path, variable="olr", time_range=time_range, lat_range=lat_range, lon_range=lon_range)
    u850 = load_dataarray(
        u850_path,
        variable="uwnd",
        time_range=time_range,
        lat_range=lat_range,
        lon_range=lon_range,
    ).rename("u850")
    v850 = load_dataarray(
        v850_path,
        variable="vwnd",
        time_range=time_range,
        lat_range=lat_range,
        lon_range=lon_range,
    ).rename("v850")

    olr, u850, v850 = xr.align(olr, u850, v850, join="inner")
    return xr.Dataset(
        {
            "olr": fill_missing_with_time_mean(olr),
            "u850": fill_missing_with_time_mean(u850),
            "v850": fill_missing_with_time_mean(v850),
        }
    )


def compute_monthly_rms(data: xr.DataArray) -> xr.DataArray:
    """Compute monthly root-mean-square amplitude."""
    return np.sqrt((data**2).groupby("time.month").mean("time"))


def compute_yearly_rms(data: xr.DataArray) -> xr.DataArray:
    """Compute yearly root-mean-square amplitude."""
    return np.sqrt((data**2).groupby("time.year").mean("time"))


def compute_monthly_rms_samples(data: xr.DataArray) -> xr.DataArray:
    """Compute monthly RMS samples on the original time axis."""
    return np.sqrt((data**2).resample(time="MS").mean()).rename("monthly_rms")


def compute_monthly_climatology_and_significance(data: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Return monthly RMS climatology and a p-value against each year's annual-mean RMS."""
    monthly_samples = compute_monthly_rms_samples(data)
    climatology = monthly_samples.groupby("time.month").mean("time")
    annual_mean = monthly_samples.groupby("time.year").mean("time")
    monthly_anomaly = monthly_samples.groupby("time.year") - annual_mean

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


def compute_longitude_mean_monthly_rms_climatology_and_significance(
    data: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return monthly RMS climatology after averaging the monthly RMS amplitude over longitude."""
    monthly_samples = compute_monthly_rms_samples(data)
    if "lon" in monthly_samples.dims:
        monthly_samples = monthly_samples.mean("lon")
    climatology = monthly_samples.groupby("time.month").mean("time")
    annual_mean = monthly_samples.groupby("time.year").mean("time")
    monthly_anomaly = monthly_samples.groupby("time.year") - annual_mean

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


def compute_monthly_variance_fraction_samples(
    raw_anomaly: xr.DataArray,
    filtered: xr.DataArray,
    *,
    lat_range: Optional[tuple[float, float]] = None,
    lon_range: Optional[tuple[float, float]] = None,
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
    lon_range: Optional[tuple[float, float]] = None,
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


def equatorial_mean(data: xr.DataArray, *, lat_band: Tuple[float, float] = (-5.0, 5.0)) -> xr.DataArray:
    """Compute an equatorial-band mean."""
    return data.sel(lat=slice(*lat_band)).mean("lat")


def is_antisymmetric_wave(wave_name: Optional[str]) -> bool:
    """Return whether a wave is typically diagnosed with an antisymmetric projection."""
    if wave_name is None:
        return False
    return wave_name.lower() in ANTISYMMETRIC_WAVES


def wave_longitude_projection(
    data: xr.DataArray,
    *,
    wave_name: Optional[str] = None,
    lat_band: Tuple[float, float] = (-5.0, 5.0),
    antisymmetric_lat_band: Tuple[float, float] = (-10.0, 10.0),
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
        north = data.sel(lat=slice(0.0, upper))
        south = data.sel(lat=slice(lower, 0.0))
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


def _case07_component_field(data: xr.DataArray, *, projection: str) -> xr.DataArray:
    """Return a full-field equatorial component for wave-aware EOF analysis."""
    if projection == "full" or "lat" not in data.dims:
        component = data.copy()
    else:
        mirrored = data.interp(lat=(-data["lat"]).values)
        mirrored = mirrored.assign_coords(lat=data["lat"])
        if projection == "symmetric":
            component = 0.5 * (data + mirrored)
        elif projection == "antisymmetric":
            component = 0.5 * (data - mirrored)
        else:
            raise ValueError("`projection` must be one of {'full', 'symmetric', 'antisymmetric'}.")
    component = component.rename(data.name or "wave_component")
    component.attrs.update(data.attrs)
    component.attrs["case07_projection"] = projection
    return component


def _case07_latitude_weights(data: xr.DataArray) -> xr.DataArray:
    """Return sqrt(cos(lat)) weights for latitude-aware EOF fitting."""
    lat_values = np.asarray(data["lat"].values, dtype=float)
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat_values)), 1.0e-6, None))
    return xr.DataArray(weights, dims=("lat",), coords={"lat": data["lat"]}, name="sqrt_coslat")


def detect_wave_events(
    filtered_olr: xr.DataArray,
    *,
    wave_name: Optional[str] = None,
    lat_band: Tuple[float, float] = (-5.0, 5.0),
    antisymmetric_lat_band: Tuple[float, float] = (-10.0, 10.0),
    lon_ref: float = 180.0,
    sign: str = "negative",
    threshold_std: float = 1.0,
    min_distance_days: int = 7,
) -> tuple[np.ndarray, xr.DataArray]:
    """Detect strong wave events from equatorial filtered OLR at a reference longitude."""
    reference = wave_longitude_projection(
        filtered_olr,
        wave_name=wave_name,
        lat_band=lat_band,
        antisymmetric_lat_band=antisymmetric_lat_band,
    ).sel(lon=lon_ref, method="nearest")
    values = np.asarray(reference.values, dtype=float)
    sigma = float(np.nanstd(values))
    if not np.isfinite(sigma) or sigma == 0.0:
        return np.array([], dtype=int), reference

    if sign == "negative":
        peaks, _ = find_peaks(-values, height=threshold_std * sigma, distance=min_distance_days)
    elif sign == "positive":
        peaks, _ = find_peaks(values, height=threshold_std * sigma, distance=min_distance_days)
    elif sign == "absolute":
        peaks, _ = find_peaks(np.abs(values), height=threshold_std * sigma, distance=min_distance_days)
    else:
        raise ValueError("`sign` must be one of {'negative', 'positive', 'absolute'}.")
    return peaks.astype(int), reference


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

    values = np.asarray(reference.values, dtype=float)
    sigma = float(np.nanstd(values))
    if not np.isfinite(sigma) or sigma == 0.0:
        return np.array([], dtype=int), reference

    if sign == "negative":
        peaks, _ = find_peaks(-values, height=threshold_std * sigma, distance=min_distance_days)
    elif sign == "positive":
        peaks, _ = find_peaks(values, height=threshold_std * sigma, distance=min_distance_days)
    elif sign == "absolute":
        peaks, _ = find_peaks(np.abs(values), height=threshold_std * sigma, distance=min_distance_days)
    else:
        raise ValueError("`sign` must be one of {'negative', 'positive', 'absolute'}.")
    return peaks.astype(int), reference


def select_hovmoller_window(
    filtered_olr: xr.DataArray,
    *,
    wave_name: Optional[str] = None,
    lat_band: Tuple[float, float] = (-5.0, 5.0),
    antisymmetric_lat_band: Tuple[float, float] = (-10.0, 10.0),
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
    wave_name: Optional[str] = None,
    lat_band: Tuple[float, float] = (-5.0, 5.0),
) -> xr.DataArray:
    """Return the longitude projection used for Case 03 diagnostics."""
    wave_key = wave_name.lower() if wave_name is not None else None
    custom_lat_band = CASE03_HOV_LAT_BANDS.get(wave_key)
    if custom_lat_band is None or "lat" not in data.dims:
        return wave_longitude_projection(data, wave_name=wave_name, lat_band=lat_band)

    projection = data.sel(lat=slice(*custom_lat_band)).mean("lat").rename(data.name or "wave_projection")
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
    """Composite a field at fixed lags relative to event indices."""
    composites: list[xr.DataArray] = []
    nt = data.sizes["time"]
    template = data.isel(time=0, drop=True) * np.nan

    for lag in lags:
        lag_indices = event_indices + int(lag)
        lag_indices = lag_indices[(lag_indices >= 0) & (lag_indices < nt)]
        if lag_indices.size == 0:
            composite = template.copy()
        else:
            composite = data.isel(time=lag_indices).mean("time")
        composites.append(composite.expand_dims(lag=[int(lag)]))

    return xr.concat(composites, dim="lag")


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
    active_mask: Optional[xr.DataArray] = None,
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

    shading_source = convective_proxy.sel(lat=slice(*lat_band)).mean("lat")
    contour_source = filtered_olr.sel(lat=slice(*lat_band)).mean("lat")

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
    shading.attrs["base_lat"] = float(filtered_reference["lat"].item())
    shading.attrs["base_lon"] = float(filtered_reference["lon"].item())
    shading.attrs["lat_band"] = lat_band
    shading.attrs["window_days"] = CASE10_WINDOW_VARIANCE_DAYS[wave_key]
    contours.name = "contours"
    contours.attrs.update(shading.attrs)

    return xr.Dataset(
        {
            "shading": shading,
            "contours": contours,
            "reference": filtered_reference,
            "active_mask": active_mask,
        }
    )


def compute_wave_eof(
    data: xr.DataArray,
    *,
    wave_name: Optional[str] = None,
    n_modes: int = 2,
) -> tuple[xr.DataArray, xr.DataArray, np.ndarray, xr.DataArray]:
    """Compute wave-aware EOFs and return the analyzed OLR field used for fitting."""
    wave_key = wave_name.lower() if wave_name is not None else None
    lat_range = CASE07_WAVE_LAT_RANGES.get(wave_key, None)
    projection = CASE07_WAVE_PROJECTIONS.get(wave_key, "full")
    analysis_field = data
    if lat_range is not None and "lat" in analysis_field.dims:
        analysis_field = analysis_field.sel(lat=slice(*lat_range))
    analysis_field = _case07_component_field(analysis_field, projection=projection)

    eof_input = analysis_field
    if "lat" in eof_input.dims:
        lat_weights = _case07_latitude_weights(eof_input)
        eof_input = eof_input * lat_weights
    else:
        lat_weights = None

    analyzer = EOFAnalyzer(method="svd")
    results = analyzer.fit(eof_input, n_modes=n_modes, n_harmonics=0)
    eofs = results["eofs"]
    if lat_weights is not None and {"mode", "lat", "lon"}.issubset(eofs.dims):
        eofs = eofs / lat_weights
    pcs = results["pc_scores"]
    explained = np.asarray(results["explained_variance"], dtype=float)
    analysis_field.attrs["case07_projection"] = projection
    if wave_key is not None:
        analysis_field.attrs["wave_name"] = wave_key
    if lat_range is not None:
        analysis_field.attrs["case07_lat_range"] = lat_range
    return eofs, pcs, explained, analysis_field


def regress_field_onto_pcs(
    field: xr.DataArray,
    pcs: xr.DataArray,
    *,
    standardize_pc: bool = False,
) -> xr.DataArray:
    """Regress one field onto each EOF principal component."""
    time_dim = next((dim for dim in pcs.dims if dim != "mode"), None)
    if time_dim is None:
        raise ValueError("`pcs` must contain one non-mode dimension.")

    patterns: list[xr.DataArray] = []
    for mode in pcs["mode"].values:
        pc = pcs.sel(mode=mode)
        field_aligned, pc_aligned = xr.align(field, pc, join="inner")
        pc_anom = pc_aligned - pc_aligned.mean(time_dim)
        if standardize_pc:
            pc_std = float(pc_anom.std(time_dim).item())
            if np.isfinite(pc_std) and pc_std > 0.0:
                pc_anom = pc_anom / pc_std
        field_anom = field_aligned - field_aligned.mean("time")
        denominator = (pc_anom**2).mean(time_dim)
        slope = (field_anom * pc_anom).mean(time_dim) / denominator
        patterns.append(slope.expand_dims(mode=[mode]))

    return xr.concat(patterns, dim="mode")


def _save_wk_triplet(
    dataset: xr.Dataset,
    output_root: Path,
    *,
    spectral_lat_range: Tuple[float, float] = (-15.0, 15.0),
    spectral_config: Optional[SpectralConfig] = None,
) -> None:
    """Generate WK spectra for OLR, U850, and V850."""
    spectra_dir = output_root / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    config = spectral_config or SpectralConfig(window_size_days=128, window_skip_days=32)

    for var_name in ("olr", "u850", "v850"):
        result = analyze_wk_spectrum(dataset[var_name].sel(lat=slice(*spectral_lat_range)), config=config)
        figure, _ = plot_wk_spectrum(result, save_path=spectra_dir / f"wk_spectrum_{var_name}.png")
        plt.close(figure)
        save_dataset(result.to_dataset(), spectra_dir / f"wk_spectrum_{var_name}.nc")


def generate_local_wave_atlas(
    *,
    output_dir: PathLike,
    olr_path: PathLike = DEFAULT_LOCAL_PATHS["olr"],
    u850_path: PathLike = DEFAULT_LOCAL_PATHS["u850"],
    v850_path: PathLike = DEFAULT_LOCAL_PATHS["v850"],
    waves: Optional[Iterable[str]] = None,
    time_range: Optional[Tuple[str, str]] = ("1979-01-01", "2014-12-31"),
    lat_range: Tuple[float, float] = (-20.0, 20.0),
    lat_band: Tuple[float, float] = (-5.0, 5.0),
    lon_ref: float = 180.0,
    hovmoller_days: int = 180,
    event_threshold_std: float = 1.0,
    event_min_spacing_days: int = 7,
    lags: Sequence[int] = (-8, -6, -4, -2, 0, 2, 4, 6, 8),
    n_harm: int = 3,
    n_workers: int = 4,
) -> pd.DataFrame:
    """Generate a publication-oriented atlas from local OLR and low-level wind data."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_local_wave_fields(
        olr_path=olr_path,
        u850_path=u850_path,
        v850_path=v850_path,
        time_range=time_range,
        lat_range=lat_range,
    )
    raw_olr_anomaly = compute_anomaly(dataset["olr"], group="month")
    wave_list = [wave.lower() for wave in (waves or DEFAULT_WAVE_SPECS.keys())]
    multiwave_std_fields: dict[str, xr.DataArray] = {}
    multiwave_olr_lagged: dict[str, xr.DataArray] = {}
    multiwave_u_lagged: dict[str, xr.DataArray] = {}
    multiwave_v_lagged: dict[str, xr.DataArray] = {}
    multiwave_monthly_cycles: dict[str, dict[str, np.ndarray]] = {}
    multiwave_monthly_significance: dict[str, dict[str, xr.DataArray]] = {}
    multiwave_monthly_olr: dict[str, xr.DataArray] = {}
    multiwave_monthly_olr_significance: dict[str, xr.DataArray] = {}
    multiwave_annual_rms: dict[str, xr.DataArray] = {}
    multiwave_annual_trend_pvalue: dict[str, float] = {}
    multiwave_eofs: dict[str, xr.DataArray] = {}
    multiwave_eof_u: dict[str, xr.DataArray] = {}
    multiwave_eof_v: dict[str, xr.DataArray] = {}
    multiwave_eof_explained: dict[str, np.ndarray] = {}
    multiwave_event_indices: dict[str, np.ndarray] = {}
    multiwave_filtered_olr: dict[str, xr.DataArray] = {}

    _save_wk_triplet(dataset, output_root)

    rows: list[dict[str, object]] = []
    for wave_name in wave_list:
        wave_root = output_root / wave_name
        wave_root.mkdir(parents=True, exist_ok=True)

        filtered = {
            name: filter_wave_signal(
                dataset[name],
                wave_name=wave_name,
                method="cckw",
                obs_per_day=1,
                n_harm=n_harm,
                n_workers=n_workers,
            )
            for name in ("olr", "u850", "v850")
        }
        multiwave_filtered_olr[wave_name] = filtered["olr"]

        std_dataset = xr.Dataset({f"{name}_std": field.std("time") for name, field in filtered.items()})
        multiwave_std_fields[wave_name] = std_dataset["olr_std"]
        save_dataset(std_dataset, wave_root / f"{wave_name}_std_maps.nc")
        std_figure, _ = plot_spatial_std_triptych(
            std_dataset["olr_std"],
            std_dataset["u850_std"],
            std_dataset["v850_std"],
            titles=(
                f"{wave_name.upper()} OLR STD",
                f"{wave_name.upper()} U850 STD",
                f"{wave_name.upper()} V850 STD",
            ),
            save_path=wave_root / f"{wave_name}_spatial_std_triptych.png",
        )
        plt.close(std_figure)

        hov_lon_ref = CASE03_HOV_LON_REFS.get(wave_name, lon_ref)
        hov_projection_raw = _hovmoller_projection(raw_olr_anomaly, wave_name=wave_name, lat_band=lat_band)
        hov_projection_olr = _hovmoller_projection(filtered["olr"], wave_name=wave_name, lat_band=lat_band)
        hov_projection_u = _hovmoller_projection(filtered["u850"], wave_name=wave_name, lat_band=lat_band)
        hov_slice = select_hovmoller_window(
            hov_projection_olr,
            wave_name=None,
            lat_band=lat_band,
            lon_ref=hov_lon_ref,
            window_days=WAVE_HOVMOLLER_WINDOWS.get(wave_name, hovmoller_days),
        )
        hov_fields = [
            hov_projection_raw.sel(time=hov_slice),
            hov_projection_olr.sel(time=hov_slice),
            hov_projection_u.sel(time=hov_slice),
        ]
        hov_figure, _ = plot_hovmoller_triptych(
            hov_fields,
            titles=(
                f"{wave_name.upper()} raw OLR anomaly",
                f"{wave_name.upper()} filtered OLR",
                f"{wave_name.upper()} filtered U850",
            ),
            colorbar_labels=("Raw OLR anomaly", "Filtered OLR", "Filtered U850"),
            cmaps=("olr_diverging", "olr_diverging", "wave_diverging"),
            save_path=wave_root / f"{wave_name}_hovmoller_triptych.png",
        )
        plt.close(hov_figure)

        if wave_name in CASE10_WAVES:
            case10_ds = compute_case10_regression_hovmoller(
                raw_olr_anomaly,
                filtered["olr"],
                wave_name=wave_name,
            )
            base_lat, base_lon = CASE10_BASE_POINTS[wave_name]
            base_point_label = f"Base point: {_format_lat_label(base_lat)}, {_format_lon_label(base_lon)}"
            case10_figure, _ = plot_paper_style_hovmoller(
                case10_ds["shading"],
                case10_ds["contours"],
                title=_case10_wave_figure_title(wave_name),
                base_point_label=base_point_label,
                cmap="paper_hovmoller_diverging",
                figsize=CASE10_FIGURE_SIZES.get(wave_name, (7.4, 5.1)),
                shading_label="Regressed convection proxy from -OLR' (W m$^{-2}$)",
                shading_range_scale=CASE10_SHADING_RANGE_SCALE.get(wave_name, 1.25),
                shading_min_vmax=CASE10_SHADING_MIN_LIMIT.get(wave_name),
                contour_target_steps=CASE10_CONTOUR_TARGET_STEPS.get(wave_name, 6),
                xticks=CASE10_XTICKS.get(wave_name),
                save_path=wave_root / f"{wave_name}_paper_hovmoller.png",
            )
            plt.close(case10_figure)
            save_dataset(case10_ds, wave_root / f"{wave_name}_paper_hovmoller.nc")

        event_indices, reference = detect_wave_events(
            filtered["olr"],
            wave_name=wave_name,
            lat_band=lat_band,
            lon_ref=lon_ref,
            threshold_std=event_threshold_std,
            min_distance_days=event_min_spacing_days,
        )
        event_note: Optional[str] = None
        if event_indices.size == 0:
            reference_values = np.asarray(reference.values, dtype=float)
            if np.any(np.isfinite(reference_values)):
                event_indices = np.array([int(np.nanargmax(np.abs(reference_values)))], dtype=int)
                event_note = "Used strongest absolute event as fallback."
            else:
                rows.append({"wave": wave_name, "event_count": 0, "note": "No finite reference values."})
                continue

        event_times = pd.to_datetime(reference.time.isel(time=event_indices).values)
        pd.DataFrame({"event_time": event_times}).to_csv(wave_root / f"{wave_name}_event_times.csv", index=False)
        multiwave_event_indices[wave_name] = np.asarray(event_indices, dtype=int)

        lagged_olr = lagged_composite(filtered["olr"], event_indices, lags=lags)
        lagged_u = lagged_composite(filtered["u850"], event_indices, lags=lags)
        lagged_v = lagged_composite(filtered["v850"], event_indices, lags=lags)
        composite_ds = xr.Dataset({"olr": lagged_olr, "u850": lagged_u, "v850": lagged_v})
        save_dataset(composite_ds, wave_root / f"{wave_name}_lagged_composites.nc")

        structure_figure, _ = plot_horizontal_structure(
            lagged_olr.sel(lag=0),
            lagged_u.sel(lag=0),
            lagged_v.sel(lag=0),
            title=f"{wave_name.upper()} composite horizontal structure (lag 0 d)",
            quiver_scale=WAVE_QUIVER_SCALES.get(wave_name, 55.0),
            integer_colorbar=True,
            save_path=wave_root / f"{wave_name}_horizontal_structure_lag0.png",
        )
        plt.close(structure_figure)
        multiwave_olr_lagged[wave_name] = lagged_olr
        multiwave_u_lagged[wave_name] = lagged_u
        multiwave_v_lagged[wave_name] = lagged_v

        lagged_figure, _ = plot_lagged_horizontal_structure(
            lagged_olr,
            lagged_u,
            lagged_v,
            lags=lags,
            quiver_scale=WAVE_QUIVER_SCALES.get(wave_name, 55.0),
            integer_colorbar=True,
            save_path=wave_root / f"{wave_name}_horizontal_structure_lags.png",
        )
        plt.close(lagged_figure)

        lead_lag_lags = tuple(int(value) for value in CASE08_WAVE_LAGS.get(wave_name, tuple(int(value) for value in lags)))
        lead_lag_event_lat_band = CASE08_EVENT_LAT_BANDS.get(wave_name, lat_band)
        lead_lag_lon_ref = CASE08_EVENT_LON_REFS.get(wave_name, lon_ref)
        lead_lag_event_spacing = CASE08_EVENT_MIN_SPACING.get(wave_name, event_min_spacing_days)
        lead_lag_base_point = CASE08_EVENT_BASE_POINTS.get(wave_name)
        if lead_lag_base_point is not None:
            lead_lag_event_indices, lead_lag_reference = detect_point_events(
                filtered["olr"],
                base_lat=lead_lag_base_point[0],
                base_lon=lead_lag_base_point[1],
                threshold_std=event_threshold_std,
                min_distance_days=lead_lag_event_spacing,
            )
        else:
            lead_lag_event_indices, lead_lag_reference = detect_wave_events(
                filtered["olr"],
                wave_name=wave_name,
                lat_band=lead_lag_event_lat_band,
                lon_ref=lead_lag_lon_ref,
                threshold_std=event_threshold_std,
                min_distance_days=lead_lag_event_spacing,
            )
        if lead_lag_event_indices.size == 0:
            lead_lag_reference_values = np.asarray(lead_lag_reference.values, dtype=float)
            if np.any(np.isfinite(lead_lag_reference_values)):
                lead_lag_event_indices = np.array([int(np.nanargmax(np.abs(lead_lag_reference_values)))], dtype=int)
            else:
                lead_lag_event_indices = event_indices
        lead_lag_olr = lagged_olr
        lead_lag_u = lagged_u
        lead_lag_v = lagged_v
        if tuple(int(value) for value in lagged_olr["lag"].values) != lead_lag_lags:
            lead_lag_olr = lagged_composite(filtered["olr"], lead_lag_event_indices, lags=lead_lag_lags)
            lead_lag_u = lagged_composite(filtered["u850"], lead_lag_event_indices, lags=lead_lag_lags)
            lead_lag_v = lagged_composite(filtered["v850"], lead_lag_event_indices, lags=lead_lag_lags)
        else:
            lead_lag_olr = lagged_composite(filtered["olr"], lead_lag_event_indices, lags=CASE08_WAVE_MAP_LAGS.get(wave_name, lags))
            lead_lag_u = lagged_composite(filtered["u850"], lead_lag_event_indices, lags=CASE08_WAVE_MAP_LAGS.get(wave_name, lags))
            lead_lag_v = lagged_composite(filtered["v850"], lead_lag_event_indices, lags=CASE08_WAVE_MAP_LAGS.get(wave_name, lags))
        if wave_name in CASE08_WAVE_MAP_LAGS:
            lead_lag_figure, _ = plot_lagged_horizontal_structure(
                lead_lag_olr,
                lead_lag_u,
                lead_lag_v,
                lags=CASE08_WAVE_MAP_LAGS[wave_name],
                ncols=3,
                quiver_stride=CASE08_WAVE_QUIVER_STRIDES.get(wave_name, 3),
                quiver_scale=CASE08_WAVE_QUIVER_SCALES.get(
                    wave_name,
                    WAVE_QUIVER_SCALES.get(wave_name, 55.0),
                ),
                integer_colorbar=False,
                colorbar_extend="both",
                olr_quantile=0.95 if wave_name == "mjo" else 0.94,
                olr_level_count=CASE08_WAVE_OLR_LEVEL_COUNT.get(wave_name, 33),
                olr_range_scale=CASE08_WAVE_OLR_RANGE_SCALE.get(wave_name, 1.25),
                olr_min_vmax=CASE08_WAVE_OLR_MIN_LIMIT.get(wave_name),
                wind_overlay="vectors",
                quiver_width=CASE08_WAVE_QUIVER_WIDTHS.get(wave_name, 0.0019),
                quiver_headwidth=3.4,
                quiver_headlength=4.2,
                quiver_headaxislength=3.9,
                focus_longitude=True,
                focus_center_lon=CASE08_WAVE_FOCUS_CENTERS.get(wave_name),
                focus_half_width=CASE08_WAVE_FOCUS_HALF_WIDTH.get(wave_name, 85.0),
                suptitle=_case08_wave_figure_title(wave_name),
                panel_title_template="Day {lag:+d}",
                save_path=wave_root / f"{wave_name}_lead_lag_evolution.png",
            )
        else:
            lead_lag_figure, _ = plot_lag_longitude_evolution(
                wave_longitude_projection(lead_lag_olr, wave_name=wave_name, lat_band=lat_band),
                title=f"{wave_name.upper()} lead-lag evolution",
                integer_colorbar=False,
                colorbar_extend="both",
                save_path=wave_root / f"{wave_name}_lead_lag_evolution.png",
            )
        plt.close(lead_lag_figure)

        divergence = horizontal_divergence(lagged_u.sel(lag=0), lagged_v.sel(lag=0))
        vorticity = relative_vorticity(lagged_u.sel(lag=0), lagged_v.sel(lag=0))
        wind_diag_figure, _ = plot_wind_diagnostics_panel(
            divergence,
            vorticity,
            lagged_u.sel(lag=0),
            lagged_v.sel(lag=0),
            titles=(
                f"{wave_name.upper()} divergence",
                f"{wave_name.upper()} vorticity",
            ),
            quiver_stride=CASE06_WAVE_QUIVER_STRIDES.get(wave_name, 4),
            quiver_scale=CASE06_WAVE_QUIVER_SCALES.get(wave_name, WAVE_QUIVER_SCALES.get(wave_name, 55.0)),
            integer_colorbar=True,
            colorbar_extend="neither",
            save_path=wave_root / f"{wave_name}_wind_diagnostics_lag0.png",
        )
        plt.close(wind_diag_figure)
        save_dataset(
            xr.Dataset({"divergence": divergence, "vorticity": vorticity}),
            wave_root / f"{wave_name}_wind_diagnostics_lag0.nc",
        )

        eofs, pcs, explained, eof_input = compute_wave_eof(filtered["olr"], wave_name=wave_name, n_modes=2)
        eof_olr = regress_field_onto_pcs(eof_input, pcs, standardize_pc=True)
        eof_u = regress_field_onto_pcs(filtered["u850"].sel(lat=eof_input["lat"]), pcs, standardize_pc=True)
        eof_v = regress_field_onto_pcs(filtered["v850"].sel(lat=eof_input["lat"]), pcs, standardize_pc=True)
        multiwave_eofs[wave_name] = eof_olr
        multiwave_eof_u[wave_name] = eof_u
        multiwave_eof_v[wave_name] = eof_v
        multiwave_eof_explained[wave_name] = explained
        if wave_name in CASE07_WAVES:
            eof_figure, _ = plot_eof_modes_with_wind(
                eof_olr,
                eof_u,
                eof_v,
                explained,
                modes=(1, 2),
                wave_name=wave_name,
                quiver_stride=CASE07_WAVE_QUIVER_STRIDES.get(wave_name, 4),
                quiver_scale=CASE07_WAVE_QUIVER_SCALES.get(
                    wave_name,
                    WAVE_QUIVER_SCALES.get(wave_name, 55.0),
                ),
                integer_colorbar=True,
                field_label="PC-regressed OLR (W m$^{-2}$)",
                save_path=wave_root / f"{wave_name}_eof_modes.png",
            )
            plt.close(eof_figure)
        save_dataset(
            xr.Dataset(
                {
                    "eofs": eofs,
                    "pcs": pcs,
                    "olr_regressed": eof_olr,
                    "u850_regressed": eof_u,
                    "v850_regressed": eof_v,
                    "eof_input_olr": eof_input,
                    "explained_variance": xr.DataArray(
                        explained[: eofs.sizes["mode"]],
                        dims=("mode",),
                        coords={"mode": eofs["mode"]},
                    ),
                }
            ),
            wave_root / f"{wave_name}_eof_modes.nc",
        )

        monthly_projection = {
            name: wave_longitude_projection(field, wave_name=wave_name, lat_band=lat_band)
            for name, field in filtered.items()
        }
        olr_line_clim, olr_line_pvalue = compute_longitude_mean_monthly_rms_climatology_and_significance(
            monthly_projection["olr"]
        )
        u_line_clim, u_line_pvalue = compute_longitude_mean_monthly_rms_climatology_and_significance(
            monthly_projection["u850"]
        )
        v_line_clim, v_line_pvalue = compute_longitude_mean_monthly_rms_climatology_and_significance(
            monthly_projection["v850"]
        )
        olr_monthly_rms, olr_monthly_pvalue = compute_monthly_climatology_and_significance(monthly_projection["olr"])
        u_monthly_rms = compute_monthly_rms(monthly_projection["u850"])
        v_monthly_rms = compute_monthly_rms(monthly_projection["v850"])
        monthly_cycle_data = {
            "OLR": olr_line_clim.values,
            "U850": u_line_clim.values,
            "V850": v_line_clim.values,
        }
        multiwave_monthly_cycles[wave_name] = monthly_cycle_data
        multiwave_monthly_significance[wave_name] = {
            "OLR": olr_line_pvalue,
            "U850": u_line_pvalue,
            "V850": v_line_pvalue,
        }
        multiwave_monthly_olr[wave_name] = olr_monthly_rms
        multiwave_monthly_olr_significance[wave_name] = olr_monthly_pvalue
        cycle_figure, _ = plot_monthly_cycle(
            np.arange(1, 13),
            monthly_cycle_data,
            title=f"{wave_name.upper()} projected monthly RMS cycle",
            ylabel="Projected RMS amplitude",
            save_path=wave_root / f"{wave_name}_monthly_cycle.png",
        )
        plt.close(cycle_figure)

        month_lon_figure, _ = plot_monthly_longitude_heatmap(
            olr_monthly_rms,
            title=f"{wave_name.upper()} OLR seasonal evolution",
            colorbar_label="Projected monthly RMS",
            save_path=wave_root / f"{wave_name}_seasonal_longitude_olr.png",
        )
        plt.close(month_lon_figure)

        annual_rms_olr = compute_yearly_rms(monthly_projection["olr"]).mean("lon").rename("annual_rms")
        multiwave_annual_rms[wave_name] = annual_rms_olr
        trend_ds = linear_trend(annual_rms_olr, dim="year")
        multiwave_annual_trend_pvalue[wave_name] = float(trend_ds["pvalue"].item())

        save_dataset(
            xr.Dataset(
                {
                    "olr_monthly_rms": olr_monthly_rms,
                    "u850_monthly_rms": u_monthly_rms,
                    "v850_monthly_rms": v_monthly_rms,
                    "olr_monthly_pvalue": olr_monthly_pvalue,
                    "olr_line_monthly_pvalue": olr_line_pvalue,
                    "u850_line_monthly_pvalue": u_line_pvalue,
                    "v850_line_monthly_pvalue": v_line_pvalue,
                    "olr_annual_rms": annual_rms_olr,
                    "olr_annual_rms_trend_slope": trend_ds["slope"],
                    "olr_annual_rms_trend_intercept": trend_ds["intercept"],
                    "olr_annual_rms_trend_pvalue": trend_ds["pvalue"],
                    "olr_annual_rms_trend_stderr": trend_ds["stderr"],
                }
            ),
            wave_root / f"{wave_name}_monthly_rms.nc",
        )

        rows.append(
            {
                "wave": wave_name,
                "event_count": int(event_indices.size),
                "time_start": str(dataset.time.values[0]),
                "time_end": str(dataset.time.values[-1]),
                "reference_projection": reference.attrs.get("wave_projection", "equatorial_mean"),
                "expected_direction": WAVE_PROPAGATION_DIRECTIONS.get(wave_name, "unknown"),
                "note": event_note,
            }
        )

    summary = pd.DataFrame(rows)
    compare_spatial_waves = [wave for wave in CASE04_WAVES if wave in multiwave_std_fields]
    if compare_spatial_waves:
        case04_olr = fill_missing_with_time_mean(
            load_dataarray(
                olr_path,
                variable="olr",
                time_range=time_range,
                lat_range=CASE04_LAT_RANGE,
            )
        )
        case04_std_fields = {
            wave: filter_wave_signal(
                case04_olr,
                wave_name=wave,
                method="cckw",
                obs_per_day=1,
                n_harm=n_harm,
                n_workers=n_workers,
            ).std("time")
            for wave in compare_spatial_waves
        }
        save_dataset(
            xr.Dataset({f"{wave}_olr_std": case04_std_fields[wave] for wave in compare_spatial_waves}),
            output_root / "multiwave_filtered_olr_std.nc",
        )
        for group_name, group_waves in _group_comparison_waves(compare_spatial_waves):
            compare_figure, _ = plot_wave_spatial_comparison(
                [case04_std_fields[wave] for wave in group_waves],
                titles=[wave.upper() for wave in group_waves],
                colorbar_label="Filtered OLR STD",
                ncols=1,
                cmap="wave_std_red",
                colorbar_orientation="vertical",
                integer_colorbar=True,
                target_steps=16,
                save_path=output_root / f"multiwave_{group_name}_filtered_olr_std.png",
            )
            plt.close(compare_figure)

    case05_waves = [wave for wave in CASE05_WAVES if wave in wave_list]
    if case05_waves:
        case05_source_name = "OLR"
        case05_note_text = "Monthly mean ± 1σ\n20°S–20°N regional mean\nOLR variance fraction"
        case05_description = "Monthly OLR variance fraction summaries for Case 05."
        case05_raw_field = raw_olr_anomaly.sel(lat=slice(*CASE05_LAT_RANGE))
        case05_filtered_fields: dict[str, xr.DataArray] = {
            wave: multiwave_filtered_olr[wave].sel(lat=slice(*CASE05_LAT_RANGE))
            for wave in case05_waves
            if wave in multiwave_filtered_olr
        }

        case05_gpcp_path = Path(DEFAULT_LOCAL_PATHS["gpcp"])
        if case05_gpcp_path.exists():
            case05_precip = fill_missing_with_time_mean(
                load_dataarray(
                    case05_gpcp_path,
                    variable=CASE05_PRECIP_VARIABLE,
                    time_range=time_range,
                    lat_range=CASE05_LAT_RANGE,
                )
            )
            case05_raw_field = compute_anomaly(case05_precip, group="month")
            case05_filtered_fields = {
                wave: filter_wave_signal(
                    case05_raw_field,
                    wave_name=wave,
                    method="cckw",
                    obs_per_day=1,
                    n_harm=n_harm,
                    n_workers=n_workers,
                ).sel(lat=slice(*CASE05_LAT_RANGE))
                for wave in case05_waves
            }
            case05_source_name = "GPCP precipitation"
            case05_note_text = "Monthly mean ± 1σ\n20°S–20°N regional mean\nGPCP precipitation variance fraction"
            case05_description = "Monthly GPCP precipitation variance fraction summaries for Case 05."

        case05_waves = [wave for wave in case05_waves if wave in case05_filtered_fields]
        tropical_means: list[xr.DataArray] = []
        tropical_stds: list[xr.DataArray] = []
        regional_means: list[xr.DataArray] = []
        regional_stds: list[xr.DataArray] = []

        for wave in case05_waves:
            variance_fraction = compute_monthly_variance_fraction_samples(
                case05_raw_field,
                case05_filtered_fields[wave],
                lat_range=CASE05_LAT_RANGE,
            )
            tropical_mean, tropical_std = summarize_variance_fraction_cycle(
                variance_fraction,
            )
            tropical_means.append(tropical_mean.expand_dims(wave=[wave]))
            tropical_stds.append(tropical_std.expand_dims(wave=[wave]))

            wave_regional_means: list[xr.DataArray] = []
            wave_regional_stds: list[xr.DataArray] = []
            for region_name, lon_range in CASE05_REGION_RANGES:
                region_mean, region_std = summarize_variance_fraction_cycle(
                    compute_monthly_variance_fraction_samples(
                        case05_raw_field,
                        case05_filtered_fields[wave],
                        lat_range=CASE05_LAT_RANGE,
                        lon_range=lon_range,
                    ),
                )
                wave_regional_means.append(region_mean.expand_dims(region=[region_name]))
                wave_regional_stds.append(region_std.expand_dims(region=[region_name]))

            regional_means.append(xr.concat(wave_regional_means, dim="region").expand_dims(wave=[wave]))
            regional_stds.append(xr.concat(wave_regional_stds, dim="region").expand_dims(wave=[wave]))

        tropical_mean_da = xr.concat(tropical_means, dim="wave")
        tropical_std_da = xr.concat(tropical_stds, dim="wave")
        regional_mean_da = xr.concat(regional_means, dim="wave").transpose("region", "wave", "month")
        regional_std_da = xr.concat(regional_stds, dim="wave").transpose("region", "wave", "month")
        case05_summary = xr.Dataset(
            {
                "tropical_mean": tropical_mean_da,
                "tropical_std": tropical_std_da,
                "regional_mean": regional_mean_da,
                "regional_std": regional_std_da,
            }
        )
        case05_summary.attrs["description"] = case05_description
        case05_summary.attrs["source"] = case05_source_name
        save_dataset(case05_summary, output_root / "case05_variance_fraction_summary.nc")

        case05_cycle_figure, _ = plot_case05_seasonal_variance_cycles(
            tropical_mean_da,
            tropical_std_da,
            wave_order=case05_waves,
            ylabel="Variance contribution [%]",
            save_path=output_root / "case05_seasonal_variance_cycle.png",
        )
        plt.close(case05_cycle_figure)

        case05_regional_figure, _ = plot_case05_regional_variance_cycles(
            regional_mean_da,
            regional_std_da,
            region_order=[name for name, _ in CASE05_REGION_RANGES],
            wave_order=case05_waves,
            region_labels=CASE05_REGION_LABELS,
            ylabel="Variance contribution [%]",
            note_text=case05_note_text,
            save_path=output_root / "case05_regional_variance_cycle.png",
        )
        plt.close(case05_regional_figure)

    compare_summary_waves = [
        wave
        for wave in CASE09_WAVES
        if wave in multiwave_monthly_cycles and wave in multiwave_monthly_olr and wave in multiwave_annual_rms
    ]
    if compare_summary_waves:
        for group_name, group_waves in _group_comparison_waves(compare_summary_waves):
            monthly_compare_figure, _ = plot_wave_monthly_cycle_comparison(
                {wave: multiwave_monthly_cycles[wave] for wave in group_waves},
                wave_names=group_waves,
                ncols=_comparison_ncols(len(group_waves)),
                monthly_significance={wave: multiwave_monthly_significance[wave] for wave in group_waves},
                normalize_each_series=(group_name == "westward"),
                normalization="annual_mean",
                save_path=output_root / f"multiwave_{group_name}_monthly_cycle.png",
            )
            plt.close(monthly_compare_figure)

            seasonal_compare_figure, _ = plot_wave_monthly_longitude_comparison(
                [multiwave_monthly_olr[wave] for wave in group_waves],
                wave_names=group_waves,
                ncols=_comparison_ncols(len(group_waves)),
                significance_fields=[multiwave_monthly_olr_significance[wave] for wave in group_waves],
                save_path=output_root / f"multiwave_{group_name}_seasonal_longitude_olr.png",
            )
            plt.close(seasonal_compare_figure)

            annual_compare_figure, _ = plot_wave_annual_trend_comparison(
                {wave: multiwave_annual_rms[wave] for wave in group_waves},
                wave_names=group_waves,
                ncols=_comparison_ncols(len(group_waves)),
                trend_pvalues={wave: multiwave_annual_trend_pvalue[wave] for wave in group_waves},
                save_path=output_root / f"multiwave_{group_name}_annual_rms_trend.png",
            )
            plt.close(annual_compare_figure)
    summary.to_csv(output_root / "wave_atlas_summary.csv", index=False)
    return summary
