"""Dual-channel Kelvin-wave event identification.

Convection-based (OLR) and dynamics-based (wind / geopotential) Kelvin-wave
signals can disagree, most visibly over the Pacific where the convective
envelope sits in the ITCZ several degrees north of the equator while the
circulation response stays roughly symmetric about the equator.  A single
fixed equatorial-band average of filtered OLR therefore under-samples the
convective signal that belongs to the wave.

This module identifies events in two separate channels and then compares them:

``dynamical channel``
    Kelvin-band filtered ``u`` (and optionally ``z``) projected on the
    Kelvin meridional structure ``psi_0(y) = exp(-y^2 / 2 Ld^2)``.  Using the
    Matsuno variable ``q = u + phi / c`` the projection ``int q psi_0 dy``
    obeys ``q_t + c q_x = (heating term)`` exactly in linear shallow-water
    theory, because ``int psi_0 (d/dy - y/Ld^2) v dy = 0``.  It is therefore
    insensitive to meridional wind and to every other equatorial wave mode.

``convective channel``
    Kelvin-band filtered OLR sampled with a Gaussian weight centred on a
    time-varying *convective axis latitude* (where Kelvin-band OLR variance is
    largest), instead of on a fixed equatorial band.

Both channels are turned into discrete events by tracking eastward-moving
ridges in the (time, longitude) plane.  Dynamical events are then scored for
convective coupling by a lagged correlation along the fitted wave path, and
convective-only events (no matching dynamical wave) are flagged separately.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from tropical_wave_tools.filters import WaveFilter
from tropical_wave_tools.matsuno import EARTH_OMEGA, EARTH_RADIUS, GRAVITY
from tropical_wave_tools.preprocessing import extract_low_harmonics

METERS_PER_DEGREE = EARTH_RADIUS * np.pi / 180.0
SECONDS_PER_DAY = 86400.0

__all__ = [
    "KelvinEventCatalog",
    "KelvinEventConfig",
    "convective_axis_latitude",
    "convective_kelvin_index",
    "detect_propagating_ridges",
    "equatorial_symmetry_index",
    "event_composite",
    "event_count_sensitivity",
    "fixed_band_convective_index",
    "identify_kelvin_events",
    "kelvin_band_filter",
    "kelvin_dynamical_index",
    "kelvin_meridional_structure",
    "mirror_latitude",
    "project_onto_kelvin",
    "select_events_in_region",
    "symmetric_antisymmetric",
]


# --------------------------------------------------------------------------- #
# Configuration and result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KelvinEventConfig:
    """Parameters of the dual-channel Kelvin event identification."""

    # --- Kelvin-band filter (Wheeler-Kiladis style) -------------------------
    period_days: Tuple[float, float] = (3.0, 20.0)
    wavenumber: Tuple[int, int] = (2, 14)
    equivalent_depth: Tuple[float, float] = (8.0, 90.0)
    obs_per_day: int = 1
    remove_seasonal_cycle: bool = True
    n_harm: int = 3

    # --- dynamical channel ---------------------------------------------------
    projection_depth: float = 25.0
    """Equivalent depth (m) that sets the meridional e-folding scale and ``c``."""
    projection_lat_limit: float = 15.0
    projection_method: str = "auto"
    """``"q"`` uses (u + g z / c) / 2, ``"u"`` uses u only, ``"auto"`` picks ``q`` if ``z`` is given."""

    # --- convective channel --------------------------------------------------
    axis_lat_range: Tuple[float, float] = (-15.0, 20.0)
    axis_window_days: Optional[int] = 91
    axis_smooth_lon_deg: float = 20.0
    conv_width_deg: float = 5.0
    fixed_band: Tuple[float, float] = (-5.0, 5.0)

    # --- ridge tracking ------------------------------------------------------
    peak_sigma: float = 1.5
    continue_sigma: float = 0.75
    normalize: str = "global"
    min_lifetime_days: float = 4.0
    min_distance_deg: float = 40.0
    speed_range: Tuple[float, float] = (8.0, 30.0)
    expected_speed: float = 15.0
    max_gap_steps: int = 1
    dyn_min_amplitude: Optional[float] = None
    """Absolute peak amplitude (m s-1) required of a dynamical event, in addition to ``peak_sigma``."""
    conv_min_amplitude: Optional[float] = None
    """Absolute peak amplitude (``-OLR`` units) required of a convective event."""

    # --- coupling / matching -------------------------------------------------
    coupling_max_offset_deg: float = 30.0
    coupling_threshold: float = 0.5
    coupling_alpha: float = 0.05
    coupling_surrogates: int = 40
    """Number of circularly time-shifted copies of the convective index used as the null."""
    match_lon_tol_deg: float = 40.0
    match_overlap: float = 0.5
    symmetry_window_deg: float = 20.0
    symmetry_lat_max: float = 15.0


@dataclass
class KelvinEventCatalog:
    """Container returned by :func:`identify_kelvin_events`."""

    events: pd.DataFrame
    tracks: pd.DataFrame
    indices: xr.Dataset
    config: KelvinEventConfig = field(default_factory=KelvinEventConfig)

    def summary(self) -> pd.Series:
        """Return the number of events of each ``kind``."""
        if self.events.empty:
            return pd.Series(dtype=int, name="count")
        return self.events["kind"].value_counts().rename("count")

    def in_region(self, lon_range: Tuple[float, float], *, min_points: int = 3) -> "KelvinEventCatalog":
        """Keep events whose tracked ridge spends at least ``min_points`` steps inside ``lon_range``.

        Detection must run on a global longitude grid; this only restricts the
        *reporting* to a basin (e.g. ``(120, 280)`` for the Pacific).  Ranges
        that wrap through 0 deg, such as ``(300, 60)``, are supported.
        """
        events, tracks = select_events_in_region(self.events, self.tracks, lon_range, min_points=min_points)
        return KelvinEventCatalog(events=events, tracks=tracks, indices=self.indices, config=self.config)

    def in_window(self, start: "str | pd.Timestamp", end: "str | pd.Timestamp") -> "KelvinEventCatalog":
        """Keep events whose ``peak_time`` lies in ``[start, end]`` (use to drop padding/edge periods)."""
        if self.events.empty:
            return self
        peak = pd.to_datetime(self.events["peak_time"])
        keep = (peak >= pd.Timestamp(start)) & (peak <= pd.Timestamp(end))
        events = self.events[keep].reset_index(drop=True)
        tracks = _tracks_of_events(events, self.tracks)
        return KelvinEventCatalog(events=events, tracks=tracks, indices=self.indices, config=self.config)

    def save(self, directory: "str | Path") -> Path:
        """Write ``events.csv``, ``tracks.csv`` and (if possible) ``indices.nc``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.events.to_csv(out / "events.csv", index=False)
        self.tracks.to_csv(out / "tracks.csv", index=False)
        # Writing NetCDF needs an optional backend; the CSV outputs are the primary product.
        with contextlib.suppress(ImportError, ValueError, RuntimeError):
            self.indices.to_netcdf(out / "indices.nc")
        return out


# --------------------------------------------------------------------------- #
# Input handling and filtering
# --------------------------------------------------------------------------- #
_DIM_ALIASES = {"latitude": "lat", "longitude": "lon", "Time": "time"}


def _standardize(da: xr.DataArray, name: str = "field") -> xr.DataArray:
    """Return ``da`` as float ``(time, lat, lon)`` with ascending lat and lon in [0, 360)."""
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"`{name}` must be an xarray.DataArray.")
    rename = {old: new for old, new in _DIM_ALIASES.items() if old in da.dims}
    if rename:
        da = da.rename(rename)
    for dim in ("time", "lat", "lon"):
        if dim not in da.dims:
            raise ValueError(f"`{name}` must have dimensions (time, lat, lon); missing '{dim}'.")
    if set(da.dims) != {"time", "lat", "lon"}:
        raise ValueError(f"`{name}` must be exactly 3-D (time, lat, lon); got {da.dims}.")
    da = da.transpose("time", "lat", "lon").astype(float)
    da = da.assign_coords(lon=np.mod(da.lon.values, 360.0)).sortby(["time", "lat", "lon"])
    _, keep = np.unique(da.lon.values, return_index=True)
    return da.isel(lon=np.sort(keep))


def _require_global_longitude(lon: np.ndarray) -> float:
    """Return the longitude spacing, requiring a (nearly) global periodic grid."""
    if lon.size < 8:
        raise ValueError("At least 8 longitudes are required.")
    dlon = 360.0 / lon.size
    if not np.allclose(np.diff(lon), dlon, rtol=0.0, atol=1.0e-3 * dlon) or (lon[-1] - lon[0]) < 360.0 - 1.5 * dlon:
        raise ValueError(
            "Event identification needs a global, uniformly spaced longitude grid "
            "(Kelvin waves are tracked on a periodic domain)."
        )
    return dlon


def _remove_seasonal_cycle(da: xr.DataArray, n_harm: int) -> xr.DataArray:
    climatology = da.groupby("time.dayofyear").mean(dim="time")
    fit = extract_low_harmonics(climatology, n_harm=n_harm)
    return (da.groupby("time.dayofyear") - fit).transpose("time", "lat", "lon")


def kelvin_band_filter(
    data: xr.DataArray,
    *,
    period_days: Tuple[float, float] = (3.0, 20.0),
    wavenumber: Tuple[int, int] = (2, 14),
    equivalent_depth: Tuple[float, float] = (8.0, 90.0),
    obs_per_day: int = 1,
    remove_seasonal_cycle: bool = True,
    n_harm: int = 3,
) -> xr.DataArray:
    """Wheeler-Kiladis Kelvin-band filter of a ``(time, lat, lon)`` field.

    The repository's :class:`~tropical_wave_tools.filters.WaveFilter` core is
    reused so the result is identical to ``filter_wave_signal(..., "kelvin")``.
    With ``remove_seasonal_cycle=False`` only the time mean is removed, which
    is what short synthetic records need.
    """
    da = _standardize(data, "data")
    _require_global_longitude(da.lon.values)
    anomaly = _remove_seasonal_cycle(da, n_harm) if remove_seasonal_cycle else da - da.mean("time")

    values = np.nan_to_num(anomaly.values.astype(float))
    lon = da.lon.values
    kf = WaveFilter()
    out = np.empty_like(values)
    for j in range(values.shape[1]):
        out[:, j, :] = kf._kf_filter(
            values[:, j, :],
            lon=lon,
            obs_per_day=obs_per_day,
            period_days=period_days,
            wavenumber=wavenumber,
            equivalent_depth=equivalent_depth,
            wave_name="kelvin",
        )
    result = xr.DataArray(out, dims=da.dims, coords=da.coords, name=data.name)
    result.attrs.update(
        {
            "long_name": "Kelvin-band filtered anomaly",
            "units": data.attrs.get("units", ""),
            "period_days": period_days,
            "wavenumber": wavenumber,
            "equivalent_depth": equivalent_depth,
        }
    )
    return result


# --------------------------------------------------------------------------- #
# Dynamical channel: Kelvin meridional projection
# --------------------------------------------------------------------------- #
def kelvin_meridional_structure(
    lat: np.ndarray,
    *,
    equivalent_depth: float = 25.0,
    ref_lat: float = 0.0,
) -> np.ndarray:
    """Return ``psi_0(y) = exp(-y^2 / 2 Ld^2)`` with ``Ld = sqrt(c / beta)``.

    ``ref_lat`` shifts the axis, which is useful to ask how much of an
    off-equatorial heating or OLR pattern projects on the Kelvin mode.
    """
    lat = np.asarray(lat, dtype=float)
    c = np.sqrt(GRAVITY * equivalent_depth)
    beta = 2.0 * EARTH_OMEGA / EARTH_RADIUS
    deformation_radius = np.sqrt(c / beta)
    y = METERS_PER_DEGREE * (lat - ref_lat)
    return np.exp(-0.5 * (y / deformation_radius) ** 2)


def project_onto_kelvin(
    field_: xr.DataArray,
    *,
    equivalent_depth: float = 25.0,
    ref_lat: float = 0.0,
    lat_limit: Optional[float] = 15.0,
    min_coverage: float = 0.6,
) -> xr.DataArray:
    """Least-squares Kelvin amplitude ``A`` such that ``field ~ A psi_0(y)``.

    The discrete normalisation ``sum(F psi w) / sum(psi^2 w)`` recovers ``A``
    exactly for ``F = A psi_0`` even when the latitude band truncates the
    Gaussian.  Points with missing data are excluded and the result is NaN
    where less than ``min_coverage`` of the weight is available.
    """
    if "lat" not in field_.dims:
        raise ValueError("`field_` needs a 'lat' dimension.")
    f = field_.sortby("lat")
    if lat_limit is not None:
        f = f.sel(lat=slice(ref_lat - lat_limit, ref_lat + lat_limit))
    lat = f.lat.values
    if lat.size < 3:
        raise ValueError("At least 3 latitudes are needed for the Kelvin projection.")
    psi = xr.DataArray(
        kelvin_meridional_structure(lat, equivalent_depth=equivalent_depth, ref_lat=ref_lat),
        dims="lat",
        coords={"lat": lat},
    )
    weight = xr.DataArray(np.gradient(lat), dims="lat", coords={"lat": lat})
    valid = f.notnull()
    numerator = (f.fillna(0.0) * psi * weight).sum("lat")
    denominator = (valid * psi**2 * weight).sum("lat")
    full = float((psi**2 * weight).sum())
    amplitude = numerator / denominator.where(denominator > 0.0)
    amplitude = amplitude.where(denominator >= min_coverage * full)
    amplitude.name = "kelvin_projection"
    return amplitude


def kelvin_dynamical_index(
    u: xr.DataArray,
    z: Optional[xr.DataArray] = None,
    *,
    equivalent_depth: float = 25.0,
    method: str = "auto",
    lat_limit: float = 15.0,
) -> xr.DataArray:
    """Project (Kelvin-band) ``u`` and optionally ``z`` on the Kelvin mode.

    Parameters
    ----------
    u:
        Zonal wind anomaly, m s-1, ``(time, lat, lon)``.
    z:
        Geopotential-height anomaly, m, same grid as ``u``.  With ``z`` the
        Matsuno combination ``q = (u + g z / c) / 2`` is projected, which is
        exact for the Kelvin mode; without it only ``u`` is projected.
    method:
        ``"auto"``, ``"q"`` or ``"u"``.

    Returns
    -------
    xarray.DataArray
        ``(time, lon)`` Kelvin amplitude in m s-1; positive values are
        westerly crests.
    """
    if method not in {"auto", "q", "u"}:
        raise ValueError("`method` must be 'auto', 'q' or 'u'.")
    use_q = method == "q" or (method == "auto" and z is not None)
    if use_q:
        if z is None:
            raise ValueError("method='q' requires `z`.")
        c = np.sqrt(GRAVITY * equivalent_depth)
        combined = 0.5 * (u + GRAVITY * z / c)
    else:
        combined = u
    index = project_onto_kelvin(combined, equivalent_depth=equivalent_depth, lat_limit=lat_limit)
    index.name = "kelvin_dynamical"
    index.attrs.update({"units": "m s-1", "method": "q" if use_q else "u", "equivalent_depth": equivalent_depth})
    return index


# --------------------------------------------------------------------------- #
# Equatorial symmetry diagnostics
# --------------------------------------------------------------------------- #
def mirror_latitude(field_: xr.DataArray) -> xr.DataArray:
    """Return ``field_(-lat)`` on the original latitude coordinates (NaN outside the grid)."""
    f = field_.sortby("lat")
    mirrored = f.interp(lat=-f.lat.values)
    return mirrored.assign_coords(lat=f.lat.values)


def symmetric_antisymmetric(
    field_: xr.DataArray,
    *,
    lat_max: Optional[float] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Split a field into equatorially symmetric and antisymmetric parts."""
    f = field_.sortby("lat")
    if lat_max is not None:
        f = f.sel(lat=slice(-lat_max, lat_max))
    m = mirror_latitude(f)
    return 0.5 * (f + m), 0.5 * (f - m)


def equatorial_symmetry_index(
    field_: xr.DataArray,
    *,
    lat_max: float = 15.0,
    dims: Sequence[str] = ("lat",),
) -> xr.DataArray:
    """Return ``(E_sym - E_asym) / (E_sym + E_asym)`` in [-1, 1].

    ``+1`` means purely symmetric (Kelvin-like ``u``), ``-1`` purely
    antisymmetric.  Energies are means of the squared parts over ``dims``
    (add ``"time"``/``"lon"`` to aggregate further).
    """
    sym, asym = symmetric_antisymmetric(field_, lat_max=lat_max)
    e_sym = (sym**2).mean(list(dims), skipna=True)
    e_asym = (asym**2).mean(list(dims), skipna=True)
    total = e_sym + e_asym
    return ((e_sym - e_asym) / total.where(total > 0.0)).rename("equatorial_symmetry_index")


# --------------------------------------------------------------------------- #
# Convective channel
# --------------------------------------------------------------------------- #
def convective_axis_latitude(
    olr_kelvin: xr.DataArray,
    *,
    lat_range: Tuple[float, float] = (-15.0, 20.0),
    window_days: Optional[int] = 91,
    smooth_lon_deg: float = 20.0,
    obs_per_day: int = 1,
) -> xr.DataArray:
    """Latitude of maximum Kelvin-band OLR variance, ``(time, lon)``.

    The variance is a centred running mean over ``window_days`` (or the whole
    record when ``None``), smoothed zonally, and the maximum is located to
    sub-grid accuracy with a parabola.  The resulting axis is median-filtered
    in longitude so isolated jumps between e.g. the ITCZ and the SPCZ do not
    dominate; restrict ``lat_range`` (e.g. ``(0, 15)``) to force one branch.
    """
    x = _standardize(olr_kelvin, "olr_kelvin").sel(lat=slice(*lat_range))
    if x.sizes["lat"] < 3:
        raise ValueError("`lat_range` must contain at least 3 latitudes.")
    dlon = _require_global_longitude(x.lon.values)

    power = x**2
    if window_days:
        window = max(round(window_days * obs_per_day), 1)
        power = power.rolling(time=window, center=True, min_periods=max(window // 4, 1)).mean()
    else:
        power = power.mean("time").expand_dims(time=x.time).transpose("time", "lat", "lon")

    p = power.values
    n_smooth = max(round(smooth_lon_deg / dlon), 1)
    if n_smooth > 1:
        p = ndimage.uniform_filter1d(p, size=n_smooth, axis=2, mode="wrap")
    p = np.nan_to_num(p, nan=-np.inf)

    lat = x.lat.values
    i = np.argmax(p, axis=1)  # (time, lon)
    nlat = lat.size
    lo = np.clip(i - 1, 0, nlat - 1)
    hi = np.clip(i + 1, 0, nlat - 1)
    pl = np.take_along_axis(p, lo[:, None, :], axis=1)[:, 0, :]
    pc = np.take_along_axis(p, i[:, None, :], axis=1)[:, 0, :]
    ph = np.take_along_axis(p, hi[:, None, :], axis=1)[:, 0, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = pl - 2.0 * pc + ph
        offset = np.where((denom < 0.0) & np.isfinite(denom), 0.5 * (pl - ph) / denom, 0.0)
    offset = np.clip(offset, -1.0, 1.0)
    axis = lat[i] + offset * float(np.mean(np.diff(lat)))
    axis = np.where(np.isfinite(pc), axis, np.nan)

    n_med = max(round(smooth_lon_deg / dlon), 1)
    if n_med > 1:
        filled = np.where(np.isfinite(axis), axis, np.nanmedian(axis) if np.isfinite(axis).any() else 0.0)
        axis = ndimage.median_filter(filled, size=(1, n_med), mode="wrap")
    axis = np.clip(axis, lat[0], lat[-1])
    return xr.DataArray(
        axis,
        dims=("time", "lon"),
        coords={"time": x.time, "lon": x.lon},
        name="convective_axis_lat",
        attrs={"units": "degrees_north", "long_name": "Latitude of maximum Kelvin-band OLR variance"},
    )


def convective_kelvin_index(
    olr_kelvin: xr.DataArray,
    axis_lat: xr.DataArray,
    *,
    width_deg: float = 5.0,
) -> xr.DataArray:
    """Gaussian-weighted Kelvin-band convection index along a moving axis.

    The sign is flipped (``-OLR``) so positive means enhanced convection.
    """
    x = _standardize(olr_kelvin, "olr_kelvin")
    axis = axis_lat.transpose("time", "lon").reindex(time=x.time, lon=x.lon)
    lat = x.lat.values[None, :, None]
    weight = np.exp(-0.5 * ((lat - axis.values[:, None, :]) / width_deg) ** 2)
    conv = -x.values
    valid = np.isfinite(conv)
    weight = np.where(valid, weight, 0.0)
    total = weight.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        index = (np.where(valid, conv, 0.0) * weight).sum(axis=1) / total
    index = np.where(total > 1.0e-6, index, np.nan)
    return xr.DataArray(
        index,
        dims=("time", "lon"),
        coords={"time": x.time, "lon": x.lon},
        name="kelvin_convective_axis",
        attrs={"units": olr_kelvin.attrs.get("units", ""), "long_name": "-OLR_K along convective axis"},
    )


def fixed_band_convective_index(
    olr_kelvin: xr.DataArray,
    *,
    band: Tuple[float, float] = (-5.0, 5.0),
) -> xr.DataArray:
    """Conventional fixed equatorial-band ``-OLR_K`` index, for comparison."""
    x = _standardize(olr_kelvin, "olr_kelvin")
    out = -x.sel(lat=slice(*band)).mean("lat")
    out.name = "kelvin_convective_fixed_band"
    out.attrs.update({"units": olr_kelvin.attrs.get("units", ""), "long_name": f"-OLR_K, {band[0]:g} to {band[1]:g} lat mean"})
    return out


# --------------------------------------------------------------------------- #
# Ridge tracking
# --------------------------------------------------------------------------- #
def _wrap180(value):
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def _ridge_candidates(row: np.ndarray, threshold: float, lon: np.ndarray, dlon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Local maxima of a periodic longitude series with parabolic refinement."""
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(row) & (row > threshold) & (row >= left) & (row > right)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return idx, np.empty(0)
    v_left, v_mid, v_right = left[idx], row[idx], right[idx]
    denom = v_left - 2.0 * v_mid + v_right
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where((denom < 0.0) & np.isfinite(denom), 0.5 * (v_left - v_right) / denom, 0.0)
    off = np.clip(off, -1.0, 1.0)
    return idx, lon[idx] + off * dlon


def _time_in_days(time: np.ndarray) -> np.ndarray:
    t = pd.to_datetime(time)
    return np.asarray((t - t[0]) / pd.Timedelta(days=1), dtype=float)


def detect_propagating_ridges(
    index: xr.DataArray,
    *,
    peak_sigma: float = 1.0,
    continue_sigma: float = 0.5,
    normalize: str = "global",
    min_lifetime_days: float = 4.0,
    min_distance_deg: float = 40.0,
    speed_range: Tuple[float, float] = (8.0, 30.0),
    expected_speed: float = 15.0,
    max_gap_steps: int = 1,
    min_amplitude: Optional[float] = None,
    prefix: str = "E",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Track eastward-moving positive ridges of a ``(time, lon)`` index.

    A ridge is a local maximum in longitude.  It is linked from one time step
    to the next when it moved eastward by no more than ``1.3`` times the
    fastest allowed phase speed, choosing the pairing closest to the track's
    own recent velocity (one-to-one, greedy).  Tracks use hysteresis: they may
    continue while above ``continue_sigma`` but must exceed ``peak_sigma`` at
    least once.  Finished tracks are kept when they live at least
    ``min_lifetime_days``, travel at least ``min_distance_deg`` and their
    linear-fit phase speed lies in ``speed_range`` (m s-1).  ``min_amplitude``
    adds an absolute peak-amplitude requirement in the units of ``index``;
    use it because a purely relative (sigma) threshold still accepts the
    largest ridges of a record that contains no real waves.

    Returns
    -------
    events, tracks:
        ``events`` has one row per event; ``tracks`` has one row per
        (event, time step) with the raw ridge longitude.
    """
    if normalize not in {"global", "lon"}:
        raise ValueError("`normalize` must be 'global' or 'lon'.")
    idx = index.transpose("time", "lon")
    lon = np.mod(idx.lon.values.astype(float), 360.0)
    order = np.argsort(lon)
    idx = idx.isel(lon=order)
    lon = lon[order]
    dlon = _require_global_longitude(lon)
    values = idx.values.astype(float)
    nt, nlon = values.shape
    if nt < 3:
        return _empty_events(), _empty_tracks()

    tdays = _time_in_days(idx.time.values)
    dt = float(np.median(np.diff(tdays)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Time axis must be strictly increasing.")

    if normalize == "global":
        sigma = np.nanstd(values)
        sigma = sigma if sigma > 0 else 1.0
        z = values / sigma
    else:
        sigma = np.nanstd(values, axis=0)
        sigma = np.where(sigma > 0, sigma, 1.0)
        z = values / sigma[None, :]

    def deg_per_step(speed_ms: float) -> float:
        return speed_ms * SECONDS_PER_DAY * dt / METERS_PER_DEGREE

    d_max = 1.3 * deg_per_step(speed_range[1])
    d_min = -dlon
    d_expected = deg_per_step(expected_speed)

    active: List[Dict[str, list]] = []
    finished: List[Dict[str, list]] = []

    for it in range(nt):
        cand_idx, cand_lon = _ridge_candidates(z[it], continue_sigma, lon, dlon)
        pairs = []
        for a, tr in enumerate(active):
            gap = it - tr["t"][-1]
            if gap > max_gap_steps + 1:
                continue
            velocity = tr["vel"] if tr["vel"] is not None else d_expected
            for c in range(cand_idx.size):
                d = float(_wrap180(cand_lon[c] - tr["lon"][-1]))
                if d_min * gap <= d <= d_max * gap:
                    pairs.append((abs(d - velocity * gap), a, c, d))
        pairs.sort(key=lambda item: item[0])
        used_tracks, used_cands = set(), set()
        for _, a, c, d in pairs:
            if a in used_tracks or c in used_cands:
                continue
            used_tracks.add(a)
            used_cands.add(c)
            tr = active[a]
            tr["t"].append(it)
            tr["lon"].append(tr["lon"][-1] + d)
            tr["amp"].append(float(z[it, cand_idx[c]]))
            span = max(len(tr["t"]) - 4, 0)
            dts = tr["t"][-1] - tr["t"][span]
            tr["vel"] = (tr["lon"][-1] - tr["lon"][span]) / dts if dts > 0 else None
        for c in range(cand_idx.size):
            if c not in used_cands:
                active.append({"t": [it], "lon": [float(cand_lon[c])], "amp": [float(z[it, cand_idx[c]])], "vel": None})
        still_active = []
        for tr in active:
            if it - tr["t"][-1] > max_gap_steps:
                finished.append(tr)
            else:
                still_active.append(tr)
        active = still_active
    finished.extend(active)

    rows, track_rows = [], []
    raw = []
    for tr in finished:
        t_idx = np.asarray(tr["t"])
        if t_idx.size < 3 or max(tr["amp"]) < peak_sigma:
            continue
        td = tdays[t_idx]
        lon_u = np.asarray(tr["lon"])
        lifetime = td[-1] - td[0] + dt
        distance = lon_u[-1] - lon_u[0]
        slope, intercept = np.polyfit(td - td[0], lon_u, 1)
        speed = slope * METERS_PER_DEGREE / SECONDS_PER_DAY
        if lifetime < min_lifetime_days or distance < min_distance_deg:
            continue
        if not (speed_range[0] <= speed <= speed_range[1]):
            continue
        k_peak = int(np.argmax(tr["amp"]))
        j_peak = int(np.rint((lon_u[k_peak] % 360.0 - lon[0]) / dlon)) % nlon
        if min_amplitude is not None and not values[t_idx[k_peak], j_peak] >= min_amplitude:
            continue
        raw.append((t_idx, lon_u, np.asarray(tr["amp"]), slope, intercept, speed, lifetime, distance))

    raw.sort(key=lambda item: item[0][0])
    times = pd.to_datetime(idx.time.values)
    for n, (t_idx, lon_u, amp, slope, intercept, speed, lifetime, distance) in enumerate(raw, start=1):
        event_id = f"{prefix}{n:04d}"
        k = int(np.argmax(amp))
        j_peak = int(np.rint((lon_u[k] % 360.0 - lon[0]) / dlon)) % nlon
        raw_amp = float(values[t_idx[k], j_peak])
        rows.append(
            {
                "event_id": event_id,
                "start_time": times[t_idx[0]],
                "end_time": times[t_idx[-1]],
                "peak_time": times[t_idx[k]],
                "lifetime_days": float(lifetime),
                "lon_start": float(lon_u[0] % 360.0),
                "lon_end": float(lon_u[-1] % 360.0),
                "lon_peak": float(lon_u[k] % 360.0),
                "distance_deg": float(distance),
                "speed_ms": float(speed),
                "fit_slope_deg_per_day": float(slope),
                "fit_lon0_deg": float(intercept),
                "peak_sigma": float(amp[k]),
                "peak_amp": raw_amp,
                "mean_sigma": float(np.mean(amp)),
            }
        )
        for ti, lu, a in zip(t_idx, lon_u, amp):
            track_rows.append(
                {
                    "event_id": event_id,
                    "time": times[ti],
                    "lon": float(lu % 360.0),
                    "lon_unwrapped": float(lu),
                    "amp_sigma": float(a),
                }
            )
    events = pd.DataFrame(rows) if rows else _empty_events()
    tracks = pd.DataFrame(track_rows) if track_rows else _empty_tracks()
    return events, tracks


_EVENT_COLUMNS = [
    "event_id",
    "start_time",
    "end_time",
    "peak_time",
    "lifetime_days",
    "lon_start",
    "lon_end",
    "lon_peak",
    "distance_deg",
    "speed_ms",
    "fit_slope_deg_per_day",
    "fit_lon0_deg",
    "peak_sigma",
    "peak_amp",
    "mean_sigma",
]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=_EVENT_COLUMNS)


def _empty_tracks() -> pd.DataFrame:
    return pd.DataFrame(columns=["event_id", "time", "lon", "lon_unwrapped", "amp_sigma"])


def _tracks_of_events(events: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Tracks of the given events plus the convective ridges matched to them."""
    ids = set(events["event_id"])
    if "matched_event" in events.columns:
        for matched in events["matched_event"].fillna(""):
            ids.update(m for m in str(matched).split(",") if m)
    return tracks[tracks["event_id"].isin(ids)].reset_index(drop=True)


def select_events_in_region(
    events: pd.DataFrame,
    tracks: pd.DataFrame,
    lon_range: Tuple[float, float],
    *,
    min_points: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict events to those with ``>= min_points`` track points inside a longitude window."""
    if events.empty or tracks.empty:
        return events.iloc[0:0].copy(), tracks.iloc[0:0].copy()
    low, high = (float(lon_range[0]) % 360.0, float(lon_range[1]) % 360.0)
    lon = np.mod(tracks["lon"].to_numpy(dtype=float), 360.0)
    inside = (lon >= low) & (lon <= high) if low <= high else (lon >= low) | (lon <= high)
    counts = tracks[inside].groupby("event_id").size()
    keep = set(counts[counts >= min_points].index)
    kept_events = events[events["event_id"].isin(keep)].reset_index(drop=True)
    return kept_events, _tracks_of_events(kept_events, tracks)


def event_count_sensitivity(
    index: xr.DataArray,
    peak_sigmas: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
    *,
    continue_ratio: float = 0.5,
    **kwargs,
) -> pd.DataFrame:
    """Number of tracked events as a function of the peak threshold.

    The amplitude threshold is the main tuning knob of the tracker.  Plot or
    inspect this table before fixing ``peak_sigma``: a plateau (counts that
    barely change over a range of thresholds) marks a robust choice.
    """
    rows = []
    for sigma in peak_sigmas:
        events, _ = detect_propagating_ridges(
            index, peak_sigma=sigma, continue_sigma=continue_ratio * sigma, **kwargs
        )
        rows.append(
            {
                "peak_sigma": sigma,
                "n_events": len(events),
                "median_speed_ms": float(events["speed_ms"].median()) if len(events) else np.nan,
                "median_lifetime_days": float(events["lifetime_days"].median()) if len(events) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Coupling, matching and event diagnostics
# --------------------------------------------------------------------------- #
def _fitted_path(event: pd.Series, times: pd.DatetimeIndex, lon0: float, dlon: float, nlon: int):
    """Time indices and nearest-grid longitude indices along the fitted wave path."""
    t0 = int(times.get_indexer([event["start_time"]], method="nearest")[0])
    t1 = int(times.get_indexer([event["end_time"]], method="nearest")[0])
    t_idx = np.arange(t0, t1 + 1)
    days = (times[t_idx] - times[t0]) / pd.Timedelta(days=1)
    # Linear fit of the tracked ridge longitude: value at start_time plus slope.
    path = float(event["fit_lon0_deg"]) + float(event["fit_slope_deg_per_day"]) * np.asarray(days, dtype=float)
    j0 = np.rint((np.mod(path, 360.0) - lon0) / dlon).astype(int) % nlon
    return t_idx, j0, path


def _lagged_correlation(x: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation of ``x`` with each column of ``y_matrix``."""
    xm = x - x.mean()
    ym = y_matrix - y_matrix.mean(axis=0, keepdims=True)
    num = (xm[:, None] * ym).sum(axis=0)
    den = np.sqrt((xm**2).sum() * (ym**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _coupling(
    event: pd.Series,
    k_values: np.ndarray,
    c_values: np.ndarray,
    times: pd.DatetimeIndex,
    lon0: float,
    dlon: float,
    max_offset_deg: float,
    *,
    n_surrogates: int = 0,
    obs_per_day: int = 1,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Best lagged correlation between the dynamical ridge and a convective index.

    The maximum over longitude offsets is biased upward, particularly for the
    short paths of real events.  When ``n_surrogates > 0`` the same statistic
    is recomputed with the convective index circularly shifted in time by
    15 days and more (phase relation destroyed, spatial structure kept), and
    ``p = (1 + #{surrogate >= observed}) / (1 + N)`` is returned.

    Returns ``(r_max, offset_deg, p_value, t_idx, j0)``.
    """
    nt, nlon = c_values.shape
    t_idx, j0, _ = _fitted_path(event, times, lon0, dlon, nlon)
    m = round(max_offset_deg / dlon)
    shifts = np.arange(-m, m + 1)
    cols = (j0[:, None] + shifts[None, :]) % nlon
    x = k_values[t_idx, j0]

    def best(rows: np.ndarray) -> Tuple[float, int]:
        y = c_values[rows[:, None], cols]
        if t_idx.size < 4 or not (np.isfinite(x).all() and np.isfinite(y).all()):
            return np.nan, 0
        r = _lagged_correlation(x, y)
        if not np.isfinite(r).any():
            return np.nan, 0
        k = int(np.nanargmax(r))
        return float(r[k]), k

    r_best, k_best = best(t_idx)
    if not np.isfinite(r_best):
        return np.nan, np.nan, np.nan, t_idx, j0

    p_value = np.nan
    if n_surrogates > 0:
        base = max(round(15 * obs_per_day), 1)
        step = max(round(5 * obs_per_day), 1)
        offsets = []
        for i in range(n_surrogates // 2 + 1):
            offsets.extend([base + i * step, -(base + i * step)])
        null = [best((t_idx + off) % nt)[0] for off in offsets[:n_surrogates]]
        null = np.asarray([v for v in null if np.isfinite(v)])
        if null.size:
            p_value = float((1 + np.sum(null >= r_best)) / (1 + null.size))
    return r_best, float(shifts[k_best] * dlon), p_value, t_idx, j0


def _structure_diagnostics(
    u_k: Optional[np.ndarray],
    v_k: Optional[np.ndarray],
    lat: np.ndarray,
    t_idx: np.ndarray,
    j0: np.ndarray,
    dlon: float,
    window_deg: float,
    lat_max: float,
) -> Tuple[float, float]:
    """Equatorial symmetry of ``u`` and ``rms(v)/rms(u)`` around the wave path."""
    if u_k is None:
        return np.nan, np.nan
    sel = np.flatnonzero(np.abs(lat) <= lat_max)
    if sel.size < 3 or not np.allclose(lat[sel], -lat[sel][::-1], atol=1.0e-3):
        return np.nan, np.nan
    nlon = u_k.shape[2]
    w = round(window_deg / dlon)
    cols = (j0[:, None] + np.arange(-w, w + 1)[None, :]) % nlon
    u_sub = u_k[t_idx[:, None, None], sel[None, None, :], cols[:, :, None]]
    u_sym = 0.5 * (u_sub + u_sub[..., ::-1])
    u_asym = 0.5 * (u_sub - u_sub[..., ::-1])
    e_sym, e_asym = np.mean(u_sym**2), np.mean(u_asym**2)
    sym_index = (e_sym - e_asym) / (e_sym + e_asym) if (e_sym + e_asym) > 0 else np.nan
    ratio = np.nan
    if v_k is not None:
        v_sub = v_k[t_idx[:, None, None], sel[None, None, :], cols[:, :, None]]
        denom = np.sqrt(np.mean(u_sub**2))
        ratio = float(np.sqrt(np.mean(v_sub**2)) / denom) if denom > 0 else np.nan
    return float(sym_index), ratio


def _match_convective_to_dynamical(
    conv_events: pd.DataFrame,
    conv_tracks: pd.DataFrame,
    dyn_tracks: pd.DataFrame,
    *,
    lon_tol: float,
    overlap: float,
) -> Dict[str, str]:
    """Map convective event ids to the dynamical event they overlap in space-time."""
    matches: Dict[str, str] = {}
    if conv_events.empty or dyn_tracks.empty:
        return matches
    dyn_groups = {k: g.set_index("time")["lon"] for k, g in dyn_tracks.groupby("event_id")}
    for cid, group in conv_tracks.groupby("event_id"):
        c_series = group.set_index("time")["lon"]
        best_id, best_score = None, np.inf
        for did, d_series in dyn_groups.items():
            common = c_series.index.intersection(d_series.index)
            if len(common) / len(c_series) < overlap:
                continue
            dist = np.abs(_wrap180(c_series.loc[common].values - d_series.loc[common].values)).mean()
            if dist <= lon_tol and dist < best_score:
                best_id, best_score = did, dist
        if best_id is not None:
            matches[cid] = best_id
    return matches


# --------------------------------------------------------------------------- #
# Public workflow
# --------------------------------------------------------------------------- #
def identify_kelvin_events(
    *,
    u: Optional[xr.DataArray] = None,
    z: Optional[xr.DataArray] = None,
    v: Optional[xr.DataArray] = None,
    olr: Optional[xr.DataArray] = None,
    config: Optional[KelvinEventConfig] = None,
    axis_lat: Optional[xr.DataArray] = None,
    prefiltered: bool = False,
) -> KelvinEventCatalog:
    """Identify Kelvin-wave events from dynamical and/or convective data.

    Parameters
    ----------
    u, z, v:
        Zonal wind (m s-1), geopotential height (m) and meridional wind
        (m s-1) anomalies/raw fields on one pressure level, ``(time, lat, lon)``.
        At least ``u`` or ``olr`` is required.  ``z`` enables the exact
        Matsuno projection; ``v`` only feeds the ``v_to_u_ratio`` diagnostic.
    olr:
        Outgoing longwave radiation (or any convective proxy where low means
        enhanced convection), ``(time, lat, lon)``.
    config:
        :class:`KelvinEventConfig`.
    axis_lat:
        Optional externally supplied ``(time, lon)`` convective axis latitude
        (for instance from a precipitation climatology); overrides the
        variance-based axis.
    prefiltered:
        Set ``True`` when inputs are already Kelvin-band filtered anomalies.

    Returns
    -------
    KelvinEventCatalog
        ``events`` carries a ``kind`` column: ``"coupled"`` (dynamical event
        whose convection correlates with it), ``"dynamical_only"`` and
        ``"convective_only"``.
    """
    cfg = config or KelvinEventConfig()
    if u is None and olr is None:
        raise ValueError("Provide at least `u` or `olr`.")

    def prep(da: Optional[xr.DataArray], name: str) -> Optional[xr.DataArray]:
        if da is None:
            return None
        if prefiltered:
            return _standardize(da, name)
        return kelvin_band_filter(
            da,
            period_days=cfg.period_days,
            wavenumber=cfg.wavenumber,
            equivalent_depth=cfg.equivalent_depth,
            obs_per_day=cfg.obs_per_day,
            remove_seasonal_cycle=cfg.remove_seasonal_cycle,
            n_harm=cfg.n_harm,
        )

    u_k = prep(u, "u")
    z_k = prep(z, "z")
    v_k = prep(v, "v")
    olr_k = prep(olr, "olr")

    indices: Dict[str, xr.DataArray] = {}
    track_kwargs = dict(
        peak_sigma=cfg.peak_sigma,
        continue_sigma=cfg.continue_sigma,
        normalize=cfg.normalize,
        min_lifetime_days=cfg.min_lifetime_days,
        min_distance_deg=cfg.min_distance_deg,
        speed_range=cfg.speed_range,
        expected_speed=cfg.expected_speed,
        max_gap_steps=cfg.max_gap_steps,
    )

    dyn_events, dyn_tracks = _empty_events(), _empty_tracks()
    k_dyn = None
    if u_k is not None:
        k_dyn = kelvin_dynamical_index(
            u_k,
            z_k,
            equivalent_depth=cfg.projection_depth,
            method=cfg.projection_method,
            lat_limit=cfg.projection_lat_limit,
        )
        indices["kelvin_dynamical"] = k_dyn
        dyn_events, dyn_tracks = detect_propagating_ridges(k_dyn, prefix="D", min_amplitude=cfg.dyn_min_amplitude, **track_kwargs)

    conv_events, conv_tracks = _empty_events(), _empty_tracks()
    c_axis = c_fixed = axis = None
    if olr_k is not None:
        axis = (
            axis_lat
            if axis_lat is not None
            else convective_axis_latitude(
                olr_k,
                lat_range=cfg.axis_lat_range,
                window_days=cfg.axis_window_days,
                smooth_lon_deg=cfg.axis_smooth_lon_deg,
                obs_per_day=cfg.obs_per_day,
            )
        )
        c_axis = convective_kelvin_index(olr_k, axis, width_deg=cfg.conv_width_deg)
        c_fixed = fixed_band_convective_index(olr_k, band=cfg.fixed_band)
        indices.update({"kelvin_convective_axis": c_axis, "kelvin_convective_fixed_band": c_fixed, "convective_axis_lat": axis})
        conv_events, conv_tracks = detect_propagating_ridges(c_axis, prefix="C", min_amplitude=cfg.conv_min_amplitude, **track_kwargs)

    # ---- dynamical events: coupling and structure ---------------------------
    dyn_rows: List[dict] = []
    if k_dyn is not None and not dyn_events.empty:
        times = pd.to_datetime(k_dyn.time.values)
        lon0 = float(k_dyn.lon.values[0])
        dlon = _require_global_longitude(k_dyn.lon.values)
        k_values = np.nan_to_num(k_dyn.values.astype(float))
        c_axis_values = None if c_axis is None else c_axis.reindex(time=k_dyn.time).values.astype(float)
        c_fixed_values = None if c_fixed is None else c_fixed.reindex(time=k_dyn.time).values.astype(float)
        axis_values = None if axis is None else axis.reindex(time=k_dyn.time).values.astype(float)
        u_values = None if u_k is None else u_k.values.astype(float)
        v_values = None if v_k is None else v_k.values.astype(float)

        for _, ev in dyn_events.iterrows():
            row = ev.to_dict()
            row.update(
                {
                    "coupling_corr": np.nan,
                    "coupling_pvalue": np.nan,
                    "coupling_corr_fixed_band": np.nan,
                    "conv_offset_deg": np.nan,
                    "conv_axis_lat": np.nan,
                    "conv_peak": np.nan,
                    "conv_peak_fixed_band": np.nan,
                }
            )
            t_idx, j0, _ = _fitted_path(ev, times, lon0, dlon, k_values.shape[1])
            if c_axis_values is not None:
                r, shift, pval, t_idx, j0 = _coupling(
                    ev,
                    k_values,
                    c_axis_values,
                    times,
                    lon0,
                    dlon,
                    cfg.coupling_max_offset_deg,
                    n_surrogates=cfg.coupling_surrogates,
                    obs_per_day=cfg.obs_per_day,
                )
                row["coupling_corr"], row["conv_offset_deg"], row["coupling_pvalue"] = r, shift, pval
                if np.isfinite(shift):
                    j_conv = (j0 + round(shift / dlon)) % k_values.shape[1]
                    row["conv_axis_lat"] = float(np.nanmean(axis_values[t_idx, j_conv]))
                    row["conv_peak"] = float(np.nanmax(c_axis_values[t_idx, j_conv]))
                r_fixed, _, _, _, _ = _coupling(ev, k_values, c_fixed_values, times, lon0, dlon, cfg.coupling_max_offset_deg)
                row["coupling_corr_fixed_band"] = r_fixed
                if np.isfinite(shift):
                    row["conv_peak_fixed_band"] = float(np.nanmax(c_fixed_values[t_idx, j_conv]))
            lat_grid = u_k.lat.values if u_k is not None else np.array([])
            sym, ratio = _structure_diagnostics(
                u_values, v_values, lat_grid, t_idx, j0, dlon, cfg.symmetry_window_deg, cfg.symmetry_lat_max
            )
            row["u_symmetry_index"], row["v_to_u_ratio"] = sym, ratio
            coupled = (
                np.isfinite(row["coupling_corr"])
                and row["coupling_corr"] >= cfg.coupling_threshold
                and np.isfinite(row["coupling_pvalue"])
                and row["coupling_pvalue"] <= cfg.coupling_alpha
            )
            row["kind"] = "coupled" if coupled else "dynamical_only"
            row["matched_event"] = ""
            dyn_rows.append(row)

    dyn_table = pd.DataFrame(dyn_rows)

    # ---- convective events without a dynamical counterpart ------------------
    conv_rows: List[dict] = []
    matches = _match_convective_to_dynamical(
        conv_events, conv_tracks, dyn_tracks, lon_tol=cfg.match_lon_tol_deg, overlap=cfg.match_overlap
    )
    for _, ev in conv_events.iterrows():
        row = ev.to_dict()
        if ev["event_id"] in matches:
            continue
        row["kind"] = "convective_only" if u_k is not None else "convective"
        row["matched_event"] = ""
        if axis is not None:
            t_mid = pd.Timestamp(ev["peak_time"])
            j = int(np.rint((ev["lon_peak"] - float(axis.lon.values[0])) / (360.0 / axis.sizes["lon"]))) % axis.sizes["lon"]
            row["conv_axis_lat"] = float(axis.sel(time=t_mid, method="nearest").values[j])
        conv_rows.append(row)
    conv_table = pd.DataFrame(conv_rows)

    # Record which convective ridge belongs to which dynamical event.
    if not dyn_table.empty and matches:
        reverse: Dict[str, List[str]] = {}
        for cid, did in matches.items():
            reverse.setdefault(did, []).append(cid)
        dyn_table["matched_event"] = dyn_table["event_id"].map(lambda i: ",".join(sorted(reverse.get(i, []))))

    parts = [t for t in (dyn_table, conv_table) if not t.empty]
    events = pd.concat(parts, ignore_index=True) if parts else _empty_events().assign(kind=pd.Series(dtype=object))
    if not events.empty:
        events = events.sort_values("start_time").reset_index(drop=True)

    tracks_parts = []
    if not dyn_tracks.empty:
        tracks_parts.append(dyn_tracks.assign(channel="dynamical"))
    if not conv_tracks.empty:
        tracks_parts.append(conv_tracks.assign(channel="convective"))
    tracks = pd.concat(tracks_parts, ignore_index=True) if tracks_parts else _empty_tracks().assign(channel=pd.Series(dtype=object))

    return KelvinEventCatalog(events=events, tracks=tracks, indices=xr.Dataset(indices), config=cfg)


# --------------------------------------------------------------------------- #
# Composites
# --------------------------------------------------------------------------- #
def event_composite(
    field_: xr.DataArray,
    events: pd.DataFrame,
    *,
    kinds: Sequence[str] = ("coupled",),
    lags_days: Sequence[float] = tuple(range(-4, 5)),
    half_width_deg: float = 60.0,
) -> xr.DataArray:
    """Wave-following composite ``(lag, lat, rel_lon)`` of a ``(time, lat, lon)`` field.

    Each event is aligned on its ``peak_time``/``lon_peak`` and moves with its
    own fitted phase speed, so the composite is in the frame of the wave.  Use
    it with the *unfiltered or lightly filtered* anomaly of ``u`` and ``OLR``
    to see directly whether the circulation is equatorially symmetric while
    the convection sits off the equator.
    """
    f = _standardize(field_, "field_")
    dlon = _require_global_longitude(f.lon.values)
    lon0 = float(f.lon.values[0])
    nlon = f.sizes["lon"]
    times = pd.to_datetime(f.time.values)
    dt = float(np.median(np.diff(_time_in_days(f.time.values))))
    rel = np.arange(-half_width_deg, half_width_deg + 0.5 * dlon, dlon)
    sel = events[events["kind"].isin(list(kinds))] if "kind" in events.columns else events
    acc = np.zeros((len(lags_days), f.sizes["lat"], rel.size))
    cnt = np.zeros(len(lags_days))
    values = f.values
    for _, ev in sel.iterrows():
        t_peak = int(times.get_indexer([ev["peak_time"]], method="nearest")[0])
        for li, lag in enumerate(lags_days):
            ti = t_peak + round(lag / dt)
            if ti < 0 or ti >= times.size:
                continue
            centre = ev["lon_peak"] + ev["fit_slope_deg_per_day"] * lag
            cols = np.rint((np.mod(centre + rel, 360.0) - lon0) / dlon).astype(int) % nlon
            slab = values[ti][:, cols]
            acc[li] += np.nan_to_num(slab)
            cnt[li] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        comp = acc / cnt[:, None, None]
    return xr.DataArray(
        comp,
        dims=("lag", "lat", "rel_lon"),
        coords={"lag": np.asarray(lags_days, dtype=float), "lat": f.lat.values, "rel_lon": rel},
        name=field_.name,
        attrs={"n_events": int(sel.shape[0]), "kinds": ",".join(kinds)},
    )
