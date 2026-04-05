"""General climate diagnostics and optional moist-stability helpers."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import xarray as xr

from tropical_wave_tools.io import normalize_longitude, rename_standard_coordinates, standardize_data

CP = 1004.0
GRAVITY = 9.81
LATENT_HEAT = 2.5e6
REFERENCE_TEMPERATURE = 300.0
EARTH_RADIUS = 6.371e6
_METPY_FALLBACK_WARNED = False


def zonal_mean(data: xr.DataArray, *, dim: str = "lon") -> xr.DataArray:
    """Compute the zonal mean."""
    return standardize_data(data).mean(dim=dim)


def meridional_mean(data: xr.DataArray, *, dim: str = "lat") -> xr.DataArray:
    """Compute the meridional mean."""
    return standardize_data(data).mean(dim=dim)


def area_weighted_mean(
    data: xr.DataArray,
    *,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
) -> xr.DataArray:
    """Compute a cosine-latitude weighted regional mean."""
    array = standardize_data(data)
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

    weights = np.cos(np.deg2rad(array["lat"]))
    weights.name = "weights"
    return array.weighted(weights).mean(dim=("lat", "lon"))


def compute_dx_dy(lat: xr.DataArray, lon: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute regular-grid spacing in meters."""
    lat_values = np.asarray(lat.values, dtype=float)
    lon_values = np.asarray(lon.values, dtype=float)
    lat_rad = np.deg2rad(lat_values)
    dlat = np.abs(np.diff(lat_values).mean())
    dlon = np.abs(np.diff(lon_values).mean())
    dy = EARTH_RADIUS * np.deg2rad(dlat)
    dx = EARTH_RADIUS * np.cos(lat_rad) * np.deg2rad(dlon)
    dx = np.broadcast_to(dx[:, np.newaxis], (len(lat_values), len(lon_values)))
    dy = np.full((len(lat_values), len(lon_values)), dy)
    return dx, dy


def _ensure_lat_lon_layout(data: xr.DataArray) -> xr.DataArray:
    """Normalize an input field to either ``(lat, lon)`` or ``(time, lat, lon)``."""
    array = normalize_longitude(rename_standard_coordinates(data), target="0_360")
    required = {"lat", "lon"}
    if not required.issubset(array.dims):
        raise ValueError("Input data must contain lat/lon dimensions.")
    if "time" in array.dims:
        return array.transpose("time", "lat", "lon")
    return array.transpose("lat", "lon")


def _finite_difference_divergence(zonal: xr.DataArray, meridional: xr.DataArray) -> xr.DataArray:
    dx, dy = compute_dx_dy(zonal.lat, zonal.lon)
    du_dx = xr.DataArray(
        np.gradient(zonal.values, axis=zonal.get_axis_num("lon")) / dx,
        coords=zonal.coords,
        dims=zonal.dims,
    )
    dv_dy = xr.DataArray(
        np.gradient(meridional.values, axis=meridional.get_axis_num("lat")) / dy,
        coords=meridional.coords,
        dims=meridional.dims,
    )
    return (du_dx + dv_dy).rename("divergence")


def _finite_difference_vorticity(zonal: xr.DataArray, meridional: xr.DataArray) -> xr.DataArray:
    dx, dy = compute_dx_dy(zonal.lat, zonal.lon)
    dv_dx = xr.DataArray(
        np.gradient(meridional.values, axis=meridional.get_axis_num("lon")) / dx,
        coords=meridional.coords,
        dims=meridional.dims,
    )
    du_dy = xr.DataArray(
        np.gradient(zonal.values, axis=zonal.get_axis_num("lat")) / dy,
        coords=zonal.coords,
        dims=zonal.dims,
    )
    return (dv_dx - du_dy).rename("relative_vorticity")


def _metpy_vector_calculus(
    zonal: xr.DataArray,
    meridional: xr.DataArray,
    *,
    operation: str,
) -> xr.DataArray:
    try:
        from metpy.calc import divergence as mp_divergence
        from metpy.calc import lat_lon_grid_deltas, vorticity as mp_vorticity
        from metpy.units import units
    except Exception as exc:  # pragma: no cover - depends on optional binary stack
        raise RuntimeError("MetPy diagnostics are unavailable in this environment.") from exc

    dx, dy = lat_lon_grid_deltas(zonal.lon.values, zonal.lat.values)
    x_axis = zonal.get_axis_num("lon")
    y_axis = zonal.get_axis_num("lat")
    if zonal.ndim == 3:
        dx = dx[np.newaxis, ...]
        dy = dy[np.newaxis, ...]
    wind_units = units(str(zonal.attrs.get("units", "m/s")).replace(" ", ""))
    zonal_values = np.asarray(zonal.values, dtype=float) * wind_units
    meridional_values = np.asarray(meridional.values, dtype=float) * wind_units

    if operation == "divergence":
        values = mp_divergence(zonal_values, meridional_values, dx=dx, dy=dy, x_dim=x_axis, y_dim=y_axis)
        name = "divergence"
    elif operation == "vorticity":
        values = mp_vorticity(zonal_values, meridional_values, dx=dx, dy=dy, x_dim=x_axis, y_dim=y_axis)
        name = "relative_vorticity"
    else:
        raise ValueError("`operation` must be 'divergence' or 'vorticity'.")

    return xr.DataArray(
        values.to("s^-1").magnitude,
        coords=zonal.coords,
        dims=zonal.dims,
        name=name,
        attrs={"units": "s^-1", "method": "metpy"},
    )


def horizontal_divergence(zonal_wind: xr.DataArray, meridional_wind: xr.DataArray) -> xr.DataArray:
    """Compute horizontal wind divergence on a regular lat-lon grid."""
    zonal = _ensure_lat_lon_layout(zonal_wind)
    meridional = _ensure_lat_lon_layout(meridional_wind)
    zonal, meridional = xr.align(zonal, meridional, join="exact")
    try:
        return _metpy_vector_calculus(zonal, meridional, operation="divergence")
    except RuntimeError as exc:
        global _METPY_FALLBACK_WARNED
        if not _METPY_FALLBACK_WARNED:
            warnings.warn(f"{exc} Falling back to centered finite differences.", RuntimeWarning, stacklevel=2)
            _METPY_FALLBACK_WARNED = True
        return _finite_difference_divergence(zonal, meridional).assign_attrs(
            {"units": "s^-1", "method": "finite-difference-fallback"}
        )


def relative_vorticity(zonal_wind: xr.DataArray, meridional_wind: xr.DataArray) -> xr.DataArray:
    """Compute relative vorticity on a regular lat-lon grid."""
    zonal = _ensure_lat_lon_layout(zonal_wind)
    meridional = _ensure_lat_lon_layout(meridional_wind)
    zonal, meridional = xr.align(zonal, meridional, join="exact")
    try:
        return _metpy_vector_calculus(zonal, meridional, operation="vorticity")
    except RuntimeError as exc:
        global _METPY_FALLBACK_WARNED
        if not _METPY_FALLBACK_WARNED:
            warnings.warn(f"{exc} Falling back to centered finite differences.", RuntimeWarning, stacklevel=2)
            _METPY_FALLBACK_WARNED = True
        return _finite_difference_vorticity(zonal, meridional).assign_attrs(
            {"units": "s^-1", "method": "finite-difference-fallback"}
        )


def calc_dse(temperature: xr.DataArray, geopotential_height: xr.DataArray) -> xr.DataArray:
    """Calculate dry static energy."""
    return CP * temperature + GRAVITY * geopotential_height


def mixing_ratio_from_specific_humidity(specific_humidity: xr.DataArray) -> xr.DataArray:
    """Convert specific humidity to mixing ratio."""
    values = specific_humidity / (1.0 - specific_humidity)
    return values.rename("mixing_ratio")


def vertically_integrated_moist_flux_divergence(
    specific_humidity: xr.DataArray,
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
    *,
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> xr.DataArray:
    """Return latent-energy moist-flux divergence."""
    mixing_ratio = mixing_ratio_from_specific_humidity(specific_humidity)
    flux_u = mixing_ratio * zonal_wind
    flux_v = mixing_ratio * meridional_wind
    dx, dy = compute_dx_dy(lat, lon)
    d_flux_u_dx = xr.DataArray(np.gradient(flux_u, axis=-1) / dx, coords=flux_u.coords, dims=flux_u.dims)
    d_flux_v_dy = xr.DataArray(np.gradient(flux_v, axis=-2) / dy, coords=flux_v.coords, dims=flux_v.dims)
    divergence = d_flux_u_dx + d_flux_v_dy
    return LATENT_HEAT * divergence.integrate("plev")


def calc_horizontal_gms(
    temperature: xr.DataArray,
    geopotential_height: xr.DataArray,
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
    specific_humidity: xr.DataArray,
    *,
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> xr.DataArray:
    """Calculate horizontal gross moist stability."""
    dse = calc_dse(temperature, geopotential_height)
    dx, dy = compute_dx_dy(lat, lon)
    dse_dx = xr.DataArray(np.gradient(dse, axis=-1) / dx, coords=dse.coords, dims=dse.dims)
    dse_dy = xr.DataArray(np.gradient(dse, axis=-2) / dy, coords=dse.coords, dims=dse.dims)
    advection = zonal_wind * dse_dx + meridional_wind * dse_dy
    numerator = advection.integrate("plev")
    denominator = vertically_integrated_moist_flux_divergence(
        specific_humidity,
        zonal_wind,
        meridional_wind,
        lat=lat,
        lon=lon,
    )
    return (-REFERENCE_TEMPERATURE * numerator / denominator).rename("horizontal_gms")


def calc_vertical_gms(
    temperature: xr.DataArray,
    geopotential_height: xr.DataArray,
    vertical_velocity: xr.DataArray,
    specific_humidity: xr.DataArray,
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
    *,
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> xr.DataArray:
    """Calculate vertical gross moist stability."""
    dse = calc_dse(temperature, geopotential_height)
    dse_dp = dse.differentiate("plev")
    numerator = (vertical_velocity * dse_dp).integrate("plev")
    denominator = vertically_integrated_moist_flux_divergence(
        specific_humidity,
        zonal_wind,
        meridional_wind,
        lat=lat,
        lon=lon,
    )
    return (-REFERENCE_TEMPERATURE * numerator / denominator).rename("vertical_gms")


__all__ = [
    "area_weighted_mean",
    "calc_dse",
    "calc_horizontal_gms",
    "calc_vertical_gms",
    "compute_dx_dy",
    "horizontal_divergence",
    "meridional_mean",
    "relative_vorticity",
    "vertically_integrated_moist_flux_divergence",
    "zonal_mean",
]
