from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tropical_wave_tools.diagnostics import (
    area_weighted_mean,
    horizontal_divergence,
    relative_vorticity,
    zonal_mean,
)


def test_area_weighted_mean_reduces_to_time_series(synthetic_wave_data) -> None:
    regional_mean = area_weighted_mean(synthetic_wave_data, lat_range=(-10.0, 10.0))
    assert regional_mean.dims == ("time",)


def test_zonal_mean_removes_longitude_dimension(synthetic_wave_data) -> None:
    output = zonal_mean(synthetic_wave_data)
    assert "lon" not in output.dims


EARTH_RADIUS = 6.371e6


def _grid(lat_step: float = 1.0, lon_step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(-30.0, 30.1, lat_step), np.arange(0.0, 360.0, lon_step)


def _field(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, units: str = "m s-1") -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": units},
    )


def test_constant_components_follow_spherical_metric_terms() -> None:
    """On a sphere uniform (u, v) is not planar: zeta = u tan(phi)/a, div = -v tan(phi)/a."""
    lat = np.array([-10.0, 0.0, 10.0])
    lon = np.arange(0.0, 360.0, 30.0)
    u = _field(np.full((lat.size, lon.size), 5.0), lat, lon)
    v = _field(np.full((lat.size, lon.size), -2.0), lat, lon)
    tangent = np.tan(np.deg2rad(lat))[:, None]
    np.testing.assert_allclose(relative_vorticity(u, v).values, 5.0 * tangent / EARTH_RADIUS + 0 * lon, atol=1e-12)
    np.testing.assert_allclose(horizontal_divergence(u, v).values, 2.0 * tangent / EARTH_RADIUS + 0 * lon, atol=1e-12)
    assert np.allclose(relative_vorticity(u, v).sel(lat=0.0), 0.0)


def test_solid_body_rotation_vorticity_is_two_omega_sin_phi() -> None:
    """u = U0 cos(phi): zeta = 2 U0 sin(phi) / a and no divergence (needs the curvature term)."""
    lat, lon = _grid()
    u0 = 20.0
    u = _field(u0 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)), lat, lon)
    v = _field(np.zeros((lat.size, lon.size)), lat, lon)
    expected = 2.0 * u0 * np.sin(np.deg2rad(lat))[:, None] / EARTH_RADIUS
    inner = slice(2, -2)
    error = np.abs(relative_vorticity(u, v).values - expected)[inner]
    assert error.max() < 1e-3 * np.abs(expected).max()
    assert np.abs(horizontal_divergence(u, v).values[inner]).max() < 1e-12


def test_divergence_and_vorticity_match_analytic_field_including_the_seam() -> None:
    lat, lon = _grid()
    phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    u = _field(20 * np.cos(phi) * np.sin(2 * lam), lat, lon)
    v = _field(10 * np.cos(phi) ** 2 * np.cos(lam), lat, lon)
    divergence = (
        20 * np.cos(phi) * 2 * np.cos(2 * lam) + 10 * (-3) * np.cos(phi) ** 2 * np.sin(phi) * np.cos(lam)
    ) / (EARTH_RADIUS * np.cos(phi))
    vorticity = (
        -10 * np.cos(phi) ** 2 * np.sin(lam) - 20 * np.sin(2 * lam) * (-2 * np.cos(phi) * np.sin(phi))
    ) / (EARTH_RADIUS * np.cos(phi))
    inner = slice(2, -2)  # one-sided differences at the pole-most rows are not expected to be exact
    for result, expected in (
        (horizontal_divergence(u, v), divergence),
        (relative_vorticity(u, v), vorticity),
    ):
        rms = np.sqrt(np.mean((result.values[inner] - expected[inner]) ** 2)) / expected[inner].std()
        assert rms < 5e-3
        seam = np.abs(result.values[inner, 0] - expected[inner, 0]).max() / expected[inner].std()
        assert seam < 5e-3  # the periodic seam must not use one-sided differences


@pytest.mark.parametrize("units", ["m/s", "m s-1", "m s**-1", "m s^-1"])
def test_cf_wind_units_are_accepted(units: str) -> None:
    lat, lon = _grid(5.0, 10.0)
    u = _field(np.ones((lat.size, lon.size)), lat, lon, units)
    v = _field(np.zeros((lat.size, lon.size)), lat, lon, units)
    assert np.isfinite(relative_vorticity(u, v).values).all()


def test_unreadable_wind_units_raise_a_clear_error() -> None:
    pytest.importorskip("metpy")
    lat, lon = _grid(5.0, 10.0)
    field = _field(np.ones((lat.size, lon.size)), lat, lon, "furlongs per fortnight squared")
    with pytest.raises(ValueError, match="wind units"):
        relative_vorticity(field, field)
