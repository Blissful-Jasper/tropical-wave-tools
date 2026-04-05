from __future__ import annotations

import numpy as np
import pandas as pd
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


def test_constant_wind_has_zero_divergence_and_vorticity() -> None:
    time = pd.date_range("2001-01-01", periods=4, freq="D")
    lat = np.array([-10.0, 0.0, 10.0], dtype=float)
    lon = np.array([0.0, 30.0, 60.0, 90.0], dtype=float)
    u = xr.DataArray(
        np.full((time.size, lat.size, lon.size), 5.0),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="u850",
    )
    v = xr.DataArray(
        np.full((time.size, lat.size, lon.size), -2.0),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="v850",
    )
    divergence = horizontal_divergence(u, v)
    vorticity = relative_vorticity(u, v)
    assert np.allclose(divergence.values, 0.0)
    assert np.allclose(vorticity.values, 0.0)
