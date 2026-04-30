from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tropical_wave_tools.preprocess import compute_anomaly, compute_climatology, seasonal_mean
from tropical_wave_tools.preprocessing import (
    build_wk_decomposition_layout,
    decompose_symmetric_antisymmetric,
    smooth_121,
)


def test_smooth_121_preserves_length() -> None:
    values = np.array([1.0, 3.0, 9.0, 3.0, 1.0])
    smoothed = smooth_121(values)
    assert smoothed.shape == values.shape
    assert np.isfinite(smoothed).all()


def test_symmetric_antisymmetric_shapes(synthetic_wave_data: xr.DataArray) -> None:
    symmetric, antisymmetric = decompose_symmetric_antisymmetric(synthetic_wave_data)
    assert symmetric.shape == synthetic_wave_data.shape
    assert antisymmetric.shape == synthetic_wave_data.shape


def test_build_wk_layout_preserves_shape(synthetic_wave_data: xr.DataArray) -> None:
    layout = build_wk_decomposition_layout(synthetic_wave_data)
    assert layout.shape == synthetic_wave_data.shape
    assert layout.dims == synthetic_wave_data.dims


def test_decompose_requires_symmetric_latitude_grid(synthetic_wave_data: xr.DataArray) -> None:
    asymmetric = synthetic_wave_data.isel(lat=slice(1, None))

    with pytest.raises(ValueError, match="paired about the equator"):
        decompose_symmetric_antisymmetric(asymmetric)


def test_decompose_symmetric_antisymmetric_values() -> None:
    lat = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=float)
    lon = np.array([0.0, 90.0], dtype=float)
    time = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")

    symmetric_values = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
        ]
    )
    antisymmetric_values = np.array(
        [
            [[-1.0, -1.0], [-2.0, -2.0], [0.0, 0.0], [2.0, 2.0], [1.0, 1.0]],
            [[-1.0, -1.0], [-2.0, -2.0], [0.0, 0.0], [2.0, 2.0], [1.0, 1.0]],
        ]
    )

    symmetric_field = xr.DataArray(
        symmetric_values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )
    antisymmetric_field = xr.DataArray(
        antisymmetric_values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )

    symmetric, antisymmetric = decompose_symmetric_antisymmetric(symmetric_field)
    assert np.allclose(symmetric.values, symmetric_values)
    assert np.allclose(antisymmetric.values, 0.0)

    symmetric, antisymmetric = decompose_symmetric_antisymmetric(antisymmetric_field)
    assert np.allclose(symmetric.values, 0.0)
    assert np.allclose(antisymmetric.values, antisymmetric_values)


def test_build_wk_layout_places_components_in_expected_halves() -> None:
    lat = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=float)
    lon = np.array([0.0], dtype=float)
    time = np.array(["2000-01-01"], dtype="datetime64[ns]")
    values = np.array([[[1.0], [2.0], [3.0], [2.0], [1.0]]])
    field = xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon})

    layout = build_wk_decomposition_layout(field)
    assert np.allclose(layout.values[:, :2, :], 0.0)
    assert np.allclose(layout.values[:, 2:, :], values[:, 2:, :])


def test_compute_climatology_and_anomaly(synthetic_wave_data: xr.DataArray) -> None:
    climatology = compute_climatology(synthetic_wave_data, group="month")
    anomaly = compute_anomaly(synthetic_wave_data, group="month")
    assert "month" in climatology.dims
    assert anomaly.shape == synthetic_wave_data.shape


def test_seasonal_mean_returns_grouped_data(synthetic_wave_data: xr.DataArray) -> None:
    seasonal = seasonal_mean(synthetic_wave_data)
    assert "season" in seasonal.dims
