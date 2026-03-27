from __future__ import annotations

import numpy as np
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


def test_compute_climatology_and_anomaly(synthetic_wave_data: xr.DataArray) -> None:
    climatology = compute_climatology(synthetic_wave_data, group="month")
    anomaly = compute_anomaly(synthetic_wave_data, group="month")
    assert "month" in climatology.dims
    assert anomaly.shape == synthetic_wave_data.shape


def test_seasonal_mean_returns_grouped_data(synthetic_wave_data: xr.DataArray) -> None:
    seasonal = seasonal_mean(synthetic_wave_data)
    assert "season" in seasonal.dims
