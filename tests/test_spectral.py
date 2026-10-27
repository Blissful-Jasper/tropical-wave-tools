from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from tropical_wave_tools.config import SpectralConfig
from tropical_wave_tools.spectral import _window_starts, analyze_wk_spectrum


def test_analyze_wk_spectrum_returns_expected_coords(synthetic_wave_data) -> None:
    config = SpectralConfig(window_size_days=64, window_skip_days=16)
    result = analyze_wk_spectrum(synthetic_wave_data, config=config)

    assert result.power_symmetric.dims == ("frequency", "wavenumber")
    assert result.power_antisymmetric.dims == ("frequency", "wavenumber")
    assert result.background.dims == ("frequency", "wavenumber")
    assert np.isfinite(result.background.isel(frequency=slice(1, None)).values).all()


def test_window_starts_follow_ncl_skip_convention() -> None:
    """NCL nDaySkip: gap between windows; negative means overlap (start step = window + skip)."""
    assert _window_starts(192, 96, -60) == [0, 36, 72]
    assert _window_starts(384, 96, 30) == [0, 126, 252]
    assert _window_starts(384, 96, -60) == [0, 36, 72, 108, 144, 180, 216, 252, 288]


def test_window_starts_reject_non_advancing_windows() -> None:
    import pytest

    with pytest.raises(ValueError):
        _window_starts(384, 96, -96)


def test_default_spectral_config_matches_standard_wk_window() -> None:
    config = SpectralConfig()
    assert config.window_size_days == 96
    assert config.window_skip_days == -60  # WK99: 96-day windows overlapping by 60 days


def test_symmetric_signal_projects_onto_symmetric_power() -> None:
    time = pd.date_range("2000-01-01", periods=192, freq="D")
    lat = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=float)
    lon = np.arange(0.0, 360.0, 30.0, dtype=float)
    phase = np.arange(time.size, dtype=float)[:, None, None]
    lon_phase = np.deg2rad(lon[None, None, :])
    lat_envelope = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)[None, :, None]
    values = lat_envelope * np.sin(2.0 * np.pi * phase / 8.0 + 2.0 * lon_phase)
    field = xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon})

    result = analyze_wk_spectrum(field, config=SpectralConfig(window_size_days=96, window_skip_days=30))
    symmetric_energy = np.nansum(result.power_symmetric.values)
    antisymmetric_energy = np.nansum(result.power_antisymmetric.values)
    assert symmetric_energy > 10.0 * antisymmetric_energy


def test_antisymmetric_signal_projects_onto_antisymmetric_power() -> None:
    time = pd.date_range("2000-01-01", periods=192, freq="D")
    lat = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=float)
    lon = np.arange(0.0, 360.0, 30.0, dtype=float)
    phase = np.arange(time.size, dtype=float)[:, None, None]
    lon_phase = np.deg2rad(lon[None, None, :])
    lat_structure = np.array([-1.0, -2.0, 0.0, 2.0, 1.0], dtype=float)[None, :, None]
    values = lat_structure * np.sin(2.0 * np.pi * phase / 8.0 + 2.0 * lon_phase)
    field = xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon})

    result = analyze_wk_spectrum(field, config=SpectralConfig(window_size_days=96, window_skip_days=30))
    symmetric_energy = np.nansum(result.power_symmetric.values)
    antisymmetric_energy = np.nansum(result.power_antisymmetric.values)
    assert antisymmetric_energy > 10.0 * symmetric_energy
