from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture()
def synthetic_wave_data() -> xr.DataArray:
    time = pd.date_range("2000-01-01", periods=192, freq="D")
    lat = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    lon = np.arange(0.0, 360.0, 30.0, dtype=np.float64)

    time_phase = np.arange(len(time), dtype=np.float64)[:, None, None]
    lat_envelope = np.exp(-((lat[None, :, None] / 8.0) ** 2))
    lon_phase = np.deg2rad(lon[None, None, :])

    values = (
        220.0
        + 8.0 * np.sin(2.0 * np.pi * time_phase / 8.0 + 2.0 * lon_phase)
        + 2.0 * lat_envelope
    )

    return xr.DataArray(
        values.astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="olr",
        attrs={"units": "W m-2"},
    )


@pytest.fixture()
def synthetic_vertical_wave_data() -> xr.DataArray:
    time = pd.date_range("2000-01-01", periods=730, freq="D")
    level = np.array([1000.0, 850.0, 700.0, 500.0], dtype=np.float64)
    lat = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    lon = np.arange(0.0, 360.0, 60.0, dtype=np.float64)

    time_signal_1 = np.sin(2.0 * np.pi * np.arange(time.size, dtype=np.float64) / 30.0)
    time_signal_2 = np.cos(2.0 * np.pi * np.arange(time.size, dtype=np.float64) / 45.0)
    vertical_mode_1 = np.array([1.0, 0.4, -0.2, -0.8], dtype=np.float64)
    vertical_mode_2 = np.array([-0.3, 0.2, 0.7, 1.1], dtype=np.float64)
    lat_pattern = np.cos(np.deg2rad(lat))
    lon_pattern_1 = np.sin(np.deg2rad(lon))
    lon_pattern_2 = np.cos(2.0 * np.deg2rad(lon))

    values = (
        time_signal_1[:, None, None, None]
        * vertical_mode_1[None, :, None, None]
        * lat_pattern[None, None, :, None]
        * lon_pattern_1[None, None, None, :]
        + 0.6
        * time_signal_2[:, None, None, None]
        * vertical_mode_2[None, :, None, None]
        * lat_pattern[None, None, :, None]
        * lon_pattern_2[None, None, None, :]
    )

    return xr.DataArray(
        values.astype("float32"),
        dims=("time", "level", "lat", "lon"),
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
        name="omega",
        attrs={"units": "Pa s-1"},
    )
