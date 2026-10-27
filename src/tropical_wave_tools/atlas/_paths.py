"""Local data paths and loading of the pre-filtered wave fields."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

from tropical_wave_tools.io import load_dataarray

PathLike = Union[str, Path]


_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../atlas/_paths.py -> repository root


def _resolve_local_project_path(path: PathLike) -> Path:
    """Resolve local-data paths relative to the repository when needed."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (_PROJECT_ROOT / candidate).resolve()


DEFAULT_LOCAL_PATHS = {
    "olr": _resolve_local_project_path("data/local/olr.day.mean.nc"),
    "gpcp": _resolve_local_project_path("data/local/GPCP_data_1997-2020-2.5x2.5_stand.nc"),
    "u850": _resolve_local_project_path("data/local/uwnd_850hPa_1979-2024.nc"),
    "v850": _resolve_local_project_path("data/local/vwnd_850hPa_1979-2024.nc"),
}


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
    time_range: tuple[str, str] | None = ("1979-01-01", "2014-12-31"),
    lat_range: tuple[float, float] = (-20.0, 20.0),
    lon_range: tuple[float, float] | None = None,
) -> xr.Dataset:
    """Load and align local OLR/U850/V850 datasets on a common grid."""
    olr_path = _resolve_local_project_path(olr_path)
    u850_path = _resolve_local_project_path(u850_path)
    v850_path = _resolve_local_project_path(v850_path)
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
