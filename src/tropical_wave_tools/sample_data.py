"""Helpers for packaged sample data."""

from __future__ import annotations

import shutil
from pathlib import Path

import xarray as xr

from tropical_wave_tools.io import load_dataarray

DEFAULT_SAMPLE_FILE = "olr_equatorial_1979.nc"


def get_sample_path(filename: str = DEFAULT_SAMPLE_FILE) -> Path:
    """Return the path to a packaged sample file."""
    return Path(__file__).resolve().parent / "data" / filename


def open_example_olr(*, chunks: dict[str, int] | None = None) -> xr.DataArray:
    """Open the packaged equatorial OLR sample."""
    return load_dataarray(get_sample_path(), variable="olr", chunks=chunks)


def copy_full_example_data(source: str | Path, target_dir: str | Path) -> Path:
    """Copy the full local OLR file into the project-local data area."""
    source_path = Path(source)
    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / source_path.name
    shutil.copy2(source_path, target_path)
    return target_path
