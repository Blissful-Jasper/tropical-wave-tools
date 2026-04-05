from __future__ import annotations

import numpy as np
import xarray as xr

from tropical_wave_tools.io import normalize_longitude, standardize_data


def test_standardize_data_recognizes_common_dimension_aliases(synthetic_wave_data: xr.DataArray) -> None:
    aliased = synthetic_wave_data.rename({"time": "valid_time", "lat": "latitude", "lon": "longitude"})
    standardized = standardize_data(aliased)
    assert standardized.dims == ("time", "lat", "lon")


def test_normalize_longitude_converts_negative_longitudes(synthetic_wave_data: xr.DataArray) -> None:
    shifted = synthetic_wave_data.assign_coords(lon=np.linspace(-180.0, 150.0, synthetic_wave_data.sizes["lon"]))
    normalized = normalize_longitude(shifted, target="0_360")
    assert float(normalized.lon.min()) >= 0.0
    assert float(normalized.lon.max()) <= 360.0


def test_standardize_data_squeezes_singleton_level_dimension(
    synthetic_wave_data: xr.DataArray,
) -> None:
    with_level = synthetic_wave_data.expand_dims(level=[850.0]).transpose("time", "level", "lat", "lon")
    standardized = standardize_data(with_level)
    assert standardized.dims == ("time", "lat", "lon")
