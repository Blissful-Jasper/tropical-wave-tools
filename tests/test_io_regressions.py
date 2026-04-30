from __future__ import annotations

import pytest

from tropical_wave_tools.exceptions import InvalidDataArrayError
from tropical_wave_tools.io import load_dataarray


def test_load_dataarray_raises_clear_error_for_empty_subset(synthetic_wave_data, tmp_path) -> None:
    path = tmp_path / "synthetic_wave.nc"
    synthetic_wave_data.to_dataset(name="olr").to_netcdf(path)

    with pytest.raises(InvalidDataArrayError, match="empty after selection"):
        load_dataarray(path, variable="olr", time_range=("1995-01-01", "1995-01-31"))
