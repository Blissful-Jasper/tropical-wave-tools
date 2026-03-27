from __future__ import annotations

from tropical_wave_tools.diagnostics import area_weighted_mean, zonal_mean


def test_area_weighted_mean_reduces_to_time_series(synthetic_wave_data) -> None:
    regional_mean = area_weighted_mean(synthetic_wave_data, lat_range=(-10.0, 10.0))
    assert regional_mean.dims == ("time",)


def test_zonal_mean_removes_longitude_dimension(synthetic_wave_data) -> None:
    output = zonal_mean(synthetic_wave_data)
    assert "lon" not in output.dims
