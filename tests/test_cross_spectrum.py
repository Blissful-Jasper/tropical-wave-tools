from __future__ import annotations

import numpy as np
import xarray as xr

from tropical_wave_tools.cross_spectrum import _segment_starts, calculate_cross_spectrum
from tropical_wave_tools.cross_spectrum_analysis import compute_cross_spectrum_for_experiments


def test_calculate_cross_spectrum_returns_expected_components(synthetic_wave_data) -> None:
    result = calculate_cross_spectrum(
        synthetic_wave_data,
        synthetic_wave_data,
        segLen=64,
        segOverLap=-32,
        symmetry="symm",
    )

    spectrum = result["STC"]
    assert spectrum.dims == ("component", "frequency", "wavenumber")
    assert set(spectrum.component.values.tolist()) == {"PX", "PY", "CXY", "QXY", "COH2", "PHAS", "V1", "V2"}
    assert int(result["nseg"]) > 0
    assert float(result["dof"]) > 0.0
    assert np.isfinite(spectrum.sel(component="COH2").values[1:, :]).any()


def test_segment_starts_use_requested_overlap() -> None:
    assert _segment_starts(192, 64, -32) == [0, 32, 64, 96, 128]


def test_calculate_cross_spectrum_scales_frequency_by_samples_per_day(synthetic_wave_data) -> None:
    result = calculate_cross_spectrum(
        synthetic_wave_data,
        synthetic_wave_data,
        segLen=64,
        segOverLap=-32,
        symmetry="symm",
        samples_per_day=4,
    )
    assert np.isclose(float(result["freq"][-1]), 2.0)


def test_masked_cross_spectrum_preserves_wavenumber_geometry(synthetic_wave_data) -> None:
    mask = xr.DataArray(
        np.ones((synthetic_wave_data.sizes["lat"], synthetic_wave_data.sizes["lon"]), dtype=bool),
        dims=("lat", "lon"),
        coords={"lat": synthetic_wave_data.lat, "lon": synthetic_wave_data.lon},
    )
    mask.loc[{"lon": synthetic_wave_data.lon.values[0]}] = False

    results = compute_cross_spectrum_for_experiments(
        {"ctrl": synthetic_wave_data},
        {"ctrl": synthetic_wave_data},
        experiments=("ctrl",),
        mask=mask,
        seg_length=64,
        seg_overlap=-32,
        verbose=False,
    )

    spectrum = results["ctrl"]["STC"]
    assert spectrum.sizes["wavenumber"] == synthetic_wave_data.sizes["lon"] + 1
