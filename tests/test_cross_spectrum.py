from __future__ import annotations

import numpy as np

from tropical_wave_tools.cross_spectrum import calculate_cross_spectrum


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

