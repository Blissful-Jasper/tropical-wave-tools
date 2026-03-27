from __future__ import annotations

import numpy as np

from tropical_wave_tools.config import SpectralConfig
from tropical_wave_tools.spectral import analyze_wk_spectrum


def test_analyze_wk_spectrum_returns_expected_coords(synthetic_wave_data) -> None:
    config = SpectralConfig(window_size_days=64, window_skip_days=16)
    result = analyze_wk_spectrum(synthetic_wave_data, config=config)

    assert result.power_symmetric.dims == ("frequency", "wavenumber")
    assert result.power_antisymmetric.dims == ("frequency", "wavenumber")
    assert result.background.dims == ("frequency", "wavenumber")
    assert np.isfinite(result.background.isel(frequency=slice(1, None)).values).all()

