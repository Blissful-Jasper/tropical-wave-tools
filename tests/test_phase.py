from __future__ import annotations

import numpy as np

from tropical_wave_tools.phase import (
    calculate_kelvin_phase,
    meridional_projection,
    optimize_peak_detection,
    phase_composite,
)


def test_kelvin_phase_workflow_returns_phase_and_composites(synthetic_wave_data) -> None:
    kelvin_ref = meridional_projection(synthetic_wave_data, synthetic_wave_data.lat.values)
    value_std = float(np.nanstd(kelvin_ref.values))
    peak_values, extrema = optimize_peak_detection(
        kelvin_ref.values,
        kelvin_ref,
        value_std,
        use_parallel=False,
    )
    phase = calculate_kelvin_phase(kelvin_ref, peak_values)
    phase_bin, means, counts = phase_composite(kelvin_ref, phase, n_bins=8)

    assert extrema.dims == kelvin_ref.dims
    assert peak_values.dims == kelvin_ref.dims
    assert phase.dims == kelvin_ref.dims
    assert np.isfinite(phase.values).any()
    assert phase_bin.shape == (8,)
    assert means.shape == (8,)
    assert counts.shape == (8,)
    assert int(np.nansum(counts)) > 0

