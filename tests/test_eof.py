from __future__ import annotations

import numpy as np

from tropical_wave_tools.eof import EOFAnalyzer


def test_eof_analyzer_svd_returns_vertical_modes(synthetic_vertical_wave_data) -> None:
    analyzer = EOFAnalyzer(method="svd")
    results = analyzer.fit(synthetic_vertical_wave_data, n_modes=2, n_harmonics=2)

    eofs = results["eofs"]
    pcs = results["pc_scores"]

    assert eofs.dims == ("mode", "level")
    assert pcs.dims == ("mode", "time", "lat", "lon")
    assert eofs.sizes["mode"] == 2
    assert np.isfinite(np.asarray(results["explained_variance"][:2])).all()
    assert float(np.asarray(results["explained_variance"])[0]) > 0.0


def test_eof_zero_harmonics_removes_time_mean_only(synthetic_wave_data) -> None:
    analyzer = EOFAnalyzer(method="svd")
    results = analyzer.fit(synthetic_wave_data, n_modes=2, n_harmonics=0)

    anomalies = results["anomalies"]
    climatology = results["climatology"]

    assert "dayofyear" not in climatology.dims
    assert np.allclose(anomalies.mean("time"), 0.0, atol=5.0e-4)
