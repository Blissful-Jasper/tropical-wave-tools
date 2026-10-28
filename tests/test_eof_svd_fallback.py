"""EOFs must survive LAPACK's divide-and-conquer SVD failing to converge on degenerate input."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tropical_wave_tools.atlas import compute_wave_eof
from tropical_wave_tools.eof import EOFAnalyzer, _thin_svd


def _travelling_wave(n_days=1095, wavenumber=5, period=6.0):
    """An eastward wave with a latitudinal envelope, as filtered Kelvin-wave OLR looks like."""
    time = pd.date_range("1979-01-01", periods=n_days)
    lat = np.arange(-20.0, 20.1, 5.0)
    lon = np.arange(0.0, 360.0, 10.0)
    t = np.arange(n_days)[:, None, None]
    values = (
        np.exp(-((lat[None, :, None] / 8.0) ** 2))
        * np.cos(wavenumber * np.deg2rad(lon)[None, None, :] - 2.0 * np.pi * t / period)
        * (1.0 + 0.5 * np.sin(2.0 * np.pi * t / 365.25))
    )
    return xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}, name="olr")


def test_wave_eof_of_a_travelling_wave_is_a_quadrature_pair() -> None:
    """The wave-EOF input of a propagating wave is rank-deficient (mirrored latitudes repeat) with a degenerate pair.

    With some LAPACK builds ``np.linalg.svd`` raised "SVD did not converge" on such finite matrices
    (it did for the filtered Kelvin-wave OLR of the atlas workflow). Whether a given matrix triggers
    that depends on the BLAS in use, so the deterministic check of the fallback is the monkeypatched
    test below; this one guards the physics (a travelling wave is a quadrature pair of EOFs).
    """
    eofs, pcs, explained, _ = compute_wave_eof(_travelling_wave(), wave_name="kelvin", n_modes=2)
    assert explained[0] == pytest.approx(explained[1], rel=1e-3)
    assert explained[:2].sum() > 99.0
    assert abs(float(xr.corr(pcs.sel(mode=1), pcs.sel(mode=2)))) < 1e-3
    assert eofs.dims == ("mode", "lat", "lon")


def test_svd_backend_falls_back_to_the_qr_driver(monkeypatch) -> None:
    field = _travelling_wave(n_days=400)
    expected = EOFAnalyzer(method="svd").fit(field, n_harmonics=0, n_modes=2)

    def does_not_converge(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", does_not_converge)
    result = EOFAnalyzer(method="svd").fit(field, n_harmonics=0, n_modes=2)
    np.testing.assert_allclose(result["eigenvalues"][:2], expected["eigenvalues"][:2], rtol=1e-10)
    np.testing.assert_allclose(result["explained_variance"][:2], expected["explained_variance"][:2], rtol=1e-10)
    # the degenerate pair may come back rotated; the pair as a whole spans the same plane
    basis = expected["eofs"].values[:2].reshape(2, -1)
    rotated = result["eofs"].values[:2].reshape(2, -1)
    np.testing.assert_allclose(basis.T @ (basis @ rotated.T), rotated.T, atol=1e-8)


def test_thin_svd_reconstructs_the_matrix_with_either_driver(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((30, 200))
    left, singular_values = _thin_svd(matrix)
    _, reference, _ = np.linalg.svd(matrix, full_matrices=False)
    np.testing.assert_allclose(singular_values, reference, rtol=1e-12)
    np.testing.assert_allclose(left.T @ left, np.eye(30), atol=1e-12)

    def does_not_converge(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", does_not_converge)
    left_fallback, values_fallback = _thin_svd(matrix)
    np.testing.assert_allclose(values_fallback, reference, rtol=1e-12)
    np.testing.assert_allclose(np.abs(left_fallback.T @ left), np.eye(30), atol=1e-8)  # same singular vectors up to sign
