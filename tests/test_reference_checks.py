"""Checks of statistics, EOF, dispersion and cross-spectrum code against independent references.

Each test compares with something computed a different way (scipy, an eigen-decomposition, the
Matsuno cubic itself, plane waves with known phase) rather than re-running the same formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats as scipy_stats

from tropical_wave_tools import matsuno
from tropical_wave_tools.cross_spectrum import calculate_cross_spectrum
from tropical_wave_tools.eof import EOFAnalyzer
from tropical_wave_tools.stats import linear_regression, linear_trend, pearson_correlation


def _series(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values, dims="time", coords={"time": pd.date_range("2000-01-01", periods=values.size)}
    )


# ------------------------------------------------------------------ statistics
def test_regression_and_correlation_use_only_samples_valid_in_both() -> None:
    """Different missing values in x and y must not push the correlation outside [-1, 1]."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    y = 2.5 * x + 1.0 + 0.5 * rng.standard_normal(200)
    x[:20] = np.nan
    y[100:140] = np.nan
    valid = np.isfinite(x) & np.isfinite(y)
    reference = scipy_stats.linregress(x[valid], y[valid])

    result = linear_regression(_series(x), _series(y))
    assert float(result.slope) == pytest.approx(reference.slope, rel=1e-10)
    assert float(result.intercept) == pytest.approx(reference.intercept, rel=1e-10)
    assert float(result.correlation) == pytest.approx(reference.rvalue, rel=1e-10)
    assert abs(float(pearson_correlation(_series(x), _series(y)))) <= 1.0


def test_linear_trend_matches_scipy_and_autocorrelation_option_widens_uncertainty() -> None:
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(300)
    series = np.zeros(300)
    for index in range(1, 300):
        series[index] = 0.8 * series[index - 1] + noise[index]
    reference = scipy_stats.linregress(np.arange(300.0), series)

    plain = linear_trend(_series(series))
    adjusted = linear_trend(_series(series), adjust_for_autocorrelation=True)
    assert float(plain.slope) == pytest.approx(reference.slope, rel=1e-10)
    assert float(plain.pvalue) == pytest.approx(reference.pvalue, rel=1e-10)
    assert float(adjusted.slope) == pytest.approx(float(plain.slope))
    assert float(adjusted.stderr) > float(plain.stderr)
    assert float(adjusted.pvalue) > float(plain.pvalue)


# ------------------------------------------------------------------------- EOF
def _two_mode_field(seed: int = 3) -> tuple[xr.DataArray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nt, nlat, nlon = 600, 9, 16
    lat = np.linspace(-20, 20, nlat)
    lon = np.arange(nlon) * 360.0 / nlon
    pattern_1 = np.cos(np.deg2rad(lat))[:, None] * np.sin(2 * np.deg2rad(lon))[None, :]
    pattern_2 = np.exp(-((lat / 10) ** 2))[:, None] * np.cos(3 * np.deg2rad(lon))[None, :]
    pattern_1 /= np.linalg.norm(pattern_1)
    pattern_2 /= np.linalg.norm(pattern_2)
    amplitude_1, amplitude_2 = 4.0 * rng.standard_normal(nt), 2.0 * rng.standard_normal(nt)
    field = (
        amplitude_1[:, None, None] * pattern_1
        + amplitude_2[:, None, None] * pattern_2
        + 0.05 * rng.standard_normal((nt, nlat, nlon))
    )
    data = xr.DataArray(
        field,
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2001-01-01", periods=nt), "lat": lat, "lon": lon},
    )
    return data, pattern_1, pattern_2, amplitude_1, amplitude_2


def test_horizontal_eof_matches_covariance_eigendecomposition() -> None:
    data, pattern_1, pattern_2, _, _ = _two_mode_field()
    result = EOFAnalyzer(method="svd").fit(data, n_modes=3, n_harmonics=0)

    anomalies = data.values - data.values.mean(0)
    flat = anomalies.reshape(anomalies.shape[0], -1)
    eigenvalues, eigenvectors = np.linalg.eigh(flat.T @ flat / flat.shape[0])
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

    np.testing.assert_allclose(result["eigenvalues"][:3], eigenvalues[:3], rtol=1e-6)
    patterns = result["eofs"].transpose("mode", "lat", "lon").values.reshape(3, -1)
    for mode, truth in enumerate((pattern_1, pattern_2)):
        assert abs(np.dot(patterns[mode], eigenvectors[:, mode])) == pytest.approx(1.0, abs=1e-6)
        assert abs(np.corrcoef(patterns[mode], truth.ravel())[0, 1]) > 0.999


def test_horizontal_eof_survives_a_masked_grid_point() -> None:
    """One land point used to discard every time step and return zero modes."""
    data, *_ = _two_mode_field()
    masked = data.copy()
    masked[:, 2, 3] = np.nan
    result = EOFAnalyzer(method="svd").fit(masked, n_modes=2, n_harmonics=0)

    assert result["eofs"].sizes["mode"] == 2
    assert int(result["valid_mask"].sum()) == data.sizes["time"]
    eofs = result["eofs"].transpose("mode", "lat", "lon").values
    assert np.isnan(eofs[:, 2, 3]).all()
    assert np.isfinite(np.delete(eofs.reshape(2, -1), 2 * 16 + 3, axis=1)).all()

    keep = np.ones(data.shape[1:], dtype=bool)
    keep[2, 3] = False
    anomalies = (data.values - data.values.mean(0))[:, keep]
    eigenvalues = np.linalg.eigvalsh(anomalies.T @ anomalies / anomalies.shape[0])[::-1]
    np.testing.assert_allclose(result["eigenvalues"][:2], eigenvalues[:2], rtol=1e-6)


def test_vertical_eof_degrees_of_freedom_use_time_autocorrelation() -> None:
    """The lag must be taken in time, not between neighbouring grid points of the stacked axis."""
    rng = np.random.default_rng(4)
    nt, nlev, nlat, nlon = 600, 4, 5, 8
    signal = np.zeros(nt)
    shocks = rng.standard_normal(nt)
    for index in range(1, nt):
        signal[index] = 0.95 * signal[index - 1] + shocks[index]
    profile = np.array([1.0, 0.5, -0.2, -0.8])
    spatial = rng.standard_normal((nlat, nlon))
    values = signal[:, None, None, None] * profile[None, :, None, None] * spatial[None, None]
    values = values + 0.1 * rng.standard_normal((nt, nlev, nlat, nlon))
    data = xr.DataArray(
        values,
        dims=("time", "level", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=nt),
            "level": [1000.0, 850.0, 700.0, 500.0],
            "lat": np.linspace(-10, 10, nlat),
            "lon": np.arange(nlon) * 45.0,
        },
    )
    dof = EOFAnalyzer(method="svd").fit(data, n_modes=2, n_harmonics=0)["degrees_of_freedom"]
    r1 = np.corrcoef(signal[1:], signal[:-1])[0, 1]
    expected = (1 - r1**2) / (1 + r1**2) * nt
    assert dof == pytest.approx(expected, rel=0.25)
    assert dof < 0.2 * nt  # the pre-fix value was ~ nt * nlat * nlon


# -------------------------------------------------------------------- Matsuno
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("branch", ["er_n", "eig_n", "wig_n"])
def test_matsuno_branches_are_roots_of_the_dispersion_cubic(branch: str, n: int) -> None:
    he = 25.0
    beta, perimeter = matsuno.beta_parameters(0.0)
    c = np.sqrt(matsuno.GRAVITY * he)
    curve = getattr(matsuno, branch)(he, n, max_wn=20, n_wn=41).iloc[:, 0]
    wavenumber = matsuno.wn2k(curve.index.values, perimeter)
    for k, cpd in zip(wavenumber, curve.values):
        if not np.isfinite(cpd):
            continue
        omega = cpd / matsuno.ANGULAR_TO_CPD
        scale = abs(c**2 * k**2 + (2 * n + 1) * beta * c) * omega + abs(beta * c**2 * k) + 1e-30
        residual = min(
            abs(matsuno.dispersion(omega, k, n, he, beta)),
            abs(matsuno.dispersion(-omega, k, n, he, beta)),
        )
        assert residual / scale < 1e-9


# --------------------------------------------------------------- cross spectrum
@pytest.mark.parametrize("k", [3, -3])
def test_cross_spectrum_reads_back_known_phase_and_coherence(k: int) -> None:
    """y lagging x by a quarter period gives phase -pi/2; y = -x gives pi; identical gives 0."""
    rng = np.random.default_rng(5)
    nt, nlon, period = 480, 36, 12.0
    lat = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    lon = np.arange(nlon) * 10.0
    t = np.arange(nt)[:, None, None]
    envelope = np.exp(-((lat / 8.0) ** 2))[None, :, None]

    def wave(lag: float = 0.0, sign: float = 1.0) -> np.ndarray:
        return sign * envelope * np.cos(2 * np.pi * (k * lon[None, None, :] / 360 - (t - lag) / period))

    def field(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": pd.date_range("2000-01-01", periods=nt), "lat": lat, "lon": lon},
        )

    x = field(wave() + 0.2 * rng.standard_normal((nt, lat.size, nlon)))
    cases = {
        "identical": (field(wave()), 0.0),
        "lag": (field(wave(lag=period / 4) + 0.2 * rng.standard_normal((nt, lat.size, nlon))), -np.pi / 2),
        "negative": (field(wave(sign=-1.0) + 0.2 * rng.standard_normal((nt, lat.size, nlon))), np.pi),
    }
    for name, (y, expected_phase) in cases.items():
        spectra = calculate_cross_spectrum(x, y, segLen=96, segOverLap=-60, symmetry="symm")["STC"]
        power = spectra.sel(component="PX").values
        i, j = np.unravel_index(np.nanargmax(power), power.shape)
        frequency, wavenumber = float(spectra.frequency[i]), float(spectra.wavenumber[j])
        assert frequency == pytest.approx(1 / period, abs=1 / 96)
        assert wavenumber == k, name
        coherence = float(spectra.sel(component="COH2").isel(frequency=i, wavenumber=j))
        phase = float(spectra.sel(component="PHAS").isel(frequency=i, wavenumber=j))
        assert coherence > 0.97, name
        difference = np.angle(np.exp(1j * (phase - expected_phase)))  # wrap to [-pi, pi]
        assert abs(difference) < 0.1, name
