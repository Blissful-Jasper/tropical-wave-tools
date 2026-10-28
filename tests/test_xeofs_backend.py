"""The xeofs backend of ``EOFAnalyzer`` against the SVD backend and an independent eigendecomposition.

xeofs is an optional dependency (``pip install "tropical-wave-tools[advanced]"``); without it these
tests are skipped. xeofs itself refuses any NaN, so most of the checks are about the wrapper: it must
drop the missing data like the SVD backend does and return the same results.
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

pytest.importorskip("xeofs")

import matplotlib.pyplot as plt

from tropical_wave_tools import eof as eof_module
from tropical_wave_tools.eof import EOFAnalyzer

N_MODES = 3


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _horizontal_field(n_time=300, seed=0):
    """Two orthogonal spatial modes with well separated variances plus weak noise."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-15.0, 15.0, 7)
    lon = np.arange(12) * 30.0
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    mode_1 = np.cos(np.deg2rad(lat_grid) * 4) * np.cos(np.deg2rad(lon_grid))
    mode_2 = np.sin(np.deg2rad(lat_grid) * 5) * np.sin(2 * np.deg2rad(lon_grid))
    values = (
        rng.standard_normal(n_time)[:, None, None] * 3.0 * mode_1
        + rng.standard_normal(n_time)[:, None, None] * 1.2 * mode_2
        + 0.1 * rng.standard_normal((n_time, lat.size, lon.size))
    )
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=n_time), "lat": lat, "lon": lon},
    )


def _vertical_field(n_time=120, seed=1):
    rng = np.random.default_rng(seed)
    level = np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0])
    profile_1 = np.sin(np.linspace(0.0, np.pi, level.size))
    profile_2 = np.cos(np.linspace(0.0, 2.0 * np.pi, level.size))
    lat, lon = np.array([-5.0, 0.0, 5.0]), np.array([0.0, 90.0, 180.0, 270.0])
    amplitude_1 = rng.standard_normal((n_time, lat.size, lon.size)) * 2.0
    amplitude_2 = rng.standard_normal((n_time, lat.size, lon.size)) * 0.8
    values = (
        amplitude_1[:, None] * profile_1[None, :, None, None]
        + amplitude_2[:, None] * profile_2[None, :, None, None]
        + 0.05 * rng.standard_normal((n_time, level.size, lat.size, lon.size))
    )
    return xr.DataArray(
        values,
        dims=("time", "plev", "lat", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=n_time), "plev": level, "lat": lat, "lon": lon},
    )


def _reference_horizontal(field, n_modes=N_MODES):
    """Eigendecomposition of the covariance of the grid points after removing each point's time mean."""
    flat = field.values.reshape(field.sizes["time"], -1)
    flat = flat - flat.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(flat.T @ flat / flat.shape[0])
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order] / eigenvalues.sum() * 100.0, vectors[:, order][:, :n_modes], flat @ vectors[:, order][:, :n_modes]


def _reference_vertical(field, n_modes=N_MODES):
    """Same for levels, with each (level, lat, lon) point's time mean removed (what the analyzer does)."""
    anomalies = field - field.mean("time")
    matrix = anomalies.transpose("plev", "time", "lat", "lon").values.reshape(field.sizes["plev"], -1)
    eigenvalues, vectors = np.linalg.eigh(matrix @ matrix.T / matrix.shape[1])
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order] / eigenvalues.sum() * 100.0, vectors[:, order][:, :n_modes]


def _align(reference, candidate):
    """Candidate with the sign that makes it point the same way as the reference."""
    sign = np.sign(np.sum(reference * candidate))
    return candidate * (sign if sign != 0 else 1.0)


def _fit(method, field, **kwargs):
    kwargs.setdefault("n_harmonics", 0)
    kwargs.setdefault("n_modes", N_MODES)
    return EOFAnalyzer(method=method).fit(field, **kwargs)


# ============================================================================== clean data
def test_horizontal_eof_matches_the_covariance_eigendecomposition() -> None:
    field = _horizontal_field()
    explained, vectors, pcs = _reference_horizontal(field)
    for method in ("svd", "xeofs"):
        result = _fit(method, field)
        assert result["eofs"].dims == ("mode", "lat", "lon") and result["pc_scores"].dims == ("mode", "time")
        np.testing.assert_allclose(np.asarray(result["explained_variance"])[:2], explained[:2], rtol=1e-8, err_msg=method)
        for mode in range(2):  # the third mode is noise, nearly degenerate with its neighbours
            pattern = result["eofs"].values[mode].ravel()
            assert np.linalg.norm(pattern) == pytest.approx(1.0)
            np.testing.assert_allclose(_align(vectors[:, mode], pattern), vectors[:, mode], atol=1e-7, err_msg=method)
            np.testing.assert_allclose(
                _align(pcs[:, mode], result["pc_scores"].values[mode]), pcs[:, mode], atol=1e-6, err_msg=method
            )


def test_vertical_eof_matches_the_covariance_eigendecomposition() -> None:
    field = _vertical_field()
    explained, vectors = _reference_vertical(field)
    for method in ("svd", "xeofs"):
        result = _fit(method, field)
        assert result["eofs"].dims == ("mode", "plev")
        assert result["pc_scores"].dims == ("mode", "time", "lat", "lon")
        np.testing.assert_allclose(np.asarray(result["explained_variance"])[:2], explained[:2], rtol=1e-8, err_msg=method)
        for mode in range(2):
            np.testing.assert_allclose(
                _align(vectors[:, mode], result["eofs"].values[mode]), vectors[:, mode], atol=1e-7, err_msg=method
            )


@pytest.mark.parametrize("n_harmonics", [0, 3])
def test_backends_agree_with_each_other(n_harmonics: int) -> None:
    """Including with the harmonic seasonal cycle removed (the anomalies are then not exactly centred)."""
    time = pd.date_range("2000-01-01", periods=3 * 365)
    field = _horizontal_field(n_time=time.size, seed=4)
    seasonal = 6.0 * np.cos(2 * np.pi * time.dayofyear.values / 365.25)[:, None, None] * np.sin(np.deg2rad(field["lat"].values))[None, :, None]
    field = field + seasonal
    svd = _fit("svd", field, n_harmonics=n_harmonics)
    xe = _fit("xeofs", field, n_harmonics=n_harmonics)
    np.testing.assert_allclose(np.asarray(xe["explained_variance"])[:2], np.asarray(svd["explained_variance"])[:2], rtol=1e-8)
    np.testing.assert_allclose(xe["eigenvalues"][:2], svd["eigenvalues"][:2], rtol=1e-8)
    for mode in range(2):
        np.testing.assert_allclose(
            _align(svd["eofs"].values[mode], xe["eofs"].values[mode]), svd["eofs"].values[mode], atol=1e-8
        )
        np.testing.assert_allclose(
            _align(svd["pc_scores"].values[mode], xe["pc_scores"].values[mode]), svd["pc_scores"].values[mode], atol=1e-7
        )


def test_xeofs_results_carry_the_same_fields_as_the_svd_results() -> None:
    """The xeofs results used to hold None for the eigenvalues, their North errors and the masks."""
    field = _horizontal_field()
    svd, xe = _fit("svd", field), _fit("xeofs", field)
    assert set(svd) == set(xe)
    for key in ("eigenvalues", "eigenvalue_errors", "valid_mask", "valid_feature_mask", "phi_0", "phi_L"):
        assert xe[key] is not None, key
    assert xe["degrees_of_freedom"] == pytest.approx(svd["degrees_of_freedom"])
    np.testing.assert_allclose(xe["eigenvalue_errors"][:2], svd["eigenvalue_errors"][:2], rtol=1e-8)
    np.testing.assert_array_equal(xe["valid_mask"], svd["valid_mask"])
    np.testing.assert_allclose(
        xe["eigenvalue_errors"][:2], np.asarray(xe["explained_variance"])[:2] * np.sqrt(2.0 / xe["degrees_of_freedom"])
    )  # North et al. (1982)
    assert xe["method"] == "xeofs" and svd["method"] == "svd"


def test_requesting_more_modes_than_the_data_can_give_is_clipped() -> None:
    field = _vertical_field()  # 8 levels
    result = _fit("xeofs", field, n_modes=20)
    assert result["eofs"].sizes["mode"] == field.sizes["plev"]


def test_randomized_solver_path_reproduces_the_leading_modes() -> None:
    """For large problems xeofs switches to a randomized SVD; the leading, separated modes must stay exact."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(3)
    n_time, n_lat, n_lon, n_modes = 600, 20, 36, 4  # max(shape) >= 500 triggers the randomized solver
    modes = np.stack([gaussian_filter(rng.standard_normal((n_lat, n_lon)), (2, 4), mode="wrap") for _ in range(30)])
    amplitudes = rng.standard_normal((n_time, 30)) / np.arange(1, 31) ** 0.7
    values = np.tensordot(amplitudes, modes, axes=(1, 0)) + 0.02 * rng.standard_normal((n_time, n_lat, n_lon))
    field = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=n_time), "lat": np.linspace(-30, 30, n_lat), "lon": np.arange(n_lon) * 10.0},
    )
    svd, xe = _fit("svd", field, n_modes=n_modes), _fit("xeofs", field, n_modes=n_modes)
    np.testing.assert_allclose(np.asarray(xe["explained_variance"])[:n_modes], np.asarray(svd["explained_variance"])[:n_modes], rtol=1e-5)
    for mode in range(n_modes):
        assert abs(np.corrcoef(xe["eofs"].values[mode].ravel(), svd["eofs"].values[mode].ravel())[0, 1]) > 0.9999


# ============================================================================== missing data
def _masked_cases():
    field = _horizontal_field(n_time=300, seed=2)
    masked_points = field.copy()
    masked_points[:, 2, 3] = np.nan
    masked_points[:, 5, 8] = np.nan
    missing_times = field.copy()
    missing_times[10:20] = np.nan
    one_value = field.copy()
    one_value[50, 3, 3] = np.nan
    return {"masked points": masked_points, "missing time steps": missing_times, "one missing value": one_value}


@pytest.mark.parametrize("case", ["masked points", "missing time steps", "one missing value"])
def test_horizontal_eof_with_missing_data_matches_the_complete_subset(case: str) -> None:
    """Drop the all-missing times, then the grid points that still have a gap; the rest is a clean EOF."""
    field = _masked_cases()[case]
    flat = field.values.reshape(field.sizes["time"], -1)
    flat = flat[~np.isnan(flat).all(axis=1)]
    flat = flat[:, ~np.isnan(flat).any(axis=0)]
    flat = flat - flat.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(flat.T @ flat / flat.shape[0])
    order = np.argsort(eigenvalues)[::-1]
    explained = eigenvalues[order] / eigenvalues.sum() * 100.0
    vectors = vectors[:, order]
    results = {method: _fit(method, field) for method in ("svd", "xeofs")}  # xeofs alone raises "Input contains NaN"
    keep = ~np.isnan(results["svd"]["eofs"].values[0].ravel())
    for method, result in results.items():
        np.testing.assert_allclose(np.asarray(result["explained_variance"])[:2], explained[:2], rtol=1e-6, err_msg=method)
        pattern = result["eofs"].values[0].ravel()
        np.testing.assert_array_equal(np.isnan(pattern), ~keep)  # dropped points come back as NaN
        np.testing.assert_allclose(_align(vectors[:, 0], pattern[keep]), vectors[:, 0], atol=1e-6, err_msg=method)
    np.testing.assert_array_equal(np.isnan(results["xeofs"]["pc_scores"].values), np.isnan(results["svd"]["pc_scores"].values))
    assert (~results["xeofs"]["valid_mask"]).sum() == (10 if case == "missing time steps" else 0)


def test_vertical_eof_drops_samples_with_a_missing_level_like_the_svd_backend() -> None:
    """Levels below the ground are NaN at some points; those samples leave the decomposition."""
    field = _vertical_field()
    below_ground = field.copy()
    below_ground[:, 0:2, 1, 2] = np.nan  # two lowest levels missing at one point, at every time
    whole_steps = field.copy()
    whole_steps[5:8] = np.nan
    for gappy in (below_ground, whole_steps):
        anomalies = gappy - gappy.mean("time")
        matrix = anomalies.transpose("plev", "time", "lat", "lon").values.reshape(gappy.sizes["plev"], -1)
        matrix = matrix[:, ~np.isnan(matrix).any(axis=0)]
        eigenvalues, vectors = np.linalg.eigh(matrix @ matrix.T / matrix.shape[1])
        order = np.argsort(eigenvalues)[::-1]
        explained = eigenvalues[order] / eigenvalues.sum() * 100.0
        for method in ("svd", "xeofs"):
            result = _fit(method, gappy)
            np.testing.assert_allclose(np.asarray(result["explained_variance"])[:2], explained[:2], rtol=1e-6, err_msg=method)
            np.testing.assert_allclose(_align(vectors[:, order[0]], result["eofs"].values[0]), vectors[:, order[0]], atol=1e-6, err_msg=method)
            assert np.isnan(result["pc_scores"].values).any()  # the dropped samples have no PC value


def test_all_missing_input_raises_a_clear_error() -> None:
    field = _horizontal_field() * np.nan
    for method in ("svd", "xeofs"):
        with pytest.raises(ValueError, match="No valid data"):
            _fit(method, field)


# ====================================================================== functional vertical API
def _functional_input(field):
    return field.rename({"plev": "level"})


def test_vertical_eof_with_nan_handling_handles_levels_below_ground() -> None:
    field = _functional_input(_vertical_field())
    gappy = field.copy()
    gappy[:, 0:2, 1, 2] = np.nan
    eofs, scores, explained = eof_module.vertical_eof_with_nan_handling(gappy, n_modes=2)
    assert eofs.dims == ("mode", "level") and scores.dims == ("mode", "time", "lat", "lon")
    assert not np.isnan(eofs.values).any() and explained.shape == (2,)
    # reference: remove the mean of each level over the complete samples, then the covariance eigenvectors
    matrix = gappy.transpose("level", "time", "lat", "lon").values.reshape(field.sizes["level"], -1)
    matrix = matrix[:, ~np.isnan(matrix).any(axis=0)]
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    eigenvalues, vectors = np.linalg.eigh(matrix @ matrix.T / matrix.shape[1])
    order = np.argsort(eigenvalues)[::-1]
    np.testing.assert_allclose(_align(vectors[:, order[0]], eofs.values[0]), vectors[:, order[0]], atol=1e-6)
    np.testing.assert_allclose(explained.values, eigenvalues[order][:2] / eigenvalues.sum(), rtol=1e-6)  # a fraction, not %
    # the dropped samples (the gap point, every time) carry NaN scores, the others do not
    assert np.isnan(scores.isel(mode=0, lat=1, lon=2)).all()
    assert not np.isnan(scores.isel(mode=0, lat=0, lon=0)).any()


def test_vertical_eof_with_nan_handling_on_clean_data_matches_the_level_covariance() -> None:
    """The functional API removes the mean of each *level* (xeofs ``center=True``), the analyzer that of each point."""
    field = _vertical_field()
    eofs, scores, explained = eof_module.vertical_eof_with_nan_handling(_functional_input(field), n_modes=2)
    matrix = field.transpose("plev", "time", "lat", "lon").values.reshape(field.sizes["plev"], -1)
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    eigenvalues, vectors = np.linalg.eigh(matrix @ matrix.T / matrix.shape[1])
    order = np.argsort(eigenvalues)[::-1]
    for mode in range(2):
        np.testing.assert_allclose(_align(vectors[:, order[mode]], eofs.values[mode]), vectors[:, order[mode]], atol=1e-7)
    np.testing.assert_allclose(explained.values, eigenvalues[order][:2] / eigenvalues.sum(), rtol=1e-8)
    assert not np.isnan(scores.values).any()


def test_eof_sign_alignment_and_profile_comparison_plot() -> None:
    level = np.array([1000.0, 850.0, 500.0, 200.0])
    profile = xr.DataArray([1.0, 0.5, -0.2, -0.8], dims="level", coords={"level": level})
    assert eof_module.align_eof_signs(profile, profile) == 1
    assert eof_module.align_eof_signs(profile, -profile) == -1
    modes = xr.concat([profile.expand_dims(mode=[1]), (profile[::-1].values * xr.ones_like(profile)).expand_dims(mode=[2])], dim="mode")
    variance = xr.DataArray([0.7, 0.2], dims="mode", coords={"mode": [1, 2]})
    flipped = modes * xr.DataArray([-1.0, 1.0], dims="mode", coords={"mode": [1, 2]})
    figure = eof_module.compare_vertical_eofs(
        {"reference": (modes, None, variance), "flipped": (flipped, None, variance)}, reference_key="reference"
    )
    first_panel = figure.axes[0]
    np.testing.assert_allclose(first_panel.lines[0].get_xdata(), modes.sel(mode=1).values)
    np.testing.assert_allclose(first_panel.lines[1].get_xdata(), modes.sel(mode=1).values)  # sign-aligned to the reference
    assert [t.get_text() for t in first_panel.get_legend().get_texts()] == ["reference (70.0%)", "flipped (70.0%)"]


# ===================================================================== analyzer plumbing
def test_vertical_profile_figure_and_convenience_wrapper() -> None:
    field = _vertical_field()
    analyzer, results, figure = eof_module.eof_xeofs(field, n_harmonics=0, n_modes=3)
    assert results["method"] == "xeofs" and figure is not None
    axis = figure.axes[0]
    explained = np.asarray(results["explained_variance"])
    for mode, line in enumerate(axis.lines[:3]):
        pattern = results["eofs"].values[mode]
        np.testing.assert_allclose(line.get_xdata(), pattern / np.abs(pattern).max())
        np.testing.assert_allclose(line.get_ydata(), field["plev"].values)
    assert next(t.get_text() for t in axis.get_legend().get_texts()) == f"EOF1 ({explained[0]:.1f}%)"
    assert "XEOFS" in axis.get_title()
    horizontal = _fit("xeofs", _horizontal_field())
    with pytest.raises(ValueError, match="level dimension"):
        analyzer.eof_results = horizontal
        analyzer.plot_vertical_profiles()


def test_results_survive_a_save_and_load_round_trip(tmp_path) -> None:
    field = _horizontal_field()
    analyzer = EOFAnalyzer(method="xeofs")
    results = analyzer.fit(field, n_harmonics=0, n_modes=2)
    analyzer.save_results(str(tmp_path / "eof.pkl"))
    loaded = EOFAnalyzer(method="svd").load_results(str(tmp_path / "eof.pkl"))
    np.testing.assert_array_equal(loaded["eofs"].values, results["eofs"].values)
    assert loaded["method"] == "xeofs"


def test_method_is_validated_and_a_missing_xeofs_is_reported(monkeypatch) -> None:
    with pytest.raises(ValueError, match="method must be"):
        EOFAnalyzer(method="nonsense")
    monkeypatch.setitem(sys.modules, "xeofs", None)  # makes ``import xeofs`` raise ImportError
    with pytest.raises(ImportError, match="xeofs is required"):
        _fit("xeofs", _horizontal_field())
