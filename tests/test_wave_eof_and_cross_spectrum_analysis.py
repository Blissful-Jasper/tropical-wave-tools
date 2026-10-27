"""Wave-aware EOFs (atlas) and the multi-experiment cross-spectrum workflow against references."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats as scipy_stats

matplotlib.use("Agg")

from tropical_wave_tools import atlas
from tropical_wave_tools import cross_spectrum_analysis as csa
from tropical_wave_tools.cross_spectrum import (
    _get_symm_asymm,
    calculate_cross_spectrum,
    remove_annual_cycle,
)

# --------------------------------------------------------------------------- wave EOF
LAT = np.arange(-20.0, 20.1, 2.5)
LON = np.arange(0.0, 360.0, 10.0)
NT = 800


def _olr(values: np.ndarray, lat: np.ndarray = LAT) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=values.shape[0]), "lat": lat, "lon": LON},
        name="olr",
        attrs={"units": "W m-2"},
    )


@pytest.fixture(scope="module")
def two_component_field() -> dict[str, object]:
    rng = np.random.default_rng(7)
    lat, lon = LAT[:, None], np.deg2rad(LON)[None, :]
    symmetric = np.exp(-((lat / 9.0) ** 2)) * np.cos(2 * lon)
    antisymmetric = np.sin(np.deg2rad(lat) * 8) * np.exp(-((lat / 12.0) ** 2)) * np.cos(3 * lon + 1)
    a, b = 3.0 * rng.standard_normal(NT), 2.5 * rng.standard_normal(NT)
    values = a[:, None, None] * symmetric + b[:, None, None] * antisymmetric
    values = values + 0.05 * rng.standard_normal(values.shape)
    return {"data": _olr(values), "values": values, "S": symmetric, "A": antisymmetric}


def _abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    return abs(float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1]))


@pytest.mark.parametrize(("wave", "target", "other"), [("kelvin", "S", "A"), ("mrg", "A", "S")])
def test_wave_eof_isolates_the_symmetric_or_antisymmetric_component(two_component_field, wave, target, other) -> None:
    data = two_component_field["data"]
    eofs, _, explained, analysed = atlas.compute_wave_eof(data, wave_name=wave, n_modes=2)
    selected = np.isin(LAT, analysed["lat"].values)
    pattern = eofs.sel(mode=1).transpose("lat", "lon").values
    assert _abs_corr(pattern, two_component_field[target][selected]) > 0.9999
    assert _abs_corr(pattern, two_component_field[other][selected]) < 1e-3

    # explained variance = eigenvalues of the sqrt(cos)-weighted, projected covariance
    lat_selected = analysed["lat"].values
    values = data.sel(lat=lat_selected).values
    mirrored = values[:, ::-1, :]
    component = 0.5 * (values + mirrored) if wave == "kelvin" else 0.5 * (values - mirrored)
    weighted = component * np.sqrt(np.cos(np.deg2rad(lat_selected)))[None, :, None]
    flat = (weighted - weighted.mean(0)).reshape(NT, -1)
    eigenvalues = np.linalg.eigvalsh(flat.T @ flat / NT)[::-1]
    np.testing.assert_allclose(explained[:2], 100 * eigenvalues[:2] / eigenvalues.sum(), rtol=1e-6, atol=1e-8)


def test_wave_eof_patterns_are_unit_norm_in_the_weighted_space(two_component_field) -> None:
    eofs, _, _, analysed = atlas.compute_wave_eof(two_component_field["data"], wave_name="kelvin", n_modes=2)
    weights = np.sqrt(np.cos(np.deg2rad(analysed["lat"].values)))[None, :, None]
    weighted = eofs.transpose("mode", "lat", "lon").values * weights
    np.testing.assert_allclose(np.linalg.norm(weighted.reshape(2, -1), axis=1), 1.0, rtol=1e-9)


def test_wave_eof_accepts_descending_latitude(two_component_field) -> None:
    values = two_component_field["values"]
    ascending = atlas.compute_wave_eof(two_component_field["data"], wave_name="kelvin", n_modes=2)
    descending = atlas.compute_wave_eof(_olr(values[:, ::-1, :], LAT[::-1]), wave_name="kelvin", n_modes=2)
    np.testing.assert_allclose(ascending[2], descending[2], rtol=1e-8, atol=1e-10)
    pattern_a = ascending[0].sel(mode=1).transpose("lat", "lon").values
    pattern_d = descending[0].sel(mode=1).sortby("lat").transpose("lat", "lon").values
    assert _abs_corr(pattern_a, pattern_d) > 0.99999


def test_wave_eof_signs_are_normalised_and_the_pc_follows_the_data(two_component_field) -> None:
    data = two_component_field["data"]
    eofs, pcs, _, _ = atlas.compute_wave_eof(data, wave_name="kelvin", n_modes=2)
    negated_eofs, negated_pcs, _, _ = atlas.compute_wave_eof(-data, wave_name="kelvin", n_modes=2)
    for mode in (1, 2):
        loading = eofs.sel(mode=mode).values.ravel()
        assert loading[np.nanargmax(np.abs(loading))] > 0.0  # largest loading is positive
    np.testing.assert_allclose(eofs.values, negated_eofs.values, atol=1e-9)  # same EOF for -x ...
    np.testing.assert_allclose(pcs.sel(mode=1).values, -negated_pcs.sel(mode=1).values, atol=1e-9)  # ... PC flips


def test_regression_onto_pcs_has_the_right_slope_and_uses_pairwise_complete_days(two_component_field) -> None:
    _, pcs, _, _ = atlas.compute_wave_eof(two_component_field["data"], wave_name="kelvin", n_modes=1)
    pc = pcs.sel(mode=1)
    z = ((pc - pc.mean()) / float(pc.std())).values
    rng = np.random.default_rng(1)
    field = xr.DataArray(
        np.stack([2.0 * z, -1.5 * z + 0.2 * rng.standard_normal(NT)], axis=1)[:, :, None],
        dims=("time", "lat", "lon"),
        coords={"time": pc["time"].values, "lat": [0.0, 5.0], "lon": [0.0]},
    )
    standardised = atlas.regress_field_onto_pcs(field, pcs, standardize_pc=True).sel(mode=1)
    raw = atlas.regress_field_onto_pcs(field, pcs, standardize_pc=False).sel(mode=1)
    assert float(standardised.isel(lat=0, lon=0)) == pytest.approx(2.0, rel=1e-9)
    np.testing.assert_allclose(raw.values * float(pc.std()), standardised.values, rtol=1e-9)

    gappy = field.copy()
    gappy[::7, 1, 0] = np.nan
    gappy[100:180, 1, 0] = np.nan
    valid = np.isfinite(gappy.values[:, 1, 0])
    expected = scipy_stats.linregress(z[valid], gappy.values[valid, 1, 0]).slope
    result = atlas.regress_field_onto_pcs(gappy, pcs, standardize_pc=True).sel(mode=1).isel(lat=1, lon=0)
    assert float(result) == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------ cross-spectrum workflow
def _cube(values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2001-01-01", periods=values.shape[0]), "lat": lat, "lon": lon},
        name="x",
    )


def test_float_masks_with_nan_are_honoured() -> None:
    """A 1/NaN float mask used to be ignored because ``NaN.astype(bool)`` is True."""
    rng = np.random.default_rng(3)
    lat, lon, nt = np.arange(-10.0, 10.1, 5.0), np.arange(0.0, 360.0, 15.0), 365
    x = _cube(rng.standard_normal((nt, lat.size, lon.size)), lat, lon)
    y = _cube(rng.standard_normal((nt, lat.size, lon.size)), lat, lon)
    keep = np.abs(lon)[None, :] * np.ones((lat.size, 1)) < 100
    float_mask = xr.DataArray(np.where(keep, 1.0, np.nan), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    zero_mask = xr.DataArray(np.where(keep, 1.0, 0.0), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})

    results = {
        name: csa.preprocess_data_with_mask(x, y, mask=mask, remove_annual=False, verbose=False)[0]
        for name, mask in (("bool", float_mask.notnull()), ("nan", float_mask), ("zero", zero_mask))
    }
    masked_points = int((~keep).sum()) * nt
    for name, result in results.items():
        assert int((result.values == 0).sum()) == masked_points, name
        np.testing.assert_array_equal(result.values, results["bool"].values)


def test_preprocessing_removes_the_seasonal_cycle() -> None:
    rng = np.random.default_rng(4)
    lat, lon, nt = np.arange(-10.0, 10.1, 5.0), np.arange(0.0, 360.0, 30.0), 3 * 365
    time = pd.date_range("2001-01-01", periods=nt)
    seasonal = (10 * np.cos(2 * np.pi * time.dayofyear.values / 365.25))[:, None, None] * np.ones((1, lat.size, lon.size))
    x = _cube(seasonal + rng.standard_normal(seasonal.shape), lat, lon)
    cleaned, _ = csa.preprocess_data_with_mask(x, x, verbose=False)
    daily_mean = cleaned.groupby("time.dayofyear").mean()
    assert float(daily_mean.std()) < 0.01 * float(x.groupby("time.dayofyear").mean().std())


def test_low_frequency_filter_keeps_waves_and_removes_the_annual_cycle() -> None:
    t = np.arange(3650)[:, None]
    gains = {}
    for period in (365.0, 150.0, 60.0, 30.0):
        signal = np.sin(2 * np.pi * t / period)
        filtered = remove_annual_cycle(signal)
        gains[period] = np.std(filtered[400:-400]) / np.std(signal[400:-400])
    # the default cut-off is 3 cycles per year, i.e. periods longer than ~122 days are removed
    assert gains[365.0] < 0.05
    assert gains[150.0] < 0.05
    assert gains[60.0] > 0.99
    assert gains[30.0] > 0.99


def test_cross_spectrum_requires_a_symmetric_latitude_grid_and_orders_it() -> None:
    rng = np.random.default_rng(5)
    lon = np.arange(12) * 30.0
    asymmetric = _cube(rng.standard_normal((200, 5, 12)), np.array([-5.0, 0.0, 5.0, 10.0, 15.0]), lon)
    with pytest.raises(ValueError, match="paired about the equator"):
        calculate_cross_spectrum(asymmetric, asymmetric, segLen=64, segOverLap=-32)

    lat = np.arange(-10.0, 10.1, 5.0)
    x = _cube(rng.standard_normal((300, lat.size, 12)), lat, lon)
    y = _cube(0.5 * x.values + rng.standard_normal(x.shape), lat, lon)
    for symmetry in ("symm", "asymm"):
        ascending = calculate_cross_spectrum(x, y, segLen=64, segOverLap=-32, symmetry=symmetry)["STC"]
        flipped = calculate_cross_spectrum(x.isel(lat=slice(None, None, -1)), y.isel(lat=slice(None, None, -1)), segLen=64, segOverLap=-32, symmetry=symmetry)["STC"]
        np.testing.assert_allclose(ascending.values, flipped.values, rtol=1e-10, equal_nan=True)


def test_symmetric_and_antisymmetric_components_pair_mirror_latitudes() -> None:
    lat = np.arange(-10.0, 10.1, 5.0)
    rows = np.repeat(np.arange(5.0)[None, :, None], 2, axis=2)
    assert _get_symm_asymm(rows, lat, "symm")[0, :, 0].tolist() == [2.0, 2.0, 2.0]
    assert _get_symm_asymm(rows, lat, "asymm")[0, :, 0].tolist() == [-2.0, -1.0]  # 0.5 * (south - north)


def test_coherence_squared_is_bounded_and_dof_is_configurable() -> None:
    rng = np.random.default_rng(6)
    lat, lon = np.arange(-10.0, 10.1, 5.0), np.arange(0.0, 360.0, 15.0)
    x = _cube(rng.standard_normal((900, lat.size, lon.size)), lat, lon)
    y = _cube(0.6 * x.values + rng.standard_normal(x.shape), lat, lon)
    default = calculate_cross_spectrum(x, y, segLen=96, segOverLap=-65)
    coherence = default["STC"].sel(component="COH2").values
    assert np.nanmin(coherence) >= 0.0 and np.nanmax(coherence) <= 1.0
    assert default["dof"] == pytest.approx(2.667 * default["nseg"])

    fewer = calculate_cross_spectrum(x, y, segLen=96, segOverLap=-65, dof_per_segment=2.0)
    assert fewer["dof"] == pytest.approx(2.0 * default["nseg"])
    assert np.all(fewer["prob_coh2"] > default["prob_coh2"])  # fewer degrees of freedom -> higher thresholds


def test_coherence_significance_depends_on_meridional_coherence() -> None:
    """The fixed ``dof = 2.667 * nseg`` is liberal for perfectly coherent latitudes and
    conservative for independent ones (documented in ``calculate_cross_spectrum``)."""
    rng = np.random.default_rng(8)
    nt, nlon = 1200, 24
    lat, lon = np.linspace(-10, 10, 5), np.arange(nlon) * 15.0

    def exceedance(identical_rows: bool) -> float:
        hits = total = 0
        for _ in range(6):
            if identical_rows:
                x = np.repeat(rng.standard_normal((nt, 1, nlon)), lat.size, axis=1)
                y = np.repeat(rng.standard_normal((nt, 1, nlon)), lat.size, axis=1)
            else:
                x, y = rng.standard_normal((nt, lat.size, nlon)), rng.standard_normal((nt, lat.size, nlon))
            result = calculate_cross_spectrum(_cube(x, lat, lon), _cube(y, lat, lon), segLen=96, segOverLap=-65)
            coherence = result["STC"].sel(component="COH2").isel(frequency=slice(1, -1)).values
            coherence = coherence[np.isfinite(coherence)]
            hits += int((coherence >= result["prob_coh2"][4]).sum())  # the 95% level: nominal 5%
            total += coherence.size
        return hits / total

    assert exceedance(identical_rows=True) > 0.06
    assert exceedance(identical_rows=False) < 0.02


def test_workflow_recovers_known_coherence_and_phase_and_plots(tmp_path) -> None:
    rng = np.random.default_rng(9)
    lat, lon, nt, period, k = np.array([-10.0, -5.0, 0.0, 5.0, 10.0]), np.arange(36) * 10.0, 480, 12.0, 3
    t = np.arange(nt)[:, None, None]
    envelope = np.exp(-((lat / 8.0) ** 2))[None, :, None]

    def wave(lag: float) -> np.ndarray:
        return envelope * np.cos(2 * np.pi * (k * lon[None, None, :] / 360 - (t - lag) / period))

    x = _cube(wave(0.0) + 0.2 * rng.standard_normal((nt, lat.size, lon.size)), lat, lon)
    y = _cube(wave(period / 4) + 0.2 * rng.standard_normal((nt, lat.size, lon.size)), lat, lon)
    results = csa.compute_cross_spectrum_for_experiments(
        {"exp": x}, {"exp": y}, experiments=["exp"], seg_length=96, seg_overlap=-60, verbose=False
    )
    spectra = results["exp"]["STC"]
    power = spectra.sel(component="PX").values
    i, j = np.unravel_index(np.nanargmax(power), power.shape)
    assert float(spectra.frequency[i]) == pytest.approx(1 / period, abs=1 / 96)
    assert float(spectra.wavenumber[j]) == k
    assert float(spectra.sel(component="COH2").isel(frequency=i, wavenumber=j)) > 0.97
    phase = float(spectra.sel(component="PHAS").isel(frequency=i, wavenumber=j))
    assert abs(np.angle(np.exp(1j * (phase + np.pi / 2)))) < 0.1  # y lags x by a quarter period

    _figure, axes = csa.plot_cross_spectrum_panel(
        results, experiments=["exp"], output_path=tmp_path / "panel.png", dpi=50, verbose=False
    )
    assert (tmp_path / "panel.png").exists() and len(axes) == 1
