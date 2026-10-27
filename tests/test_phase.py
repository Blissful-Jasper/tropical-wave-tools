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



# ---------------------------------------------------------------------------
# Checks against independent references (scipy, known phase, analytic bin means)
# ---------------------------------------------------------------------------
import pandas as pd
import pytest
import xarray as xr
from scipy.signal import argrelextrema

from tropical_wave_tools import phase as phase_module


def _time_lon(values: np.ndarray, lon: np.ndarray | None = None) -> xr.DataArray:
    nt, nlon = values.shape
    lon = np.arange(nlon) * 10.0 if lon is None else lon
    return xr.DataArray(
        values,
        dims=("time", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=nt), "lon": lon},
    )


def _phase_of(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = _time_lon(values)
    peaks, _ = optimize_peak_detection(values, reference, float(np.std(values)), use_parallel=False)
    return calculate_kelvin_phase(reference, peaks).values, peaks.values


def test_local_extrema_match_scipy_and_treat_plateaus_as_one_extremum() -> None:
    t = np.arange(200)
    series = (np.sin(2 * np.pi * t / 20) + 0.3 * np.sin(2 * np.pi * t / 7))[:, None]
    flags = phase_module.find_local_extrema(series)[:, 0]
    assert set(np.where(flags == 1)[0]) == set(argrelextrema(series[:, 0], np.less)[0])
    assert set(np.where(flags == -1)[0]) == set(argrelextrema(series[:, 0], np.greater)[0])

    plateau = np.array([0, 1, 2, 2, 2, 1, 0, -1, -1, 0, 1.0])[:, None]
    flags = phase_module.find_local_extrema(plateau)[:, 0]
    assert np.where(flags == -1)[0].tolist() == [2]  # one maximum for the 2, 2, 2 run
    assert np.where(flags == 1)[0].tolist() == [7]  # one minimum for the -1, -1 run
    assert not np.isfinite(phase_module.find_local_extrema(np.ones((6, 1)))).any()


def test_peak_influence_ranges_match_half_wave_segmentation() -> None:
    t = np.arange(400)
    series = (np.sin(2 * np.pi * t / 23) * (1 + 0.5 * np.sin(2 * np.pi * t / 151)))[:, None]
    threshold = float(series.std())
    result = phase_module.process_single_longitude(
        0, series, phase_module.find_local_extrema(series), threshold, 1.0
    )

    reference = np.full(400, np.nan)
    sign = np.sign(series[:, 0])
    start = 0
    for index in range(1, 401):
        if index == 400 or sign[index] != sign[index - 1]:
            segment = series[start:index, 0]
            peak = segment[np.argmax(np.abs(segment))]
            if abs(peak) >= threshold:
                reference[start:index] = peak
            start = index
    np.testing.assert_array_equal(np.isnan(result), np.isnan(reference))
    np.testing.assert_allclose(result[np.isfinite(result)], reference[np.isfinite(reference)])


@pytest.mark.parametrize("period", [20, 40])
def test_kelvin_phase_recovers_the_true_phase(period: int) -> None:
    """For x = A sin(theta) the module's convention is phase = -theta, so crests sit at -pi/2."""
    t = np.arange(480)
    theta = 2 * np.pi * t / period + 0.7
    values = (3.0 * np.sin(theta))[:, None]
    phase, _ = _phase_of(values)

    defined = np.isfinite(phase[:, 0])
    assert defined.mean() > 0.95
    error = np.abs(np.angle(np.exp(1j * (phase[defined, 0] + theta[defined]))))
    assert error.max() < 0.1
    crests = argrelextrema(values[:, 0], np.greater)[0]
    crests = crests[np.isfinite(phase[crests, 0])]
    np.testing.assert_allclose(phase[crests, 0], -np.pi / 2, atol=1e-9)


def test_kelvin_phase_tracks_a_growing_modulated_envelope() -> None:
    t = np.arange(480)
    theta = 2 * np.pi * t / 20
    envelope = np.exp(0.004 * t) * (1 + 0.3 * np.sin(2 * np.pi * t / 160))
    phase, _ = _phase_of((envelope * np.sin(theta))[:, None])
    defined = np.isfinite(phase[:, 0])
    assert np.abs(np.angle(np.exp(1j * (phase[defined, 0] + theta[defined])))).max() < 0.1


def test_phase_composite_bins_are_equal_width_and_merge_plus_minus_pi() -> None:
    """Uniform phases give equal counts, and the mean of cos(phase) equals the analytic bin mean."""
    phases = np.linspace(-np.pi, np.pi, 24001)  # includes both -pi and +pi
    data = _time_lon(np.cos(phases)[:, None])
    phase = _time_lon(phases[:, None])
    for n_bins in (8, 12):
        centres, means, counts = phase_composite(data, phase, n_bins=n_bins)
        width = 2 * np.pi / n_bins
        np.testing.assert_allclose(np.diff(centres), width)
        np.testing.assert_allclose(centres[0], -np.pi + width / 2)
        assert counts.max() - counts.min() <= 2  # no half-populated end bins
        analytic = (np.sin(centres + width / 2) - np.sin(centres - width / 2)) / width
        np.testing.assert_allclose(means, analytic, atol=2e-3)


def test_lag_composite_handles_off_grid_reference_and_a_single_event() -> None:
    nt, lon = 200, np.arange(0, 360, 10.0)
    t = np.arange(nt)[:, None]
    values = np.sin(2 * np.pi * (t / 20 - 3 * lon[None, :] / 360))
    phase, _ = _phase_of(values)
    data, phase_da = _time_lon(values, lon), _time_lon(phase, lon)
    ref_col = int(np.argmin(np.abs(lon - 180.0)))

    lags, composite, events = phase_module.lag_composite(data, phase_da, lon, lon_ref=181.0, nlag=5)
    assert events.size >= 5
    np.testing.assert_allclose(composite[5, ref_col], 1.0, atol=1e-6)  # lag 0 = crest at lon_ref
    assert lags.tolist() == list(range(-5, 6))

    single = phase.copy()
    single[:, ref_col] = np.nan
    single[100, ref_col] = -np.pi / 2
    _, composite, events = phase_module.lag_composite(data, _time_lon(single, lon), lon, lon_ref=180.0, nlag=5)
    assert events.tolist() == [100]
    assert np.isfinite(composite[5]).any()

    with pytest.raises(ValueError, match="lon_ref"):
        phase_module.lag_composite(data, phase_da, lon, lon_ref=500.0, nlag=5)


def test_meridional_projection_is_a_nan_aware_weighted_mean() -> None:
    rng = np.random.default_rng(6)
    lat = np.arange(-20, 20.1, 2.5)
    nt, nlon = 30, 6
    field = rng.standard_normal((nt, lat.size, nlon))
    field[3, 2, 1] = np.nan
    field[5, :, 2] = np.nan
    data = xr.DataArray(
        field,
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=nt), "lat": lat, "lon": np.arange(nlon) * 10.0},
    )
    projected = np.asarray(meridional_projection(data, lat).values, dtype=float)

    weights = np.exp(-((lat / 18.0) ** 2)) * (np.abs(lat) <= 10)
    expected = np.full((nt, nlon), np.nan)
    for i in range(nt):
        for j in range(nlon):
            usable = weights * np.isfinite(field[i, :, j])
            if usable.sum() > 0:
                expected[i, j] = np.nansum(field[i, :, j] * usable) / usable.sum()
    np.testing.assert_allclose(projected, expected, equal_nan=True)


def test_low_pass_removes_sub_ten_day_signals() -> None:
    t = np.arange(600)
    gains = {}
    for period in (30, 5):
        signal = np.sin(2 * np.pi * t / period)[:, None, None]
        filtered = phase_module.remove_10d_from_daily_data(signal)[:, 0, 0]
        gains[period] = np.std(filtered[100:-100]) / np.std(signal[100:-100])
    assert gains[30] > 0.99
    assert gains[5] < 0.01


def test_composite_workflow_recovers_the_phase_speed_of_a_wave_packet(tmp_path) -> None:
    """Localised eastward packets with c = 15 m/s: the fitted Cp_ave must be ~15 m/s."""
    import os

    lat = np.arange(-15, 15.1, 2.5)
    lon = np.arange(72) * 5.0
    nt, speed, wavenumber = 1200, 15.0, 2
    speed_deg_day = speed * 86400 / 111000.0
    period = 360.0 / (wavenumber * speed_deg_day)
    t = np.arange(nt)[:, None, None]
    lo = lon[None, None, :]
    field = np.zeros((nt, lat.size, lon.size))
    for centre in range(60, nt - 60, 45):
        offset = ((lo - 180.0 - speed_deg_day * (t - centre) + 180) % 360) - 180
        field += (
            np.cos(2 * np.pi * (wavenumber * lo / 360 - (t - centre) / period))
            * np.exp(-((offset / 40.0) ** 2))
            * np.exp(-(((t - centre) / 12.0) ** 2))
            * np.exp(-((lat / 8.0) ** 2))[None, :, None]
        )
    kelvin = 6.0 / 86400 * field
    coords = {"time": pd.date_range("2000-01-01", periods=nt), "lat": lat, "lon": lon}
    rng = np.random.default_rng(2)
    wrap = lambda a: xr.DataArray(a, dims=("time", "lat", "lon"), coords=coords)  # noqa: E731
    precipitation = wrap(3.0 / 86400 + kelvin + 0.4 / 86400 * rng.standard_normal(kelvin.shape))

    phase_module.composite_kw_phase(wrap(kelvin), precipitation, lon, "TEST", str(tmp_path), lon_ref=180.0, nlag=10)

    speed_file = next(name for name in os.listdir(tmp_path) if "Cp_ave" in name)
    estimated = float(np.load(tmp_path / speed_file)["Cp_ave"])
    assert estimated == pytest.approx(speed, rel=0.05)


# ---------------------------------------------------------------------------
# Phase-speed estimator
# ---------------------------------------------------------------------------
def _ridge_composite(speed_ms: float, wavenumber: int, *, amplitude: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lag-longitude composite of eastward crests moving at ``speed_ms`` through lon_ref = 180."""
    lags = np.arange(-10, 11)
    lon = np.arange(72) * 5.0
    deg_per_day = speed_ms * 86400 / 111000.0
    field = amplitude * np.cos(2 * np.pi * wavenumber * (lon[None, :] - 180.0 - deg_per_day * lags[:, None]) / 360.0)
    return field, lags, lon


@pytest.mark.parametrize("speed", [8.0, 15.0, 22.0])
def test_ridge_estimator_recovers_the_speed_of_a_single_ridge(speed: float) -> None:
    composite, lags, lon = _ridge_composite(speed, wavenumber=1)
    estimate = phase_module.estimate_phase_speed(composite, lags, lon, lon_ref=180.0, threshold=1.5, method="ridge")
    assert estimate == pytest.approx(speed, rel=0.05)


@pytest.mark.parametrize(("speed", "legacy_ok"), [(8.0, True), (15.0, True), (22.0, False)])
def test_threshold_fit_is_only_valid_while_the_ridge_stays_inside_the_longitude_range(
    speed: float, legacy_ok: bool
) -> None:
    """At 22 m/s the ridge crosses 0/360 within +-10 days and the straight-line fit is cut in two."""
    composite, lags, lon = _ridge_composite(speed, wavenumber=1)
    legacy = phase_module.estimate_phase_speed(
        composite, lags, lon, lon_ref=180.0, threshold=1.5, method="threshold_fit"
    )
    assert (abs(legacy - speed) < 0.05 * speed) is legacy_ok


def test_ridge_estimator_survives_several_simultaneous_crests() -> None:
    """Wavenumber 3 gives three crests at every lag; one straight line through all of them is useless."""
    composite, lags, lon = _ridge_composite(15.0, wavenumber=3)
    ridge = phase_module.estimate_phase_speed(composite, lags, lon, lon_ref=180.0, threshold=1.5, method="ridge")
    legacy = phase_module.estimate_phase_speed(
        composite, lags, lon, lon_ref=180.0, threshold=1.5, method="threshold_fit"
    )
    assert ridge == pytest.approx(15.0, rel=0.05)
    assert abs(legacy - 15.0) > 5.0  # the historical estimator fails here (it returned -1.5 m/s)


def test_ridge_tracking_is_circular_and_stops_below_the_threshold() -> None:
    composite, lags, lon = _ridge_composite(15.0, wavenumber=1)
    ridge_lags, ridge_lons = phase_module.trace_composite_ridge(composite, lags, lon, lon_ref=180.0, threshold=1.5)
    assert ridge_lags.tolist() == lags.tolist()  # the whole lag range is followed
    assert np.all(np.diff(ridge_lons) > 0)  # unwrapped: continuous eastward drift, no 360 degree jumps
    ridge_lags, _ = phase_module.trace_composite_ridge(composite, lags, lon, lon_ref=180.0, threshold=10.0)
    assert ridge_lags.size == 0
    assert np.isnan(phase_module.estimate_phase_speed(composite, lags, lon, lon_ref=180.0, threshold=10.0))
    with pytest.raises(ValueError, match="method"):
        phase_module.estimate_phase_speed(composite, lags, lon, lon_ref=180.0, threshold=1.5, method="magic")


def test_workflow_recovers_the_speed_of_a_global_wavenumber_three_wave(tmp_path) -> None:
    """The case that gave -1.5 m/s (true +15 m/s) with the straight-line estimator."""
    import os

    lat = np.arange(-15, 15.1, 2.5)
    lon = np.arange(72) * 5.0
    nt, speed, wavenumber = 720, 15.0, 3
    speed_deg_day = speed * 86400 / 111000.0
    period = 360.0 / (wavenumber * speed_deg_day)
    t = np.arange(nt)[:, None, None]
    kelvin = (
        6.0
        / 86400
        * np.exp(-((lat / 8.0) ** 2))[None, :, None]
        * np.cos(2 * np.pi * (wavenumber * lon[None, None, :] / 360 - t / period))
    )
    coords = {"time": pd.date_range("2000-01-01", periods=nt), "lat": lat, "lon": lon}
    wrap = lambda a: xr.DataArray(a, dims=("time", "lat", "lon"), coords=coords)  # noqa: E731
    rng = np.random.default_rng(1)
    precipitation = wrap(3.0 / 86400 + kelvin + 0.5 / 86400 * rng.standard_normal(kelvin.shape))

    estimates = {}
    for method in ("ridge", "threshold_fit"):
        directory = tmp_path / method
        phase_module.composite_kw_phase(
            wrap(kelvin), precipitation, lon, "TEST", str(directory), lon_ref=180.0, nlag=10, speed_method=method
        )
        speed_file = next(name for name in os.listdir(directory) if "Cp_ave" in name)
        estimates[method] = float(np.load(directory / speed_file)["Cp_ave"])
    assert estimates["ridge"] == pytest.approx(speed, rel=0.05)
    assert abs(estimates["threshold_fit"] - speed) > 5.0
