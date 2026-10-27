"""Atlas event detection, composites and projections against known signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tropical_wave_tools import atlas

LAT = np.arange(-15.0, 15.1, 2.5)
LON = np.arange(0.0, 360.0, 10.0)
ENVELOPE = np.exp(-((LAT / 8.0) ** 2))[None, :, None]


def _cube(values: np.ndarray, *, freq: str = "D", lat: np.ndarray = LAT) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=values.shape[0], freq=freq),
            "lat": lat,
            "lon": LON,
        },
        name="olr",
    )


def _cosine(nt: int = 200, period: float = 20.0) -> xr.DataArray:
    t = np.arange(nt)[:, None, None]
    return _cube(40.0 * np.cos(2 * np.pi * t / period) * ENVELOPE * np.ones((1, 1, LON.size)))


def test_events_of_a_known_sinusoid_are_its_minima_or_maxima() -> None:
    field = _cosine()
    negative, _ = atlas.detect_wave_events(field, wave_name="kelvin", sign="negative", lon_ref=180.0)
    positive, _ = atlas.detect_wave_events(field, wave_name="kelvin", sign="positive", lon_ref=180.0)
    np.testing.assert_array_equal(negative, np.arange(10, 200, 20))  # minima of cos
    np.testing.assert_array_equal(positive, np.arange(20, 200, 20))  # interior maxima (index 0 is an edge)


@pytest.mark.parametrize(("freq", "gap_samples"), [("D", 3), ("6h", 12), ("h", 72)])
def test_min_distance_is_in_days_not_samples(freq: str, gap_samples: int) -> None:
    """Two events 3 days apart must merge into one, whatever the sampling interval."""
    series = np.zeros(400)
    series[100] = -5.0
    series[100 + gap_samples] = -4.0
    field = _cube(series[:, None, None] * ENVELOPE * np.ones((1, 1, LON.size)), freq=freq)
    events, _ = atlas.detect_wave_events(
        field, wave_name="kelvin", lon_ref=180.0, threshold_std=0.5, min_distance_days=7
    )
    assert events.tolist() == [100]  # the stronger of the two


def test_descending_latitude_gives_the_same_events_and_projections() -> None:
    """ERA5-style 90..-90 axes used to give all-NaN projections and zero events."""
    t = np.arange(200)[:, None, None]
    values = 40.0 * np.cos(2 * np.pi * t / 20.0) * ENVELOPE * np.ones((1, 1, LON.size))
    ascending = _cube(values)
    descending = _cube(values[:, ::-1, :], lat=LAT[::-1])

    for wave in ("kelvin", "mrg"):  # symmetric and antisymmetric projections
        expected = atlas.wave_longitude_projection(ascending, wave_name=wave)
        result = atlas.wave_longitude_projection(descending, wave_name=wave)
        np.testing.assert_allclose(result.values, expected.values, atol=1e-9)
        assert result.attrs["wave_projection"] == expected.attrs["wave_projection"]
    expected_events, _ = atlas.detect_wave_events(ascending, wave_name="kelvin")
    events, _ = atlas.detect_wave_events(descending, wave_name="kelvin")
    np.testing.assert_array_equal(events, expected_events)
    assert events.size == 10


def test_all_nan_reference_warns_instead_of_silently_returning_nothing() -> None:
    field = _cosine() * np.nan
    with pytest.warns(RuntimeWarning, match="no finite values"):
        events, _ = atlas.detect_wave_events(field, wave_name="kelvin")
    assert events.size == 0


def test_point_events_match_wave_events_on_a_uniform_field_and_validate_sign() -> None:
    field = _cosine()
    point, reference = atlas.detect_point_events(field, base_lat=0.0, base_lon=180.0)
    wave, _ = atlas.detect_wave_events(field, wave_name="kelvin", lon_ref=180.0)
    np.testing.assert_array_equal(point, wave)
    assert reference.attrs["wave_projection"] == "base_point"
    with pytest.raises(ValueError, match="sign"):
        atlas.detect_point_events(field, base_lat=0.0, base_lon=180.0, sign="sideways")


def test_events_in_a_nan_gap_are_not_reported() -> None:
    field = _cosine().copy()
    field[48:62] = np.nan  # contains the minimum at index 50
    events, _ = atlas.detect_wave_events(field, wave_name="kelvin", lon_ref=180.0)
    assert 50 not in events
    assert {10, 30, 70, 90}.issubset(set(events.tolist()))


def test_lagged_composite_is_the_mean_at_event_plus_lag_and_reports_event_counts() -> None:
    rng = np.random.default_rng(0)
    data = xr.DataArray(
        rng.standard_normal((100, 3)),
        dims=("time", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=100), "lon": [0.0, 1.0, 2.0]},
    )
    events = np.array([2, 30, 60, 97])
    composite = atlas.lagged_composite(data, events, lags=[-5, 0, 5])
    for lag, expected_count in ((-5, 3), (0, 4), (5, 3)):
        indices = events + lag
        indices = indices[(indices >= 0) & (indices < 100)]
        np.testing.assert_allclose(composite.sel(lag=lag).values, data.values[indices].mean(0))
        assert int(composite["n_events"].sel(lag=lag)) == expected_count


def test_hovmoller_window_has_fixed_length_and_contains_the_strongest_event() -> None:
    nt = 300
    t = np.arange(nt)[:, None, None]
    base = 10.0 * np.sin(2 * np.pi * t / 17) * ENVELOPE * np.ones((1, 1, LON.size))
    for centre in (3, 150, 297):  # near either end and in the middle
        values = base.copy()
        values[centre] += 200.0 * ENVELOPE[0]
        window = atlas.select_hovmoller_window(_cube(values), wave_name="kelvin", lon_ref=180.0, window_days=180)
        start, stop = pd.Timestamp(window.start), pd.Timestamp(window.stop)
        assert (stop - start).days + 1 == 180
        assert start <= pd.Timestamp("2000-01-01") + pd.Timedelta(days=centre) <= stop


def test_activity_mask_marks_the_strong_stretch_only() -> None:
    t = np.arange(600)
    amplitude = np.where((t >= 200) & (t < 400), 3.0, 0.3)
    reference = xr.DataArray(
        amplitude * np.sin(2 * np.pi * t / 15),
        dims="time",
        coords={"time": pd.date_range("2000-01-01", periods=600)},
    )
    mask = atlas._case10_activity_mask(reference, window_days=60)
    active = np.isfinite(mask.values)
    assert active[200:400].all()
    assert np.r_[active[:150], active[450:]].sum() == 0  # a 60-day window blurs the edges by ~30 days
    assert 0.30 < active.mean() < 0.42


def test_projection_conventions_are_equatorial_mean_and_north_minus_south() -> None:
    nt = 10
    symmetric = _cube(np.ones((nt, 1, 1)) * np.cos(np.deg2rad(LAT))[None, :, None] * np.ones((1, 1, LON.size)))
    antisymmetric = _cube(
        np.ones((nt, 1, 1)) * np.sin(np.deg2rad(6 * LAT))[None, :, None] * np.ones((1, 1, LON.size))
    )
    kelvin = atlas.wave_longitude_projection(symmetric, wave_name="kelvin")
    np.testing.assert_allclose(kelvin.isel(time=0, lon=0), np.mean(np.cos(np.deg2rad(LAT[np.abs(LAT) <= 5]))))

    north, south = LAT[(LAT > 0) & (LAT <= 10)], LAT[(LAT < 0) & (LAT >= -10)]
    mrg = atlas.wave_longitude_projection(antisymmetric, wave_name="mrg")
    expected = np.mean(np.sin(np.deg2rad(6 * north))) - np.mean(np.sin(np.deg2rad(6 * south)))
    np.testing.assert_allclose(mrg.isel(time=0, lon=0), expected)  # N - S: twice the one-sided amplitude
    assert mrg.attrs["wave_projection"] == "antisymmetric_difference"


def test_event_composite_recovers_the_propagation_of_wave_packets() -> None:
    """Detect events at a reference longitude and composite: the crest must move at the true speed."""
    nt, speed, wavenumber = 900, 15.0, 2
    speed_deg_day = speed * 86400 / 111000.0
    period = 360.0 / (wavenumber * speed_deg_day)
    t = np.arange(nt)[:, None, None]
    lo = LON[None, None, :]
    packets = np.zeros((nt, LAT.size, LON.size))
    centres = list(range(60, nt - 60, 60))
    for centre in centres:
        offset = ((lo - 180.0 - speed_deg_day * (t - centre) + 180) % 360) - 180
        packets += (
            np.cos(2 * np.pi * (wavenumber * lo / 360 - (t - centre) / period))
            * np.exp(-((offset / 40.0) ** 2))
            * np.exp(-(((t - centre) / 12.0) ** 2))
            * ENVELOPE
        )
    olr = _cube(-30.0 * packets)  # convection = negative OLR anomaly

    events, _ = atlas.detect_wave_events(olr, wave_name="kelvin", lon_ref=180.0, threshold_std=2.0)
    assert set(centres).issubset(set(events.tolist()))  # every packet centre is found
    main = np.array([e for e in events if min(abs(e - c) for c in centres) == 0])

    lags = np.arange(-4, 5)
    composite = atlas.lagged_composite(olr.sel(lat=slice(-5, 5)).mean("lat"), main, lags=lags)
    crest = [LON[np.nanargmin(np.where(np.abs(LON - (180 + speed_deg_day * lag)) < 30, composite.sel(lag=lag).values, np.nan))] for lag in lags]
    fitted_speed = np.polyfit(lags, crest, 1)[0] * 111000.0 / 86400.0
    assert fitted_speed == pytest.approx(speed, rel=0.1)
