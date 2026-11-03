from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.kelvin_events import (
    KelvinEventConfig,
    detect_propagating_ridges,
    equatorial_symmetry_index,
    event_composite,
    event_count_sensitivity,
    identify_kelvin_events,
    kelvin_band_filter,
    kelvin_dynamical_index,
    kelvin_meridional_structure,
    project_onto_kelvin,
    select_events_in_region,
    symmetric_antisymmetric,
)
from tropical_wave_tools.sample_data import open_example_olr
from tropical_wave_tools.synthetic import SyntheticEvent, make_synthetic_kelvin_case

# Thresholds suited to the heavy-tailed synthetic case (a few strong packets in
# otherwise weak noise).  Real data usually needs a sensitivity check instead.
SYNTHETIC_CONFIG = KelvinEventConfig(
    remove_seasonal_cycle=False,
    axis_window_days=61,
    peak_sigma=3.0,
    continue_sigma=1.5,
    dyn_min_amplitude=1.5,
    conv_min_amplitude=3.0,
)


@pytest.fixture(scope="module")
def case() -> xr.Dataset:
    return make_synthetic_kelvin_case()


@pytest.fixture(scope="module")
def catalog(case):
    return identify_kelvin_events(u=case.u, z=case.z, v=case.v, olr=case.olr, config=SYNTHETIC_CONFIG)


# --------------------------------------------------------------------------- #
# Meridional projection
# --------------------------------------------------------------------------- #
def _lat_field(lat: np.ndarray, values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values[:, None], dims=("lat", "lon"), coords={"lat": lat, "lon": [0.0]})


def test_kelvin_structure_efolds_at_deformation_radius() -> None:
    lat = np.array([0.0, 5.0, 10.0])
    psi = kelvin_meridional_structure(lat, equivalent_depth=25.0)
    assert psi[0] == pytest.approx(1.0)
    assert psi[0] > psi[1] > psi[2] > 0.0
    # Ld = sqrt(c/beta) ~ 7.5 deg for he = 25 m, so psi(Ld) = exp(-1/2)
    ld_deg = np.sqrt(np.sqrt(9.80665 * 25.0) / (2 * 7.292e-5 / 6.371008e6)) / (6.371008e6 * np.pi / 180.0)
    assert 7.0 < ld_deg < 8.0
    at_ld = kelvin_meridional_structure(np.array([ld_deg]), equivalent_depth=25.0)
    assert at_ld[0] == pytest.approx(np.exp(-0.5), rel=1e-6)


@pytest.mark.parametrize("lat_limit", [10.0, 15.0, 20.0])
def test_projection_recovers_amplitude_on_truncated_band(lat_limit: float) -> None:
    lat = np.arange(-20.0, 20.1, 2.5)
    amplitude = 3.7
    field = _lat_field(lat, amplitude * kelvin_meridional_structure(lat, equivalent_depth=25.0))
    result = project_onto_kelvin(field, equivalent_depth=25.0, lat_limit=lat_limit)
    assert float(result.squeeze()) == pytest.approx(amplitude, rel=1e-9)


def test_projection_is_orthogonal_to_antisymmetric_structure() -> None:
    lat = np.arange(-20.0, 20.1, 2.5)
    psi = kelvin_meridional_structure(lat, equivalent_depth=25.0)
    antisymmetric = _lat_field(lat, (lat / 7.5) * psi)
    assert abs(float(project_onto_kelvin(antisymmetric, lat_limit=15.0).squeeze())) < 1e-10


def test_projection_masks_sparse_columns() -> None:
    lat = np.arange(-15.0, 15.1, 2.5)
    values = np.full(lat.size, np.nan)
    values[0] = 1.0
    assert np.isnan(float(project_onto_kelvin(_lat_field(lat, values), lat_limit=15.0).squeeze()))


def test_q_projection_ignores_meridional_wind_by_construction(case) -> None:
    """With z = u c / g the Kelvin amplitude equals the u projection."""
    q = kelvin_dynamical_index(case.u, case.z, equivalent_depth=25.0)
    u_only = kelvin_dynamical_index(case.u, equivalent_depth=25.0, method="u")
    np.testing.assert_allclose(q.values, u_only.values, atol=1.0e-9)
    assert q.attrs["method"] == "q" and u_only.attrs["method"] == "u"
    with pytest.raises(ValueError):
        kelvin_dynamical_index(case.u, method="q")


def test_offequatorial_heating_projects_with_gaussian_factor() -> None:
    """A heating displaced to ~Ld retains exp(-1/2) of its Kelvin projection."""
    lat = np.arange(-30.0, 30.1, 0.5)
    ld_deg = 7.5
    centred = np.exp(-0.5 * (lat / 3.0) ** 2)
    shifted = np.exp(-0.5 * ((lat - ld_deg) / 3.0) ** 2)
    a0 = float(project_onto_kelvin(_lat_field(lat, centred), lat_limit=30.0).squeeze())
    a1 = float(project_onto_kelvin(_lat_field(lat, shifted), lat_limit=30.0).squeeze())
    assert 0.45 < a1 / a0 < 0.75


# --------------------------------------------------------------------------- #
# Symmetry diagnostics
# --------------------------------------------------------------------------- #
def test_symmetric_antisymmetric_partition_reconstructs_field() -> None:
    lat = np.arange(-15.0, 15.1, 2.5)
    rng = np.random.default_rng(1)
    field = xr.DataArray(rng.normal(size=(lat.size, 4)), dims=("lat", "lon"), coords={"lat": lat, "lon": np.arange(4.0)})
    sym, asym = symmetric_antisymmetric(field)
    np.testing.assert_allclose((sym + asym).values, field.values, atol=1.0e-12)
    np.testing.assert_allclose(sym.values, sym.values[::-1], atol=1.0e-12)
    np.testing.assert_allclose(asym.values, -asym.values[::-1], atol=1.0e-12)


def test_symmetry_index_limits() -> None:
    lat = np.arange(-15.0, 15.1, 2.5)
    even = _lat_field(lat, np.exp(-((lat / 8.0) ** 2)))
    odd = _lat_field(lat, lat * np.exp(-((lat / 8.0) ** 2)))
    assert float(equatorial_symmetry_index(even).squeeze()) == pytest.approx(1.0)
    assert float(equatorial_symmetry_index(odd).squeeze()) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Filtering and input validation
# --------------------------------------------------------------------------- #
def test_band_filter_matches_repository_cckw_filter() -> None:
    olr = open_example_olr()
    mine = kelvin_band_filter(olr, obs_per_day=1)
    reference = filter_wave_signal(olr, wave_name="kelvin", method="cckw", n_workers=1)
    # The repository filter runs in float32; this one promotes to float64 first,
    # so agreement is limited by float32 round-off (~1e-5 relative).
    np.testing.assert_allclose(mine.values, reference.values, rtol=1.0e-4, atol=1.0e-4)


def test_regional_longitude_domain_is_rejected(case) -> None:
    regional = case.olr.sel(lon=slice(0, 120))
    with pytest.raises(ValueError, match="global"):
        kelvin_band_filter(regional, remove_seasonal_cycle=False)


def test_missing_dimensions_are_rejected(case) -> None:
    with pytest.raises(ValueError, match="time, lat, lon"):
        kelvin_band_filter(case.olr.isel(time=0, drop=True), remove_seasonal_cycle=False)
    with pytest.raises(ValueError):
        identify_kelvin_events()


# --------------------------------------------------------------------------- #
# Ridge tracker
# --------------------------------------------------------------------------- #
def _moving_bump(speed_ms: float, *, lon0: float = 20.0, ndays: int = 16) -> xr.DataArray:
    lon = np.arange(0.0, 360.0, 2.5)
    deg_per_day = speed_ms * 86400.0 / (6.371008e6 * np.pi / 180.0)
    t = np.arange(ndays)
    centre = lon0 + deg_per_day * t
    delta = (lon[None, :] - centre[:, None] + 180.0) % 360.0 - 180.0
    return xr.DataArray(
        np.exp(-0.5 * (delta / 12.0) ** 2),
        dims=("time", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=ndays), "lon": lon},
    )


def test_tracker_recovers_eastward_phase_speed() -> None:
    events, tracks = detect_propagating_ridges(_moving_bump(15.0), peak_sigma=1.0, continue_sigma=0.3)
    assert len(events) == 1
    assert events.loc[0, "speed_ms"] == pytest.approx(15.0, abs=0.7)
    assert events.loc[0, "distance_deg"] > 100.0
    assert set(tracks["event_id"]) == {events.loc[0, "event_id"]}


@pytest.mark.parametrize("speed", [-15.0, 0.0, 3.0])
def test_tracker_rejects_westward_and_slow_ridges(speed: float) -> None:
    events, _ = detect_propagating_ridges(_moving_bump(speed), peak_sigma=1.0, continue_sigma=0.3)
    assert events.empty


def test_tracker_rejects_ridges_that_are_too_fast() -> None:
    events, _ = detect_propagating_ridges(_moving_bump(45.0, ndays=8), peak_sigma=1.0, continue_sigma=0.3)
    assert events.empty


def test_event_count_sensitivity_is_monotonic(catalog) -> None:
    table = event_count_sensitivity(catalog.indices["kelvin_dynamical"], peak_sigmas=(1.0, 2.0, 3.0, 5.0))
    assert list(table["peak_sigma"]) == [1.0, 2.0, 3.0, 5.0]
    assert (np.diff(table["n_events"].values) <= 0).all()


# --------------------------------------------------------------------------- #
# End-to-end dual-channel identification
# --------------------------------------------------------------------------- #
def test_identifies_exactly_the_injected_kelvin_events(catalog) -> None:
    events = catalog.events
    assert len(events) == 2, events
    assert (events["kind"] == "coupled").all()
    for (_, ev), truth in zip(events.sort_values("start_time").iterrows(), [(40.0, 130.0, 15.0), (110.0, 200.0, 18.0)]):
        t0, lon0, speed = truth
        assert ev["speed_ms"] == pytest.approx(speed, abs=2.0)
        start_day = (pd.Timestamp(ev["start_time"]) - pd.Timestamp("2000-01-01")) / pd.Timedelta(days=1)
        lon_at_t0 = ev["fit_lon0_deg"] + ev["fit_slope_deg_per_day"] * (t0 - start_day)
        assert abs(((lon_at_t0 - lon0 + 180.0) % 360.0) - 180.0) < 15.0


def test_distractors_are_not_reported(catalog, case) -> None:
    """Westward packet (day 70) and antisymmetric eastward packet (day 150) are rejected."""
    peak_days = (pd.to_datetime(catalog.events["peak_time"]) - pd.Timestamp("2000-01-01")) / pd.Timedelta(days=1)
    assert not ((peak_days > 55) & (peak_days < 90)).any()
    assert not ((peak_days > 135) & (peak_days < 170)).any()
    assert catalog.events["speed_ms"].gt(0).all()


def test_wind_structure_diagnostics_confirm_kelvin_character(catalog) -> None:
    events = catalog.events
    assert (events["u_symmetry_index"] > 0.9).all()
    assert (events["v_to_u_ratio"] < 0.15).all()


def test_convective_axis_follows_the_offset_itcz(catalog) -> None:
    assert np.allclose(catalog.events["conv_axis_lat"], 8.0, atol=1.5)


def test_axis_following_index_recovers_more_signal_than_fixed_band(catalog) -> None:
    """The point of the method: fixed 5S-5N sampling under-estimates 8N convection."""
    events = catalog.events
    assert (events["conv_peak"] > 1.8 * events["conv_peak_fixed_band"]).all()
    assert (events["coupling_corr"] >= 0.9).all()


def test_coupling_is_statistically_significant(catalog) -> None:
    assert (catalog.events["coupling_pvalue"] <= 0.05).all()


def test_coupling_pvalue_is_not_small_for_unrelated_noise() -> None:
    """A strong wind wave with pure-noise OLR must not be called coupled."""
    dry = make_synthetic_kelvin_case(
        events=[SyntheticEvent(t0_day=60.0, lon0=100.0, u_amp=5.0, olr_amp=0.0)],
        add_distractors=False,
        seed=4,
    )
    result = identify_kelvin_events(u=dry.u, z=dry.z, olr=dry.olr, config=SYNTHETIC_CONFIG)
    dyn = result.events[result.events["event_id"].str.startswith("D")]
    assert len(dyn) == 1
    assert dyn.iloc[0]["kind"] == "dynamical_only"


def test_absolute_amplitude_threshold_removes_noise_only_ridges() -> None:
    quiet = make_synthetic_kelvin_case(events=[], add_distractors=False, seed=5)
    relative = KelvinEventConfig(remove_seasonal_cycle=False, peak_sigma=2.0, continue_sigma=1.0)
    absolute = KelvinEventConfig(
        remove_seasonal_cycle=False, peak_sigma=2.0, continue_sigma=1.0, dyn_min_amplitude=1.5, conv_min_amplitude=3.0
    )
    assert len(identify_kelvin_events(u=quiet.u, z=quiet.z, olr=quiet.olr, config=relative).events) > 0
    assert identify_kelvin_events(u=quiet.u, z=quiet.z, olr=quiet.olr, config=absolute).events.empty


def test_lagged_coupling_finds_convective_offset(catalog) -> None:
    ordered = catalog.events.sort_values("start_time").reset_index(drop=True)
    assert abs(ordered.loc[0, "conv_offset_deg"]) <= 5.0
    assert 5.0 <= ordered.loc[1, "conv_offset_deg"] <= 15.0  # injected: 10 deg east


def test_dynamical_only_without_convection_data(case) -> None:
    result = identify_kelvin_events(u=case.u, z=case.z, config=SYNTHETIC_CONFIG)
    assert len(result.events) == 2
    assert (result.events["kind"] == "dynamical_only").all()
    assert result.events["coupling_corr"].isna().all()


def test_convective_only_event_is_flagged_when_wind_is_quiet() -> None:
    dry = make_synthetic_kelvin_case(
        events=[SyntheticEvent(t0_day=60.0, lon0=100.0, u_amp=0.0, olr_amp=20.0)],
        add_distractors=False,
        seed=3,
    )
    result = identify_kelvin_events(u=dry.u, z=dry.z, olr=dry.olr, config=SYNTHETIC_CONFIG)
    assert len(result.events) >= 1
    assert (result.events["kind"] == "convective_only").all()


def test_dynamical_event_without_convection_is_dynamical_only() -> None:
    dry = make_synthetic_kelvin_case(
        events=[SyntheticEvent(t0_day=60.0, lon0=100.0, u_amp=5.0, olr_amp=0.0)],
        add_distractors=False,
        seed=4,
    )
    result = identify_kelvin_events(u=dry.u, z=dry.z, olr=dry.olr, config=SYNTHETIC_CONFIG)
    kinds = result.events["kind"].tolist()
    assert "dynamical_only" in kinds and "coupled" not in kinds


def test_prefiltered_inputs_skip_the_filter(case) -> None:
    u_k = kelvin_band_filter(case.u, remove_seasonal_cycle=False)
    a = identify_kelvin_events(u=u_k, config=SYNTHETIC_CONFIG, prefiltered=True)
    b = identify_kelvin_events(u=case.u, config=SYNTHETIC_CONFIG)
    assert len(a.events) == len(b.events) == 2


def test_event_composite_shows_symmetric_wind_and_northern_convection(case, catalog) -> None:
    u_comp = event_composite(case.u, catalog.events, kinds=("coupled",), lags_days=(0.0,), half_width_deg=30.0)
    olr_comp = event_composite(case.olr - 230.0, catalog.events, kinds=("coupled",), lags_days=(0.0,), half_width_deg=30.0)
    assert u_comp.dims == ("lag", "lat", "rel_lon")
    assert u_comp.attrs["n_events"] == 2

    # Strongest wind response is on the equator; the pattern is hemispherically even.
    peak = np.abs(u_comp.isel(lag=0)).max("rel_lon")
    assert abs(float(peak.lat.values[int(np.argmax(peak.values))])) <= 2.5
    north, south = float(peak.sel(lat=7.5)), float(peak.sel(lat=-7.5))
    assert north == pytest.approx(south, rel=0.35)

    # Strongest convective anomaly is poleward of 5N.
    olr_extreme = olr_comp.isel(lag=0).min("rel_lon")
    assert 5.0 <= float(olr_extreme.lat.values[int(np.argmin(olr_extreme.values))]) <= 11.0
    assert float(olr_extreme.min()) < 0.0


def test_catalog_summary_and_save(tmp_path, catalog) -> None:
    summary = catalog.summary()
    assert summary["coupled"] == 2
    out = catalog.save(tmp_path / "events")
    assert (out / "events.csv").exists() and (out / "tracks.csv").exists()
    reread = pd.read_csv(out / "events.csv")
    assert len(reread) == len(catalog.events)


# --------------------------------------------------------------------------- #
# Real sample data smoke test (convective channel only)
# --------------------------------------------------------------------------- #
def test_runs_on_packaged_olr_sample() -> None:
    olr = open_example_olr()
    result = identify_kelvin_events(olr=olr, config=KelvinEventConfig(peak_sigma=1.5, continue_sigma=0.75))
    axis = result.indices["convective_axis_lat"]
    assert float(axis.min()) >= -15.0 and float(axis.max()) <= 15.0
    assert {"kelvin_convective_axis", "kelvin_convective_fixed_band"} <= set(result.indices.data_vars)
    if not result.events.empty:
        assert result.events["speed_ms"].between(8.0, 30.0).all()
        assert (result.events["kind"] == "convective").all()


# --------------------------------------------------------------------------- #
# Regional / window reporting
# --------------------------------------------------------------------------- #
def test_region_selection_uses_track_points_and_handles_wraparound(catalog) -> None:
    ordered = catalog.events.sort_values("start_time")
    first, second = ordered["event_id"].tolist()
    # Event 1 spans ~36-210E, event 2 ~104-309E.
    west_only = catalog.in_region((240.0, 300.0))
    assert west_only.events["event_id"].tolist() == [second]
    # Tracks of the kept event plus its matched convective ridge(s) only.
    expected = {second, *[m for m in west_only.events["matched_event"].iloc[0].split(",") if m]}
    assert set(west_only.tracks["event_id"]) == expected
    assert first not in set(west_only.tracks["event_id"])
    wrapped = catalog.in_region((300.0, 60.0))
    assert first in set(wrapped.events["event_id"])
    both = catalog.in_region((120.0, 200.0))
    assert set(both.events["event_id"]) == {first, second}
    assert catalog.in_region((350.0, 355.0)).events.empty
    # A stricter point count removes marginal overlaps.
    events, _ = select_events_in_region(catalog.events, catalog.tracks, (200.0, 210.0), min_points=50)
    assert events.empty


def test_window_selection_by_peak_time(catalog) -> None:
    only_first = catalog.in_window("2000-01-01", "2000-03-01")
    assert len(only_first.events) == 1
    assert only_first.events["start_time"].iloc[0] < pd.Timestamp("2000-03-01")
    assert catalog.in_window("2001-01-01", "2001-12-31").events.empty
