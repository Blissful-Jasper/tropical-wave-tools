"""``generate_local_wave_atlas`` end to end on a synthetic record with a known Kelvin wave.

The input is an eastward wave (zonal wavenumber 5, period 6 days, so exactly 12 degrees of longitude per
day) inside Gaussian packets at known dates, over red noise and a seasonal cycle. Every output of the
workflow is then checked against what the input says it must be: event dates, filtered standard
deviation, the eastward propagation of the lagged composite and of the Case 10 regression, the
spectral peak, the wave EOF pair, and the files and figures that were written.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tropical_wave_tools import atlas
from tropical_wave_tools.atlas import workflow
from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.io import load_dataarray

WAVENUMBER = 5
PERIOD_DAYS = 6.0
SPEED = 360.0 / WAVENUMBER / PERIOD_DAYS  # degrees of longitude per day
PACKET_DAYS = (150, 330, 520, 700)
TIME_RANGE = ("1979-01-01", "1981-06-30")


def _write_inputs(directory, *, with_gpcp):
    rng = np.random.default_rng(0)
    time = pd.date_range("1979-01-01", periods=913)
    lat = np.arange(-20.0, 20.1, 5.0)
    lon = np.arange(0.0, 360.0, 10.0)
    t = np.arange(time.size)[:, None, None]
    lon_rad = np.deg2rad(lon)[None, None, :]
    lat_grid = lat[None, :, None]
    wave = np.cos(WAVENUMBER * lon_rad - 2.0 * np.pi * t / PERIOD_DAYS)
    envelope = sum(np.exp(-(((np.arange(time.size) - day) / 12.0) ** 2)) for day in PACKET_DAYS)[:, None, None]
    lat_envelope = np.exp(-((lat_grid / 8.0) ** 2))
    noise = rng.standard_normal((time.size, lat.size, lon.size))
    for index in range(1, time.size):  # red noise
        noise[index] = 0.7 * noise[index - 1] + 0.714 * noise[index]
    season = np.exp(-((lat_grid / 12.0) ** 2)) * np.cos(2.0 * np.pi * (t - 30) / 365.25)
    signal = envelope * lat_envelope * wave
    fields = {
        "olr.nc": ("olr", 240.0 + 12.0 * season - 25.0 * signal + 3.0 * noise, "W m-2"),
        "u850.nc": ("uwnd", 2.0 * np.sin(2.0 * np.pi * (t - 100) / 365.25) + 4.0 * signal + 0.8 * noise[::-1], "m s-1"),
        "v850.nc": (
            "vwnd",
            1.2 * envelope * np.sin(np.deg2rad(lat_grid)) * np.sin(WAVENUMBER * lon_rad - 2.0 * np.pi * t / PERIOD_DAYS)
            + 0.5 * rng.standard_normal(noise.shape),
            "m s-1",
        ),
    }
    if with_gpcp:
        fields["gpcp.nc"] = ("precipitation", 4.0 + 1.5 * season + 3.0 * signal + 0.6 * noise, "mm day-1")
    paths = {}
    for name, (variable, values, units) in fields.items():
        xr.Dataset(
            {variable: (("time", "lat", "lon"), values.astype("float32"), {"units": units})},
            coords={"time": time, "lat": lat, "lon": lon},
        ).to_netcdf(directory / name)
        paths[name.split(".")[0]] = directory / name
    return paths


def _stub_plots(monkeypatch):
    """Replace the (slow) figure functions of the workflow by empty figures: used where only data matter."""
    for name in dir(workflow):
        if name.startswith("plot_"):
            monkeypatch.setattr(workflow, name, lambda *args, **kwargs: (plt.figure(figsize=(1, 1)), None))


@pytest.fixture(scope="module")
def olr_atlas(tmp_path_factory):
    """The full atlas for the Kelvin wave, Case 05 on the OLR (no GPCP file)."""
    root = tmp_path_factory.mktemp("atlas_olr")
    paths = _write_inputs(root, with_gpcp=False)
    summary = atlas.generate_local_wave_atlas(
        output_dir=root / "out",
        olr_path=paths["olr"],
        u850_path=paths["u850"],
        v850_path=paths["v850"],
        gpcp_path=None,
        waves=["kelvin"],
        time_range=TIME_RANGE,
        n_workers=1,
    )
    return {"root": root, "out": root / "out", "paths": paths, "summary": summary}


@pytest.fixture(scope="module")
def gpcp_atlas(tmp_path_factory):
    """Case 05 on GPCP precipitation; figures stubbed since only the variance-fraction data matter here."""
    root = tmp_path_factory.mktemp("atlas_gpcp")
    paths = _write_inputs(root, with_gpcp=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _stub_plots(monkeypatch)
        atlas.generate_local_wave_atlas(
            output_dir=root / "out",
            olr_path=paths["olr"],
            u850_path=paths["u850"],
            v850_path=paths["v850"],
            gpcp_path=paths["gpcp"],
            waves=["kelvin"],
            time_range=TIME_RANGE,
            n_workers=1,
        )
    return {"root": root, "out": root / "out", "paths": paths}


def _wrap(angle, period):
    return (angle + period / 2.0) % period - period / 2.0


def _harmonic_speed(field, lags, lon, k=WAVENUMBER):
    """Eastward speed (degrees of longitude per lag) of the k-th zonal harmonic of a (lag, lon) field."""
    phase = [np.angle(np.sum(field.sel(lag=lag).values * np.exp(-1j * np.deg2rad(k * lon)))) for lag in lags]
    return float(-np.polyfit(lags, np.unwrap(phase), 1)[0] / k * 180.0 / np.pi)


def _windowed_speed(field):
    """Best-fit speed of a travelling cos(k (lon - c lag)) in a (lag, lon) field that spans only part of the circle."""
    lag, lon = np.meshgrid(field.lag.values, field.lon.values, indexing="ij")
    target = field.values.ravel()
    best = None
    for speed in np.arange(4.0, 20.0, 0.05):
        phase = np.deg2rad(WAVENUMBER * (lon - speed * lag))
        design = np.stack([np.cos(phase).ravel(), np.sin(phase).ravel(), np.ones(phase.size)], axis=1)
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = float(np.sum((design @ coefficients - target) ** 2))
        if best is None or residual < best[0]:
            best = (residual, speed)
    return best[1]


# ================================================================================ outputs
def test_summary_and_expected_files(olr_atlas) -> None:
    summary, out = olr_atlas["summary"], olr_atlas["out"]
    assert summary["wave"].tolist() == ["kelvin"]
    row = summary.iloc[0]
    assert row["event_count"] >= 3
    assert row["reference_projection"] == "equatorial_mean" and row["expected_direction"] == "eastward"
    assert pd.isna(row["note"])  # no fallback event was needed
    assert pd.read_csv(out / "wave_atlas_summary.csv").iloc[0]["event_count"] == row["event_count"]
    expected = {
        "kelvin_std_maps.nc", "kelvin_spatial_std_triptych.png", "kelvin_hovmoller_triptych.png",
        "kelvin_paper_hovmoller.nc", "kelvin_paper_hovmoller.png", "kelvin_event_times.csv",
        "kelvin_lagged_composites.nc", "kelvin_horizontal_structure_lag0.png", "kelvin_horizontal_structure_lags.png",
        "kelvin_lead_lag_evolution.png", "kelvin_wind_diagnostics_lag0.nc", "kelvin_wind_diagnostics_lag0.png",
        "kelvin_eof_modes.nc", "kelvin_eof_modes.png", "kelvin_monthly_cycle.png", "kelvin_seasonal_longitude_olr.png",
        "kelvin_monthly_rms.nc",
    }  # fmt: skip
    assert expected <= {path.name for path in (out / "kelvin").iterdir()}
    top_level = {path.name for path in out.iterdir() if path.is_file()}
    assert {
        "wave_atlas_summary.csv", "case05_variance_fraction_summary.nc", "case05_seasonal_variance_cycle.png",
        "case05_regional_variance_cycle.png", "multiwave_filtered_olr_std.nc",
        "multiwave_large_scale_filtered_olr_std.png", "multiwave_large_scale_monthly_cycle.png",
        "multiwave_large_scale_seasonal_longitude_olr.png", "multiwave_large_scale_annual_rms_trend.png",
    } <= top_level  # fmt: skip
    assert {"wk_spectrum_olr.png", "wk_spectrum_u850.nc", "wk_spectrum_v850.png"} <= {p.name for p in (out / "spectra").iterdir()}


def test_every_figure_is_a_real_non_blank_image(olr_atlas) -> None:
    from PIL import Image

    figures = sorted(olr_atlas["out"].rglob("*.png"))
    assert len(figures) >= 18
    for path in figures:
        image = Image.open(path).convert("RGB")
        pixels = np.asarray(image, dtype=float)
        assert min(image.size) >= 200, path.name
        assert pixels.std() > 5.0, f"{path.name} looks blank"


def test_detected_events_lie_inside_the_injected_packets(olr_atlas) -> None:
    events = pd.read_csv(olr_atlas["out"] / "kelvin" / "kelvin_event_times.csv", parse_dates=["event_time"])
    assert events["event_time"].is_monotonic_increasing and len(events) == olr_atlas["summary"].iloc[0]["event_count"]
    centres = pd.Timestamp("1979-01-01") + pd.to_timedelta(PACKET_DAYS, unit="D")
    distance = [np.min(np.abs((centres - when).days)) for when in events["event_time"]]
    assert max(distance) <= 25  # the packets have a 12-day e-folding width
    assert all(np.any(np.abs((centres - when).days) <= 25) for when in centres)  # every packet found at least once
    assert {int(np.argmin(np.abs((centres - when).days))) for when in events["event_time"]} == set(range(len(PACKET_DAYS)))


def test_standard_deviation_maps_equal_an_independent_filter(olr_atlas) -> None:
    std = xr.open_dataset(olr_atlas["out"] / "kelvin" / "kelvin_std_maps.nc")
    olr = load_dataarray(olr_atlas["paths"]["olr"], variable="olr", time_range=TIME_RANGE, lat_range=(-20.0, 20.0))
    reference = filter_wave_signal(olr, wave_name="kelvin", method="cckw", obs_per_day=1, n_harm=3, n_workers=1).std("time")
    np.testing.assert_allclose(std["olr_std"].values, reference.values, rtol=1e-6, atol=1e-9)
    assert float(std["olr_std"].max("lon").idxmax("lat")) == 0.0  # the wave's latitude envelope peaks at the equator
    assert float(std["olr_std"].sel(lat=0.0).mean()) > 3.0 * float(std["olr_std"].sel(lat=20.0).mean())
    assert float(std["u850_std"].sel(lat=0.0).mean()) > 0.0 and float(std["v850_std"].sel(lat=0.0).max()) >= 0.0


def test_lagged_composite_propagates_eastward_at_the_speed_of_the_wave(olr_atlas) -> None:
    composite = xr.open_dataset(olr_atlas["out"] / "kelvin" / "kelvin_lagged_composites.nc")
    equatorial = composite["olr"].sel(lat=slice(-5, 5)).mean("lat")
    assert _harmonic_speed(equatorial, composite.lag.values, equatorial.lon.values) == pytest.approx(SPEED, abs=0.3)
    # OLR is lowest (convection) at the reference longitude at lag 0, up to the wavelength 72 degrees
    at_reference = float(equatorial.sel(lag=0).idxmin("lon"))
    assert abs(_wrap(at_reference - 180.0, 360.0 / WAVENUMBER)) <= 10.0
    assert float(equatorial.sel(lag=0).min()) < -3.0  # a suppressed-OLR composite, i.e. convectively active
    # the number of events behind each lag is reported and does not exceed the event count
    counts = composite["olr"]["n_events"]
    assert int(counts.sel(lag=0)) == olr_atlas["summary"].iloc[0]["event_count"]


def test_case10_regression_propagates_eastward_and_is_negative_at_the_base_point(olr_atlas) -> None:
    case10 = xr.open_dataset(olr_atlas["out"] / "kelvin" / "kelvin_paper_hovmoller.nc")
    _, base_lon = atlas.CASE10_BASE_POINTS["kelvin"]
    assert float(case10["shading"].sel(lag=0, lon=base_lon, method="nearest")) < 0.0  # shading is the -OLR proxy
    assert _windowed_speed(case10["shading"]) == pytest.approx(SPEED, abs=0.3)
    assert _windowed_speed(case10["contours"]) == pytest.approx(SPEED, abs=0.3)
    assert float(case10["contours"].sel(lag=0, lon=base_lon, method="nearest")) > 0.0  # contours: OLR on itself
    assert int(case10["active_mask"].sum()) > 0


def test_saved_spectrum_peaks_at_the_wave(olr_atlas) -> None:
    for name in ("olr", "u850"):
        spectrum = xr.open_dataset(olr_atlas["out"] / "spectra" / f"wk_spectrum_{name}.nc")
        ratio = (spectrum["power_symmetric"] / spectrum["background"]).sel(wavenumber=slice(1, 15), frequency=slice(0.03, 0.3))
        peak = ratio.where(ratio == ratio.max(), drop=True)
        assert abs(float(peak.wavenumber[0]) - WAVENUMBER) <= 1
        assert float(peak.frequency[0]) == pytest.approx(1.0 / PERIOD_DAYS, abs=0.012)
        assert float(ratio.max()) > 1.5


def test_wave_eof_is_a_propagating_pair(olr_atlas) -> None:
    modes = xr.open_dataset(olr_atlas["out"] / "kelvin" / "kelvin_eof_modes.nc")
    explained = modes["explained_variance"].values
    assert explained[0] == pytest.approx(explained[1], rel=1e-3) and explained.sum() > 90.0  # a travelling wave = a quadrature pair
    assert abs(float(xr.corr(modes["pcs"].sel(mode=1), modes["pcs"].sel(mode=2)))) < 0.05
    for mode in (1, 2):
        pattern = modes["eofs"].sel(mode=mode).values
        assert pattern.flat[np.abs(pattern).argmax()] > 0.0  # the documented sign convention
    assert modes["olr_regressed"].dims == ("mode", "lat", "lon")
    # the two patterns are in spatial quadrature: a zonal shift by a quarter wavelength (18 degrees = 1.8 cells) of one is the other
    first, second = (modes["eofs"].sel(mode=m, lat=0.0).values for m in (1, 2))
    harmonic = lambda a: np.angle(np.sum(a * np.exp(-1j * np.deg2rad(WAVENUMBER * modes.lon.values))))  # noqa: E731
    assert abs(abs(_wrap(harmonic(first) - harmonic(second), 2.0 * np.pi)) - np.pi / 2.0) < 0.3


def test_monthly_statistics_and_trend_files(olr_atlas) -> None:
    monthly = xr.open_dataset(olr_atlas["out"] / "kelvin" / "kelvin_monthly_rms.nc")
    assert monthly["olr_monthly_rms"].dims == ("month", "lat", "lon") or monthly["olr_monthly_rms"].dims == ("month", "lon")
    assert (monthly["olr_monthly_rms"].values[np.isfinite(monthly["olr_monthly_rms"].values)] >= 0.0).all()
    assert monthly["olr_annual_rms"].sizes["year"] == 3  # 1979, 1980, 1981 (half)
    assert 0.0 <= float(monthly["olr_annual_rms_trend_pvalue"]) <= 1.0
    # packets are scattered through the year, so the wave's RMS is nowhere zero
    assert float(monthly["olr_monthly_rms"].min()) > 0.0


def test_case05_uses_the_olr_when_there_is_no_gpcp_file(olr_atlas) -> None:
    summary = xr.open_dataset(olr_atlas["out"] / "case05_variance_fraction_summary.nc")
    assert summary.attrs["source"] == "OLR"
    assert summary["tropical_mean"].dims == ("wave", "month") and summary["regional_mean"].dims == ("region", "wave", "month")
    assert summary["tropical_mean"].wave.values.tolist() == ["kelvin"]
    values = summary["tropical_mean"].values
    assert np.isfinite(values).all() and (values >= 0.0).all() and (values <= 100.0).all()
    assert (summary["tropical_std"].values >= 0.0).all()


def test_case05_uses_gpcp_precipitation_when_the_file_is_given(gpcp_atlas, olr_atlas) -> None:
    summary = xr.open_dataset(gpcp_atlas["out"] / "case05_variance_fraction_summary.nc")
    assert summary.attrs["source"] == "GPCP precipitation"
    assert "precipitation" in summary.attrs["description"]
    values = summary["tropical_mean"].values
    assert np.isfinite(values).all() and (values >= 0.0).all() and (values <= 100.0).all()
    olr_values = xr.open_dataset(olr_atlas["out"] / "case05_variance_fraction_summary.nc")["tropical_mean"].values
    assert not np.allclose(values, olr_values)  # a different input gives different fractions


def test_the_gpcp_path_is_optional_and_defaults_to_the_local_file() -> None:
    import inspect

    default = inspect.signature(atlas.generate_local_wave_atlas).parameters["gpcp_path"].default
    assert default == atlas.DEFAULT_LOCAL_PATHS["gpcp"]
