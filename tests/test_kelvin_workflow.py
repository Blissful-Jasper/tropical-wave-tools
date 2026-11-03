from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tropical_wave_tools.cli import main
from tropical_wave_tools.kelvin_events import KelvinEventConfig
from tropical_wave_tools.kelvin_workflow import (
    load_reanalysis_field,
    regrid_global,
    run_regional_workflow,
    summarize_events,
)
from tropical_wave_tools.matsuno import GRAVITY
from tropical_wave_tools.synthetic import make_synthetic_kelvin_case

CONFIG = KelvinEventConfig(
    remove_seasonal_cycle=False,
    axis_window_days=61,
    axis_lat_range=(0.0, 15.0),
    peak_sigma=3.0,
    continue_sigma=1.5,
    dyn_min_amplitude=1.5,
    conv_min_amplitude=3.0,
)


def _era5_style(da: xr.DataArray, name: str, *, level: bool, scale: float = 1.0, subdaily: int = 1) -> xr.Dataset:
    """Mimic CDS naming: valid_time/latitude/longitude, descending latitude, pressure_level."""
    da = da * scale
    if subdaily > 1:
        time = pd.date_range(da.time.values[0], periods=da.sizes["time"] * subdaily, freq=f"{24 // subdaily}h")
        da = da.reindex(time=time, method="ffill")
    da = da.rename({"time": "valid_time", "lat": "latitude", "lon": "longitude"}).sortby("latitude", ascending=False)
    if level:
        da = da.expand_dims(pressure_level=[850.0], axis=1)
    return da.to_dataset(name=name)


@pytest.fixture(scope="module")
def era5_files(tmp_path_factory):
    case = make_synthetic_kelvin_case(dlon=1.25, dlat=1.25, lat_max=22.5, ntime=200)
    root = tmp_path_factory.mktemp("era5")
    paths = {
        "u": root / "u850.nc",
        "v": root / "v850.nc",
        "z": root / "z850.nc",
        "olr": root / "olr.nc",
        "ttr": root / "ttr_6h.nc",
    }
    _era5_style(case.u, "u", level=True).to_netcdf(paths["u"])
    _era5_style(case.v, "v", level=True).to_netcdf(paths["v"])
    _era5_style(case.z, "z", level=True, scale=GRAVITY).to_netcdf(paths["z"])
    _era5_style(case.olr, "olr", level=False).to_netcdf(paths["olr"])
    # ERA5-like OLR: hourly-accumulated ttr (J m-2, negative), sampled 4x daily
    _era5_style(case.olr, "ttr", level=False, scale=-3600.0, subdaily=4).to_netcdf(paths["ttr"])
    return paths, case


def test_loader_normalises_era5_conventions(era5_files) -> None:
    paths, case = era5_files
    u = load_reanalysis_field(paths["u"], "u", level=850, lat_range=(-20, 20), regrid_deg=2.5)
    assert u.dims == ("time", "lat", "lon")
    assert np.all(np.diff(u.lat.values) > 0) and 0.0 in u.lat.values
    np.testing.assert_allclose(u.lat.values, -u.lat.values[::-1])
    assert u.sizes["lon"] == 144 and float(u.lon[0]) == 0.0 and float(u.lon[-1]) == 357.5
    assert u.sizes["time"] == case.sizes["time"]


def test_regrid_preserves_a_noise_free_signal(tmp_path) -> None:
    clean = make_synthetic_kelvin_case(dlon=1.25, dlat=1.25, lat_max=22.5, ntime=30, noise_u=0.0, add_distractors=False)
    path = tmp_path / "clean.nc"
    _era5_style(clean.u, "u", level=True).to_netcdf(path)
    u = load_reanalysis_field(path, "u", level=850, lat_range=(-20, 20), regrid_deg=2.5)
    truth = clean.u.interp(lat=u.lat, lon=u.lon)
    assert np.abs(u - truth).max() < 0.25 * float(np.abs(truth).max())
    assert np.corrcoef(u.values.ravel(), truth.values.ravel())[0, 1] > 0.98


def test_geopotential_is_converted_to_height(era5_files) -> None:
    paths, case = era5_files
    z = load_reanalysis_field(paths["z"], "z", level=850, geopotential=True)
    assert z.attrs["units"] == "m"
    ref = case.z.sel(lat=slice(-20, 20))
    np.testing.assert_allclose(z.sel(lat=slice(-20, 20)).values, ref.values, rtol=1e-6, atol=1e-6)


def test_subdaily_ttr_becomes_daily_olr_with_factor(era5_files) -> None:
    paths, case = era5_files
    olr = load_reanalysis_field(paths["ttr"], "ttr", lat_range=(-20, 20), factor=-1.0 / 3600.0)
    assert olr.sizes["time"] == 200
    ref = case.olr.sel(lat=slice(-20, 20))
    np.testing.assert_allclose(olr.values, ref.values, rtol=1e-6, atol=1e-6)


def test_time_buffer_is_applied(era5_files) -> None:
    paths, _ = era5_files
    da = load_reanalysis_field(paths["u"], "u", level=850, time_range=("2000-03-01", "2000-04-01"), buffer_days=10)
    assert str(da.time.values[0])[:10] == "2000-02-20"
    assert str(da.time.values[-1])[:10] == "2000-04-11"


def test_regional_data_is_not_silently_accepted(era5_files) -> None:
    """A Pacific-only subset cannot be filtered: Kelvin waves need the full circle."""
    paths, _ = era5_files
    u = load_reanalysis_field(paths["u"], "u", level=850, regrid_deg=2.5)
    regional = u.sel(lon=slice(120, 280))
    with pytest.raises(ValueError, match="global"):
        run_regional_workflow(u=regional, config=CONFIG)


def test_regrid_requires_nothing_beyond_source_coverage(era5_files) -> None:
    paths, case = era5_files
    coarse = regrid_global(case.u.sel(lat=slice(-10, 10)), 2.5, lat_limit=20)
    assert float(coarse.lat.max()) <= 10.0 and float(coarse.lat.min()) >= -10.0
    assert np.isfinite(coarse.values).all()


def test_pacific_workflow_end_to_end(era5_files, tmp_path) -> None:
    paths, _ = era5_files
    kw = {"lat_range": (-20, 20), "regrid_deg": 2.5, "time_range": ("2000-01-01", "2000-07-18")}
    result = run_regional_workflow(
        u=load_reanalysis_field(paths["u"], "u", level=850, **kw),
        z=load_reanalysis_field(paths["z"], "z", level=850, geopotential=True, **kw),
        v=load_reanalysis_field(paths["v"], "v", level=850, **kw),
        olr=load_reanalysis_field(paths["ttr"], "ttr", factor=-1.0 / 3600.0, **kw),
        config=CONFIG,
        region_lon=(100.0, 300.0),
        time_range=("2000-01-01", "2000-07-18"),
        output_dir=tmp_path / "out",
    )
    events = result.catalog.events
    assert len(events) == 2 and (events["kind"] == "coupled").all()
    assert (events["conv_axis_lat"].between(6.0, 10.0)).all()
    assert (events["u_symmetry_index"] > 0.9).all()
    assert (tmp_path / "out" / "events.csv").exists()
    assert (tmp_path / "out" / "summary.csv").exists()
    assert result.figure is not None and result.figure.exists() and result.figure.stat().st_size > 10_000
    assert {"u", "z", "v", "olr"} == set(result.filtered)

    assert result.structure_figure is not None and result.structure_figure.exists()
    assert (tmp_path / "out" / "structure_summary.csv").exists()
    assert (tmp_path / "out" / "conv_profiles.csv").exists()
    structure = result.structure
    all_row = structure[(structure["kind"] == "coupled") & (structure["season"] == "ALL")].iloc[0]
    assert all_row["n"] == 2 and all_row["n_northern"] == 2 and all_row["frac_northern"] == 1.0

    summary = result.summary
    row = summary[(summary["kind"] == "coupled") & (summary["season"] == "ALL")].iloc[0]
    assert row["median_conv_peak_lat"] == pytest.approx(8.0, abs=1.5)
    assert row["n"] == 2
    assert row["median_conv_axis_lat"] == pytest.approx(8.0, abs=1.5)
    assert row["median_peak_ratio_axis_over_fixed"] > 1.8


def test_summary_handles_empty_and_unknown_columns() -> None:
    assert summarize_events(pd.DataFrame()).empty
    events = pd.DataFrame(
        {"kind": ["coupled", "dynamical_only"], "peak_time": pd.to_datetime(["2000-01-05", "2000-07-05"]), "speed_ms": [14.0, 16.0]}
    )
    out = summarize_events(events, years=1.0)
    assert set(out["season"]) == {"DJF", "JJA", "ALL"}
    assert "median_conv_axis_lat" not in out.columns


def test_cli_era5_style_run(era5_files, tmp_path, capsys) -> None:
    paths, _ = era5_files
    status = main(
        [
            "kelvin-events",
            "--u", str(paths["u"]), "--z", str(paths["z"]), "--v", str(paths["v"]),
            "--olr", str(paths["ttr"]), "--olr-var", "ttr", "--olr-factor", str(-1.0 / 3600.0),
            "--level", "850", "--regrid-deg", "2.5", "--z-geopotential",
            "--time-start", "2000-01-01", "--time-end", "2000-07-18",
            "--region-lon", "100", "300",
            "--keep-seasonal-cycle",
            "--peak-sigma", "3", "--continue-sigma", "1.5",
            "--dyn-min-amplitude", "1.5", "--conv-min-amplitude", "3",
            "--axis-lat-min", "0", "--axis-lat-max", "15",
            "--output-dir", str(tmp_path / "cli"),
        ]
    )
    assert status == 0
    events = pd.read_csv(tmp_path / "cli" / "events.csv")
    assert len(events) == 2 and (events["kind"] == "coupled").all()
    assert "coupled" in capsys.readouterr().out


def test_structure_summary_reports_mixed_classes() -> None:
    from tropical_wave_tools.kelvin_workflow import STRUCTURE_CLASSES, structure_summary

    events = pd.DataFrame(
        {
            "kind": ["coupled"] * 4,
            "peak_time": pd.to_datetime(["2000-01-05", "2000-01-20", "2000-07-05", "2000-07-09"]),
            "conv_structure": ["northern", "equatorial", "southern", "northern"],
        }
    )
    table = structure_summary(events)
    assert structure_summary(pd.DataFrame()).empty
    total = table[(table["season"] == "ALL")].iloc[0]
    assert total["n"] == 4 and total["n_northern"] == 2 and total["frac_equatorial"] == 0.25
    djf = table[table["season"] == "DJF"].iloc[0]
    assert djf["n"] == 2 and sum(djf[f"n_{c}"] for c in STRUCTURE_CLASSES) == 2
