"""``io`` helpers, ``workflows`` and the command-line entry point against known answers."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

from tropical_wave_tools import cli, workflows
from tropical_wave_tools import io as tio
from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.sample_data import get_sample_path, open_example_olr
from tropical_wave_tools.spectral import analyze_wk_spectrum
from tropical_wave_tools.stats import linear_trend

LAT = np.arange(-10.0, 10.1, 5.0)
LON = np.arange(0.0, 360.0, 30.0)


def _cube(nt: int = 60, *, lat=LAT, lon=LON, time=None, name: str = "olr", seed: int = 0, attrs=None) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    time = pd.date_range("2000-01-01", periods=nt) if time is None else time
    return xr.DataArray(
        rng.standard_normal((len(time), len(lat), len(lon))),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name=name,
        attrs=attrs or {},
    )


# ------------------------------------------------------------------------------- longitudes
def test_cyclic_longitude_duplicate_is_removed() -> None:
    cyclic = _cube(lon=np.append(LON, 360.0))
    standardized = tio.standardize_data(cyclic)
    assert standardized.sizes["lon"] == LON.size
    assert not pd.Index(standardized["lon"].values).has_duplicates
    np.testing.assert_allclose(standardized["lon"].values, LON)

    both_ends = _cube(lon=np.append(np.arange(-180.0, 180.0, 30.0), 180.0))
    wrapped = tio.normalize_longitude(both_ends, target="-180_180")
    assert not pd.Index(wrapped["lon"].values).has_duplicates
    assert wrapped.sizes["lon"] == 12


def test_longitude_round_off_does_not_produce_360() -> None:
    result = tio.normalize_longitude(_cube(lon=np.array([-1e-14, 30.0, 60.0])))
    assert float(result["lon"].max()) < 360.0
    assert result["lon"].values.tolist() == [0.0, 30.0, 60.0]


def test_filtering_is_unaffected_by_a_stored_cyclic_point() -> None:
    """A duplicated 0/360 column used to enter the zonal FFT as an extra grid point."""
    base = _cube(nt=120, seed=3)
    cyclic = xr.concat([base, base.isel(lon=[0]).assign_coords(lon=[360.0])], dim="lon")
    plain = filter_wave_signal(tio.standardize_data(base), wave_name="kelvin", n_workers=1)
    wrapped = filter_wave_signal(tio.standardize_data(cyclic), wave_name="kelvin", n_workers=1)
    np.testing.assert_allclose(wrapped.values, plain.values, atol=1e-12)


# ------------------------------------------------------------------------------ valid_range
def _packed_file(tmp_path, valid_range, name="packed.nc"):
    rng = np.random.default_rng(1)
    physical = 300 + 50 * rng.standard_normal((10, 3, 4))
    dataset = xr.Dataset(
        {"z": (("time", "lat", "lon"), physical)},
        coords={"time": pd.date_range("2000-01-01", periods=10), "lat": [-5.0, 0.0, 5.0], "lon": [0.0, 90.0, 180.0, 270.0]},
    )
    dataset["z"].attrs["valid_range"] = np.array(valid_range, dtype="int16")
    path = tmp_path / name
    encoding = {"z": {"dtype": "int16", "scale_factor": 0.01, "add_offset": 300.0, "_FillValue": -32768}}
    dataset.to_netcdf(path, encoding=encoding)
    return path


def test_valid_range_of_packed_data_is_interpreted_in_packed_units(tmp_path) -> None:
    """CF: valid_range of packed data is in packed units; physical = packed * scale + offset."""
    path = _packed_file(tmp_path, (-1000, 1000))  # = 290..310 in physical units
    loaded = tio.load_dataarray(path, variable="z")
    kept = loaded.values[np.isfinite(loaded.values)]
    assert kept.size > 0 and kept.size < loaded.size
    assert kept.min() >= 290.0 - 1e-6 and kept.max() <= 310.0 + 1e-6

    wide = tio.load_dataarray(_packed_file(tmp_path, (-32000, 32000), "wide.nc"), variable="z")
    assert int(wide.notnull().sum()) == wide.size  # a generous packed range keeps everything


def test_valid_range_of_unpacked_data_still_masks() -> None:
    data = _cube(attrs={"valid_range": [-0.5, 0.5]})
    masked = tio.to_dataarray(data)
    inside = np.abs(data.values) <= 0.5
    np.testing.assert_array_equal(np.isfinite(masked.values), inside)


# ----------------------------------------------------------------------- variable inference
def test_infer_variable_name_prefers_the_gridded_variable() -> None:
    rng = np.random.default_rng(0)
    dataset = xr.Dataset(
        {
            "crs": ((), 0),
            "time_bnds": (("time", "nbnds"), np.zeros((5, 2))),
            "olr": (("time", "lat", "lon"), rng.standard_normal((5, 3, 4))),
        },
        coords={"time": pd.date_range("2000-01-01", periods=5), "lat": [-5.0, 0.0, 5.0], "lon": np.arange(4) * 90.0},
    )
    assert tio.infer_variable_name(dataset) == "olr"
    only_bounds = xr.Dataset({"lat_bnds": (("lat", "nb"), np.zeros((3, 2)))}, coords={"lat": [0.0, 1.0, 2.0]})
    with pytest.raises(tio.InvalidDataArrayError):
        tio.infer_variable_name(only_bounds)


# ---------------------------------------------------------------------------------- saving
def test_saved_attributes_survive_a_netcdf_round_trip(tmp_path) -> None:
    data = _cube(
        attrs={
            "units": "W m-2",
            "depth": (None, None),
            "partial": (3.0, None),
            "flag": True,
            "nested": {"a": 1},
            "period": (3.0, 20.0),
            "missing": None,
        }
    )
    path = tio.save_dataarray(data, tmp_path / "out.nc")
    attrs = xr.open_dataarray(path).attrs
    assert attrs["units"] == "W m-2"
    assert "depth" not in attrs and "missing" not in attrs
    assert attrs["flag"] == 1
    assert json.loads(attrs["nested"]) == {"a": 1}
    np.testing.assert_allclose(attrs["period"], [3.0, 20.0])
    assert np.isnan(attrs["partial"][1]) and attrs["partial"][0] == 3.0

    dataset_path = tio.save_dataset(data.to_dataset(name="olr"), tmp_path / "ds.nc")
    assert xr.open_dataset(dataset_path)["olr"].attrs["flag"] == 1


# ------------------------------------------------------------------------ time axes / CMIP
@pytest.mark.parametrize(("calendar", "year_length"), [("noleap", 365), ("360_day", 360)])
def test_non_standard_calendars_work_in_trends_and_events(calendar: str, year_length: int) -> None:
    time = xr.date_range("2000-01-01", periods=3 * year_length, freq="D", calendar=calendar, use_cftime=True)
    days = tio.time_axis_in_days(xr.DataArray(time, dims="time"))
    np.testing.assert_allclose(days, np.arange(time.size, dtype=float))

    series = xr.DataArray(0.01 * np.arange(time.size) + 2.0, dims="time", coords={"time": time})
    trend = linear_trend(series)
    assert float(trend["slope"]) == pytest.approx(0.01, rel=1e-9)  # per day
    assert trend["slope"].attrs["units"].endswith("/ day")

    from tropical_wave_tools import atlas

    hourly = xr.date_range("2000-01-01", periods=400, freq="6h", calendar=calendar, use_cftime=True)
    values = np.zeros(400)
    values[100], values[112] = -5.0, -4.0  # 12 samples = 3 days apart
    field = _cube(time=hourly, lat=np.array([-5.0, 0.0, 5.0]))
    field = field * 0 + values[:, None, None]
    events, _ = atlas.detect_wave_events(field, wave_name="kelvin", lon_ref=180.0, threshold_std=0.5)
    assert events.tolist() == [100]  # 3 days < 7 days: merged


def test_time_deltas_agree_for_numpy_and_cftime_axes() -> None:
    numpy_axis = pd.date_range("2000-01-01", periods=5, freq="12h").values
    cftime_axis = xr.date_range("2000-01-01", periods=5, freq="12h", calendar="noleap", use_cftime=True).values
    np.testing.assert_allclose(tio.time_deltas_in_days(numpy_axis), 0.5)
    np.testing.assert_allclose(tio.time_deltas_in_days(cftime_axis), 0.5)


# ------------------------------------------------------------------------ loading / selecting
def test_load_dataarray_selects_by_range_and_handles_wrapping_and_descending_axes(tmp_path) -> None:
    data = _cube(nt=40, lat=np.arange(20.0, -20.1, -5.0), lon=np.arange(0.0, 360.0, 30.0))  # descending lat
    path = tmp_path / "grid.nc"
    data.to_dataset(name="olr").to_netcdf(path)

    loaded = tio.load_dataarray(path, variable="olr", lat_range=(-10, 10), time_range=("2000-01-05", "2000-01-14"))
    assert loaded.dims == ("time", "lat", "lon")
    assert loaded["lat"].values.tolist() == [-10.0, -5.0, 0.0, 5.0, 10.0]  # sorted ascending, selected
    assert loaded.sizes["time"] == 10

    wrapped = tio.load_dataarray(path, variable="olr", lon_range=(300.0, 60.0))
    assert sorted(wrapped["lon"].values.tolist()) == [0.0, 30.0, 60.0, 300.0, 330.0]
    values = tio.load_dataarray(path, variable="olr").sel(lat=0.0, lon=30.0).values
    np.testing.assert_allclose(values, data.sel(lat=0.0, lon=30.0).values)  # values follow their coordinates


def test_describe_dataarray_summarises_extent() -> None:
    summary = tio.describe_dataarray(_cube(nt=10))
    assert summary["shape"] == (10, 5, 12)
    assert summary["lat_min"] == -10.0 and summary["lat_max"] == 10.0
    assert summary["time_start"].startswith("2000-01-01") and summary["time_end"].startswith("2000-01-10")


# ------------------------------------------------------------------------------- workflows
def test_spatial_metrics_match_numpy_and_use_the_jointly_valid_points() -> None:
    rng = np.random.default_rng(0)
    lat, lon = np.arange(-10.0, 10.1, 5.0), np.arange(0.0, 360.0, 60.0)
    a = xr.DataArray(rng.random((lat.size, lon.size)) + 1, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    b = a * 0.9 + 0.05 * rng.standard_normal(a.shape)
    metrics = workflows.compute_spatial_metrics(a, b)
    difference = (a - b).values.ravel()
    assert metrics["mean_bias"] == pytest.approx(difference.mean())
    assert metrics["mean_abs_diff"] == pytest.approx(np.abs(difference).mean())
    assert metrics["rmse"] == pytest.approx(np.sqrt((difference**2).mean()))
    assert metrics["pattern_corr"] == pytest.approx(np.corrcoef(a.values.ravel(), b.values.ravel())[0, 1])

    gappy = b.copy()
    gappy[0, :] = np.nan
    metrics = workflows.compute_spatial_metrics(a, gappy)
    assert metrics["legacy_mean_std"] - metrics["cckw_mean_std"] == pytest.approx(metrics["mean_bias"])
    assert np.isnan(workflows.compute_spatial_metrics(a * np.nan, b)["rmse"])


def test_legacy_and_cckw_filters_agree_on_resolvable_waves_and_outputs_are_written(tmp_path) -> None:
    summary = workflows.compare_filter_spatial_fields(
        get_sample_path(),
        waves=("kelvin", "er"),
        time_range=("1979-01-01", "1979-12-31"),
        lat_range=(-15.0, 15.0),
        n_workers=1,
        use_parallel=False,
        output_dir=tmp_path,
    )
    assert summary["wave"].tolist() == ["kelvin", "er"]
    np.testing.assert_allclose(summary["rmse"].values, 0.0, atol=1e-12)  # the two share one implementation
    np.testing.assert_allclose(summary["pattern_corr"].values, 1.0, atol=1e-12)
    assert (tmp_path / "kelvin_spatial_std_compare.png").exists()
    assert (tmp_path / "er_spatial_std_comparison.nc").exists()
    assert len(pd.read_csv(tmp_path / "wave_filter_spatial_comparison_summary.csv")) == 2


def test_unresolvable_bands_warn_instead_of_silently_returning_zeros() -> None:
    olr = open_example_olr()
    with pytest.warns(RuntimeWarning, match="not resolvable"):
        zeros = filter_wave_signal(olr, wave_name="eig", method="cckw", n_workers=1)
    assert float(abs(zeros).max()) == 0.0
    assert "not resolvable" in zeros.attrs["note"]
    clipped = filter_wave_signal(olr, wave_name="eig", method="legacy", use_parallel=False)
    assert float(clipped.std()) > 0.0  # the NCL-style core clips the band at the Nyquist frequency


def test_wk_workflow_matches_the_direct_call_and_saved_file(tmp_path) -> None:
    result, summary = workflows.analyze_wk_spectrum_from_file(
        get_sample_path(), time_range=("1979-01-01", "1979-12-31"), output_dir=tmp_path
    )
    direct = analyze_wk_spectrum(
        tio.load_dataarray(get_sample_path(), variable="olr", lat_range=(-15, 15), time_range=("1979-01-01", "1979-12-31"))
    )
    np.testing.assert_allclose(result.power_symmetric.values, direct.power_symmetric.values, equal_nan=True)
    saved = xr.open_dataset(tmp_path / "wk_spectrum.nc")
    np.testing.assert_allclose(saved["power_symmetric"].values, result.power_symmetric.values, equal_nan=True)
    assert (tmp_path / "wk_spectrum.png").exists()
    assert summary["shape"][0] == 365


def test_demo_subset_and_local_copy(tmp_path) -> None:
    out = workflows.create_demo_subset(
        get_sample_path(), tmp_path / "sub" / "sample.nc", time_range=("1979-02-01", "1979-12-31"), lat_range=(-10, 10)
    )
    subset = tio.load_dataarray(out, variable="olr")
    assert subset.sizes["time"] == 334 and float(subset.lat.min()) == -10.0 and float(subset.lat.max()) == 10.0
    copied = workflows.install_local_data_copy(get_sample_path(), tmp_path / "local")
    assert copied.read_bytes() == get_sample_path().read_bytes()


# ------------------------------------------------------------------------------------- CLI
def test_resolve_time_range_cases() -> None:
    assert cli._resolve_time_range(None, None) is None
    assert cli._resolve_time_range("1980-01-01", "1999-12-31") == ("1980-01-01", "1999-12-31")
    assert cli._resolve_time_range("1980-01-01", "1999-12-31", 10) == ("1990-01-01", "1999-12-31")
    assert cli._resolve_time_range("1995-01-01", "1999-12-31", 10) == ("1995-01-01", "1999-12-31")  # clipped to the start
    with pytest.raises(ValueError, match="demo_years"):
        cli._resolve_time_range(None, None, 10)


def test_cli_commands_run_end_to_end(tmp_path, monkeypatch, capsys) -> None:
    source = str(get_sample_path())
    assert cli.main(["wk-spectrum", "--input", source, "--output-dir", str(tmp_path / "wk")]) == 0
    assert (tmp_path / "wk" / "wk_spectrum.nc").exists()

    for wave, method in (("kelvin", "cckw"), ("td", "legacy"), ("mjo", "legacy")):  # legacy td/mjo failed to save
        out = tmp_path / f"{wave}_{method}.nc"
        code = cli.main(
            ["filter-wave", "--input", source, "--wave", wave, "--method", method, "--output", str(out),
             "--n-workers", "1", "--n-jobs", "1"]
        )
        assert code == 0
        saved = xr.open_dataarray(out)
        assert saved.dims == ("time", "lat", "lon") and float(saved.std()) > 0.0

    code = cli.main(
        ["compare-filters", "--input", source, "--waves", "kelvin", "--time-start", "1979-01-01", "--time-end",
         "1979-12-31", "--n-workers", "1", "--n-jobs", "1", "--output-dir", str(tmp_path / "cmp")]
    )
    assert code == 0 and (tmp_path / "cmp" / "wave_filter_spatial_comparison_summary.csv").exists()

    target = tmp_path / "subset.nc"
    code = cli.main(
        ["prepare-sample-data", "--source", source, "--time-start", "1979-03-01", "--time-end", "1979-05-31",
         "--output", str(target)]
    )
    assert code == 0 and target.exists()
    assert tio.load_dataarray(target, variable="olr").sizes["time"] == 92
    assert "Created sample subset" in capsys.readouterr().out

    # the default output is still the packaged sample (and must not be touched by this test)
    monkeypatch.setattr(cli, "get_sample_path", lambda *a, **k: tmp_path / "default_target.nc")
    cli.main(["prepare-sample-data", "--source", source])
    assert (tmp_path / "default_target.nc").exists()


def test_cli_rejects_demo_years_without_dates(tmp_path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["wk-spectrum", "--input", str(get_sample_path()), "--output-dir", str(tmp_path), "--demo-years", "10"])
