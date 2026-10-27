"""Atlas variance fractions, monthly significance and the Case 10 lagged-regression pipeline.

References are known variance ratios, hand-computed area-weighted means, ``scipy`` regressions on
valid sample pairs and wave packets of known speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats as scipy_stats

from tropical_wave_tools import atlas

LAT = np.arange(-20.0, 20.1, 5.0)
LON = np.arange(0.0, 360.0, 15.0)


def _field(values: np.ndarray, periods: int, lat: np.ndarray = LAT, lon: np.ndarray = LON) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2001-01-01", periods=periods, freq="D"), "lat": lat, "lon": lon},
        name="x",
    )


# ------------------------------------------------------------------ variance fractions
def test_variance_fraction_reproduces_known_ratios() -> None:
    rng = np.random.default_rng(0)
    nt = 4 * 365
    raw = _field(rng.standard_normal((nt, LAT.size, LON.size)), nt)
    assert float(atlas.compute_monthly_variance_fraction_samples(raw, raw).mean()) == pytest.approx(100.0)
    assert float(atlas.compute_monthly_variance_fraction_samples(raw, 0.5 * raw).mean()) == pytest.approx(25.0)
    assert float(atlas.compute_monthly_variance_fraction_samples(raw, 0.0 * raw).mean()) == pytest.approx(0.0)

    component = rng.standard_normal(raw.shape)  # variance 1 inside a field of variance 4 -> 25%
    total = _field(component + np.sqrt(3.0) * rng.standard_normal(raw.shape), nt)
    fraction = atlas.compute_monthly_variance_fraction_samples(total, _field(component, nt))
    assert float(fraction.mean()) == pytest.approx(25.0, abs=2.0)  # monthly estimates scatter by ~9 points


@pytest.mark.parametrize(
    ("lat_range", "lon_range"),
    [((-10.0, 10.0), None), ((-10.0, 10.0), (60.0, 180.0)), ((-10.0, 10.0), (300.0, 60.0))],
)
def test_regional_variance_fraction_is_the_ratio_of_area_weighted_variances(lat_range, lon_range) -> None:
    rng = np.random.default_rng(1)
    nt = 2 * 365
    raw = _field(rng.standard_normal((nt, LAT.size, LON.size)), nt)
    filtered = _field(0.6 * raw.values + 0.3 * rng.standard_normal(raw.shape), nt)
    result = atlas.compute_monthly_variance_fraction_samples(raw, filtered, lat_range=lat_range, lon_range=lon_range)

    raw_var = raw.resample(time="MS").var("time").values
    filtered_var = filtered.resample(time="MS").var("time").values
    lat_mask = (lat_range[0] <= LAT) & (lat_range[1] >= LAT)
    if lon_range is None:
        lon_mask = np.ones(LON.size, dtype=bool)
    elif lon_range[0] <= lon_range[1]:
        lon_mask = (lon_range[0] <= LON) & (lon_range[1] >= LON)
    else:  # range wrapping through 0 degrees
        lon_mask = (lon_range[0] <= LON) | (lon_range[1] >= LON)
    weights = np.cos(np.deg2rad(LAT[lat_mask]))[:, None] * np.ones((1, int(lon_mask.sum())))
    expected = 100.0 * (filtered_var[:, lat_mask][:, :, lon_mask] * weights).sum((1, 2)) / (
        raw_var[:, lat_mask][:, :, lon_mask] * weights
    ).sum((1, 2))
    np.testing.assert_allclose(result.values, expected, rtol=1e-12)


# ------------------------------------------------------------------ monthly significance
def _seasonal_noise(nt: int, seed: int = 2) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    time = pd.date_range("2001-01-01", periods=nt, freq="D")
    amplitude = 1 + 0.5 * np.cos(2 * np.pi * (time.dayofyear.values - 200) / 365.25)
    return _field(rng.standard_normal((nt, LAT.size, LON.size)) * amplitude[:, None, None], nt)


def test_monthly_significance_is_a_t_test_across_years() -> None:
    data = _seasonal_noise(4 * 365)
    climatology, pvalue = atlas.compute_monthly_climatology_and_significance(data)

    monthly = atlas.compute_monthly_rms_samples(data)
    anomaly = monthly.groupby("time.year") - monthly.groupby("time.year").mean("time")
    for month in (1, 7):
        samples = anomaly.where(anomaly["time"].dt.month == month, drop=True).isel(lat=2, lon=3).values
        expected = scipy_stats.ttest_1samp(samples, 0.0).pvalue
        assert float(pvalue.sel(month=month).isel(lat=2, lon=3)) == pytest.approx(expected, rel=1e-9)
    assert float(climatology.sel(month=7).mean()) > float(climatology.sel(month=1).mean())  # summer maximum


def test_partial_years_do_not_bias_the_significance() -> None:
    """A record starting in July must give the same p-values as its complete years alone."""
    data = _seasonal_noise(4 * 365)
    partial = data.sel(time=slice("2001-07-01", "2004-12-31"))  # 2001 has only six months
    complete = data.sel(time=slice("2002-01-01", "2004-12-31"))
    _, p_partial = atlas.compute_monthly_climatology_and_significance(partial)
    _, p_complete = atlas.compute_monthly_climatology_and_significance(complete)
    np.testing.assert_allclose(p_partial.values, p_complete.values, rtol=1e-9)


# ------------------------------------------------------------------ Case 10 regression
@pytest.fixture(scope="module")
def case10_inputs() -> dict[str, object]:
    rng = np.random.default_rng(4)
    lat = np.arange(-20.0, 20.1, 2.5)
    lon = np.arange(0.0, 360.0, 5.0)
    nt, speed = 1200, 15.0
    deg_per_day = speed * 86400 / 111000.0
    period = 360.0 / (2 * deg_per_day)
    t = np.arange(nt)[:, None, None]
    lo = lon[None, None, :]
    envelope = np.exp(-((lat / 8.0) ** 2))[None, :, None]
    packets = np.zeros((nt, lat.size, lon.size))
    for centre in range(40, nt - 40, 50):  # packets cross the Kelvin base longitude (200E) at `centre`
        offset = ((lo - 200.0 - deg_per_day * (t - centre) + 180) % 360) - 180
        packets += (
            np.cos(2 * np.pi * (2 * (lo - 200.0) / 360 - (t - centre) / period))
            * np.exp(-((offset / 40.0) ** 2))
            * np.exp(-(((t - centre) / 12.0) ** 2))
            * envelope
        )
    coords = {"time": pd.date_range("2000-01-01", periods=nt), "lat": lat, "lon": lon}

    def wrap(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, dims=("time", "lat", "lon"), coords=coords, attrs={"units": "W m-2"})

    filtered = wrap(-25.0 * packets)
    raw = wrap(-45.0 * packets + 4.0 * rng.standard_normal(packets.shape))
    result = atlas.compute_case10_regression_hovmoller(raw, filtered, wave_name="kelvin")
    return {"raw": raw, "filtered": filtered, "result": result, "deg_per_day": deg_per_day, "lon": lon}


def test_case10_matches_scipy_regressions_on_the_active_samples(case10_inputs) -> None:
    result, raw, filtered = case10_inputs["result"], case10_inputs["raw"], case10_inputs["filtered"]
    lat_band = result["shading"].attrs["lat_band"]
    reference = filtered.sel(lat=7.0, lon=200.0, method="nearest")
    predictor = ((reference - reference.mean()) / float(reference.std())).values
    sources = {
        "shading": (-raw).sel(lat=slice(*lat_band)).mean("lat"),  # convective proxy = -OLR anomaly
        "contours": filtered.sel(lat=slice(*lat_band)).mean("lat"),
    }
    active = np.isfinite(result["active_mask"].values)

    for name, source in sources.items():
        for lag in result[name].lag.values[::5]:
            shifted = source.shift(time=-int(lag)).values
            for lon_value in result[name].lon.values[::6]:
                column = int(np.argmin(np.abs(source.lon.values - lon_value)))
                valid = active & np.isfinite(shifted[:, column])
                expected = scipy_stats.linregress(predictor[valid], shifted[valid, column]).slope
                assert float(result[name].sel(lag=int(lag), lon=lon_value)) == pytest.approx(expected, abs=1e-9)


def test_case10_crest_moves_east_with_positive_lag_at_the_true_speed(case10_inputs) -> None:
    result, deg_per_day = case10_inputs["result"], case10_inputs["deg_per_day"]
    shading = result["shading"]
    lags = np.arange(-6, 7, 3)
    crest = []
    for lag in lags:
        row = shading.sel(lag=int(lag))
        near_track = np.abs(row["lon"] - (200.0 + deg_per_day * lag)) < 45
        crest.append(float(row.where(near_track).idxmin("lon")))  # proxy is -OLR and the reference is OLR
    speed = np.polyfit(lags, crest, 1)[0] * 111000.0 / 86400.0
    assert speed == pytest.approx(15.0, rel=0.1)
    # at lag 0 the extreme sits at the base longitude, and it is negative (reference = +1 sigma OLR)
    row = shading.sel(lag=0)
    assert float(row.idxmin("lon")) == pytest.approx(200.0, abs=5.0)
    assert float(row.min()) < 0.0 < float(result["contours"].sel(lag=0, lon=200.0))


def test_case10_outputs_document_their_regression_convention(case10_inputs) -> None:
    shading, contours = case10_inputs["result"]["shading"], case10_inputs["result"]["contours"]
    assert "per standard deviation" in shading.attrs["units"]
    assert "suppressed" in shading.attrs["predictor"]
    assert "positive lag" in shading.attrs["lag_convention"]
    assert contours.attrs["long_name"] != shading.attrs["long_name"]
    assert shading.sizes["lag"] == len(atlas.CASE10_LAGS["kelvin"])
    lon_min, lon_max = atlas.CASE10_LON_WINDOWS["kelvin"]
    assert float(shading["lon"].min()) >= lon_min and float(shading["lon"].max()) <= lon_max
