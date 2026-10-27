"""Physical-consistency checks for the Wheeler-Kiladis filter and spectrum.

These tests use synthetic plane waves whose zonal wavenumber ``k`` and period
``P`` are known exactly, so pass/reject behaviour can be verified analytically
without any reference dataset. Cross-checks against NCL ``kf_filter`` outputs
live in ``validation/`` and are exercised by ``test_ncl_reference.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tropical_wave_tools.config import SpectralConfig
from tropical_wave_tools.filters import WaveFilter
from tropical_wave_tools.spectral import analyze_wk_spectrum

NT, NLON = 360, 144


def _plane_wave(k: int, period: float, *, wrap: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, lon)``; ``k > 0`` propagates eastward."""
    lon = np.arange(NLON) * 360.0 / NLON
    if wrap:
        lon = np.append(lon, 360.0)
    t = np.arange(NT)[:, None]
    return np.cos(2 * np.pi * (k * lon[None, :] / 360.0 - t / period)), lon


def _gain(wave: str, k: int, period: float, *, wrap: bool = False) -> float:
    wave_filter = WaveFilter()
    spec = wave_filter.get_wave_params(wave)
    x, lon = _plane_wave(k, period, wrap=wrap)
    y = wave_filter._kf_filter(
        x,
        lon=lon,
        obs_per_day=1,
        period_days=spec.period_days,
        wavenumber=spec.wavenumber,
        equivalent_depth=spec.equivalent_depth,
        wave_name=wave,
        meridional_mode=spec.meridional_mode,
    )
    inner = slice(40, -40)  # skip taper edges
    return float(np.std(y[inner]) / np.std(x[inner]))


@pytest.mark.parametrize(
    ("wave", "k", "period"),
    [
        ("kelvin", 5, 8.0),
        ("er", -3, 20.0),
        ("mjo", 2, 40.0),
        ("td", -10, 4.0),
        ("mrg", -4, 5.0),
    ],
)
@pytest.mark.parametrize("wrap", [False, True])
def test_in_band_wave_is_preserved(wave: str, k: int, period: float, wrap: bool) -> None:
    assert _gain(wave, k, period, wrap=wrap) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize(
    ("wave", "k", "period"),
    [
        ("kelvin", -5, 8.0),  # wrong propagation direction
        ("kelvin", 5, 40.0),  # too slow
        ("er", 3, 20.0),  # eastward
        ("mjo", -2, 40.0),  # westward
    ],
)
def test_out_of_band_wave_is_rejected(wave: str, k: int, period: float) -> None:
    assert _gain(wave, k, period) < 0.02


def test_cyclic_point_does_not_corrupt_filtered_field() -> None:
    """Regression: a duplicated 0/360 column used to scramble the output."""
    wave_filter = WaveFilter()
    spec = wave_filter.get_wave_params("kelvin")
    x, lon = _plane_wave(5, 8.0, wrap=True)
    y = wave_filter._kf_filter(
        x,
        lon=lon,
        obs_per_day=1,
        period_days=spec.period_days,
        wavenumber=spec.wavenumber,
        equivalent_depth=spec.equivalent_depth,
        wave_name="kelvin",
    )
    assert y.shape == x.shape
    assert np.abs(y[:, 0] - y[:, -1]).max() < 1e-10
    assert np.sqrt(np.mean((y - x)[40:-40] ** 2)) < 0.05


def test_wk_spectrum_peak_sign_convention() -> None:
    """Eastward wave -> positive wavenumber; westward -> negative."""
    time = pd.date_range("2000-01-01", periods=384, freq="D")
    lat = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    lon = np.arange(0.0, 360.0, 5.0)
    config = SpectralConfig(window_size_days=96, window_skip_days=-48)

    for k_true in (4, -4):
        phase = 2 * np.pi * (
            k_true * lon[None, None, :] / 360.0 - np.arange(time.size)[:, None, None] / 12.0
        )
        field = xr.DataArray(
            np.cos(phase) * np.ones((1, lat.size, 1)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
        )
        power = analyze_wk_spectrum(field, config=config).power_symmetric
        peak = power.fillna(0).stack(z=("frequency", "wavenumber")).idxmax("z").item()
        assert peak[0] == pytest.approx(1 / 12.0, abs=1.0 / 96)
        assert peak[1] == k_true


def test_split_cosine_bell_matches_ncl_definition() -> None:
    from tropical_wave_tools.filters import split_cosine_bell

    window = split_cosine_bell(360, 0.05)  # m = 9 points tapered at each end
    ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(9) + 0.5) / 9))
    np.testing.assert_allclose(window[:9], ramp)
    np.testing.assert_allclose(window[-9:], ramp[::-1])
    assert np.all(window[9:-9] == 1.0)


def test_period_cutoff_excludes_bins_outside_the_band() -> None:
    """365 d / 100 d = 3.65 -> bin 4 (91 d); the 121.7 d bin (index 3) must be dropped."""
    wave_filter = WaveFilter()
    spec = wave_filter.get_wave_params("mjo")
    nt = 365
    t = np.arange(nt)[:, None]
    lon = np.arange(NLON) * 360.0 / NLON
    # exactly bin 3 (period 121.7 d), k = 2: outside 20-100 d
    x = np.cos(2 * np.pi * (2 * lon[None, :] / 360.0 - 3 * t / nt))
    y = wave_filter._kf_filter(
        x,
        lon=lon,
        obs_per_day=1,
        period_days=spec.period_days,
        wavenumber=spec.wavenumber,
        equivalent_depth=spec.equivalent_depth,
        wave_name="mjo",
    )
    # the taper leaks some energy into neighbouring in-band bins, so inspect bin 3 itself
    spectrum = np.abs(np.fft.rfft(y, axis=0))
    assert spectrum[3].max() < 1e-8 * spectrum.max()
    assert spectrum[4].max() > 1e-3 * spectrum.max()  # in-band neighbour is retained


@pytest.mark.parametrize(
    ("length", "fraction", "points"),
    [(360, 0.05, 9), (365, 0.05, 9), (382, 0.05, 9), (96, 0.1, 5)],
)
def test_split_cosine_bell_length_matches_ncl(length: int, fraction: float, points: int) -> None:
    """Taper lengths fixed by NCL output (kf_filter and wkSpaceTime golden files)."""
    from tropical_wave_tools.filters import split_cosine_bell

    window = split_cosine_bell(length, fraction)
    assert np.count_nonzero(window < 1.0) == points * 2
