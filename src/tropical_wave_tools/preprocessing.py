"""Shared preprocessing routines used by the spectral and filtering workflows."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import fft, signal
from typing import Tuple


def smooth_121(array: np.ndarray) -> np.ndarray:
    """Apply a 1-2-1 smoother while preserving edge length."""
    values = np.asarray(array, dtype=np.float64)
    if values.size == 0:
        return values
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.full_like(values, np.nan)
    if not np.all(mask):
        x_index = np.arange(values.size)
        values = np.interp(x_index, x_index[mask], values[mask])
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float64) / 4.0
    padded = np.r_[values[0], values, values[-1]]
    return np.convolve(padded, weights, mode="valid")


def detrend_with_mean(data: xr.DataArray) -> xr.DataArray:
    """Remove the linear trend while preserving the mean state."""
    mean_field = data.mean(dim="time")
    detrended = signal.detrend(data.values, axis=0, type="linear")
    return xr.DataArray(detrended, dims=data.dims, coords=data.coords, attrs=data.attrs) + mean_field


def remove_annual_cycle_fft(
    data: xr.DataArray,
    *,
    samples_per_day: float,
    freq_cutoff: float,
) -> xr.DataArray:
    """
    Remove low-frequency variability using an FFT filter.

    This follows the same intent as the original project: keep the physical
    interpretation of high-frequency tropical wave variability while attenuating
    the low-frequency seasonal signal.
    """
    n_time = int(data.sizes["time"])
    detrended = signal.detrend(data.values, axis=0)
    fourier_transform = fft.rfft(detrended, axis=0)
    frequencies = fft.rfftfreq(n_time, d=1.0 / float(samples_per_day))

    cutoff_indices = np.where(frequencies <= freq_cutoff)[0]
    if cutoff_indices.size > 1:
        cutoff_index = int(cutoff_indices.max())
        fourier_transform[1 : cutoff_index + 1, ...] = 0.0

    filtered = fft.irfft(fourier_transform, axis=0, n=n_time)
    return xr.DataArray(filtered, dims=data.dims, coords=data.coords, attrs=data.attrs)


def decompose_symmetric_antisymmetric(
    data: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Return the symmetric and antisymmetric meridional components."""
    lat_axis = data.get_axis_num("lat")
    flipped = np.flip(data.values, axis=lat_axis)
    symmetric = 0.5 * (data.values - flipped)
    antisymmetric = 0.5 * (data.values + flipped)
    symmetric_da = xr.DataArray(symmetric, dims=data.dims, coords=data.coords, attrs=data.attrs)
    antisymmetric_da = xr.DataArray(
        antisymmetric, dims=data.dims, coords=data.coords, attrs=data.attrs
    )
    return symmetric_da, antisymmetric_da


def build_wk_decomposition_layout(data: xr.DataArray) -> xr.DataArray:
    """
    Build the legacy WK array layout used by the original package.

    The original `wave_tools` implementation stored one component in the
    southern half and the other in the northern half before computing the
    wavenumber-frequency spectrum. This helper reproduces that behavior so the
    refactor does not change the scientific result definition.
    """
    symmetric, antisymmetric = decompose_symmetric_antisymmetric(data)
    result = data.copy(deep=True)
    half = data.sizes["lat"] // 2

    if data.sizes["lat"] % 2 == 0:
        result.values[:, :half, :] = symmetric.values[:, :half, :]
        result.values[:, half:, :] = antisymmetric.values[:, half:, :]
    else:
        result.values[:, :half, :] = symmetric.values[:, :half, :]
        result.values[:, half + 1 :, :] = antisymmetric.values[:, half + 1 :, :]
        result.values[:, half, :] = symmetric.values[:, half, :]

    return result


def extract_low_harmonics(
    climatology: xr.DataArray,
    *,
    n_harm: int = 3,
    dim: str = "dayofyear",
) -> xr.DataArray:
    """Retain the lowest harmonics of a daily climatology."""
    transformed = np.fft.rfft(climatology, axis=climatology.get_axis_num(dim))
    retained = transformed.copy()

    if n_harm < retained.shape[0]:
        retained[n_harm, ...] *= 0.5
        retained[n_harm + 1 :, ...] = 0.0

    reconstructed = np.fft.irfft(
        retained,
        n=climatology.sizes[dim],
        axis=climatology.get_axis_num(dim),
    ).real
    return xr.DataArray(
        reconstructed,
        coords=climatology.coords,
        dims=climatology.dims,
        attrs={
            **climatology.attrs,
            "smoothing": f"FFT low harmonics retained: 0..{n_harm}",
        },
    )


def remove_daily_climatology(
    data: xr.DataArray,
    *,
    n_harm: int = 3,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Remove the smoothed day-of-year climatology from a time series."""
    climatology = data.groupby("time.dayofyear").mean(dim="time")
    climatology_fit = extract_low_harmonics(climatology, n_harm=n_harm)
    anomaly = data.groupby("time.dayofyear") - climatology_fit
    return anomaly.transpose("time", "lat", "lon"), climatology_fit
