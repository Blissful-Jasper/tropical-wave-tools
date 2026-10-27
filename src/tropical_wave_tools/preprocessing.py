"""Shared preprocessing routines used by the spectral and filtering workflows."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import fft, signal


def validate_symmetric_latitude_grid(
    data: xr.DataArray,
    *,
    dim: str = "lat",
    atol: float = 1.0e-6,
) -> xr.DataArray:
    """Validate that a latitude grid can support equatorial symmetry decomposition."""
    if dim not in data.coords:
        raise ValueError(f"Input data must include a `{dim}` coordinate for symmetry decomposition.")
    if dim not in data.dims:
        raise ValueError(f"Input data must include `{dim}` as a dimension for symmetry decomposition.")

    lat_values = np.asarray(data[dim].values, dtype=float)
    if lat_values.ndim != 1 or lat_values.size < 2:
        raise ValueError("Latitude coordinate must be one-dimensional and contain at least two points.")
    if not np.all(np.isfinite(lat_values)):
        raise ValueError("Latitude coordinate contains non-finite values.")
    lat_diffs = np.diff(lat_values)
    if not (np.all(lat_diffs > 0.0) or np.all(lat_diffs < 0.0)):
        raise ValueError("Latitude coordinate must be strictly monotonic for symmetry decomposition.")

    pair_mismatch = np.abs(lat_values + lat_values[::-1])
    if not np.all(pair_mismatch <= atol):
        raise ValueError(
            "WK symmetric/antisymmetric decomposition requires latitude points "
            "paired about the equator. Select a symmetric latitude band such as "
            "(-15, 15) before running the decomposition."
        )
    return data


def nint(value: float) -> int:
    """Round half up, like NCL's ``round``/``nint`` (``np.round`` rounds half to even)."""
    return int(np.floor(value + 0.5))


def split_cosine_bell(length: int, fraction: float = 0.05) -> np.ndarray:
    """Split-cosine-bell taper as used by NCL (``fraction`` of the series tapered in total).

    ``fraction / 2`` of the series is tapered at each end with weights
    ``0.5 * (1 - cos(pi * (i + 0.5) / m))`` for ``i = 0..m-1``. This is *not* identical to
    ``scipy.signal.windows.tukey`` (end-point convention), which gives ~1% differences
    in filtered fields and spectra when compared with NCL.
    """
    window = np.ones(length)
    m = nint(length * fraction / 2.0)
    if m > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
        window[:m] = ramp
        window[length - m :] = ramp[::-1]
    return window


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
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return the symmetric and antisymmetric meridional components."""
    validate_symmetric_latitude_grid(data)
    lat_axis = data.get_axis_num("lat")
    flipped = np.flip(data.values, axis=lat_axis)
    symmetric = 0.5 * (data.values + flipped)
    antisymmetric = 0.5 * (data.values - flipped)
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
        result.values[:, :half, :] = antisymmetric.values[:, :half, :]
        result.values[:, half:, :] = symmetric.values[:, half:, :]
    else:
        result.values[:, :half, :] = antisymmetric.values[:, :half, :]
        result.values[:, half + 1 :, :] = symmetric.values[:, half + 1 :, :]
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
) -> tuple[xr.DataArray, xr.DataArray]:
    """Remove the smoothed day-of-year climatology from a time series."""
    climatology = data.groupby("time.dayofyear").mean(dim="time")
    climatology_fit = extract_low_harmonics(climatology, n_harm=n_harm)
    anomaly = data.groupby("time.dayofyear") - climatology_fit
    return anomaly.transpose("time", "lat", "lon"), climatology_fit
