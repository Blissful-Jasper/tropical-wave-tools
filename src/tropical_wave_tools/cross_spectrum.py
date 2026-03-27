"""Cross-spectrum analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import warnings

import numpy as np
import xarray as xr
from scipy import fft, signal


@dataclass
class CrossSpectrumConfig:
    """Configuration parameters for wavenumber-frequency cross-spectrum analysis."""

    segment_length: int = 96
    segment_overlap: int = -65
    taper_fraction: float = 0.10
    smooth_factor: float = 2.667
    freq_cutoff: float = 1.0 / 365.0
    samples_per_day: int = 1
    prob_levels: np.ndarray = field(
        default_factory=lambda: np.array([0.80, 0.85, 0.90, 0.925, 0.95, 0.99])
    )


def nan_to_value_by_interp_3D(values: np.ndarray) -> np.ndarray:
    """Fill NaNs in a ``(time, lat, lon)`` array using neighboring values."""
    interpolated = np.where(np.isnan(values), np.nan, values)
    nan_locations = np.argwhere(np.isnan(values))

    if nan_locations.size == 0:
        return interpolated

    nt, nlat, nlon = values.shape
    for t_index, lat_index, lon_index in nan_locations:
        previous_value = values[t_index - 1, lat_index, lon_index] if t_index > 0 else np.nan
        next_value = values[t_index + 1, lat_index, lon_index] if t_index < nt - 1 else np.nan
        north_value = values[t_index, lat_index + 1, lon_index] if lat_index < nlat - 1 else np.nan
        south_value = values[t_index, lat_index - 1, lon_index] if lat_index > 0 else np.nan
        east_value = values[t_index, lat_index, lon_index + 1] if lon_index < nlon - 1 else values[t_index, lat_index, 0]
        west_value = values[t_index, lat_index, lon_index - 1] if lon_index > 0 else values[t_index, lat_index, -1]
        interpolated[t_index, lat_index, lon_index] = np.nanmean(
            np.array([east_value, west_value, north_value, south_value, previous_value, next_value])
        )
    return interpolated


def remove_annual_cycle(
    data: Union[xr.DataArray, np.ndarray],
    *,
    spd: int = 1,
    fCrit: float = 1.0 / 365.0,
) -> Union[xr.DataArray, np.ndarray]:
    """Remove linear trend and low-frequency annual-cycle variability."""
    is_xarray = isinstance(data, xr.DataArray)
    if is_xarray:
        dims = data.dims
        coords = data.coords
        values = data.values
    else:
        values = np.asarray(data)

    ntim = values.shape[0]
    detrended = signal.detrend(values, axis=0)
    rf = fft.rfft(detrended, axis=0)
    freq = fft.rfftfreq(ntim, d=1.0 / float(spd))
    cutoff_indices = np.argwhere(freq <= fCrit)
    if cutoff_indices.size > 0:
        cutoff_index = int(cutoff_indices.max())
        if cutoff_index > 1:
            rf[1 : cutoff_index + 1, ...] = 0.0
    filtered = fft.irfft(rf, axis=0, n=ntim)

    if is_xarray:
        return xr.DataArray(filtered, dims=dims, coords=coords, attrs=data.attrs)
    return filtered


def _smooth121_1D(array_in: np.ndarray) -> np.ndarray:
    """Apply the classic 1-2-1 smoother to a one-dimensional array."""
    temp = np.copy(array_in)
    array_out = np.copy(temp) * 0.0

    for index in range(len(temp)):
        if np.isnan(temp[index]):
            array_out[index] = np.nan
        elif index == 0 or np.isnan(temp[index - 1]):
            array_out[index] = (3 * temp[index] + temp[index + 1]) / 4
        elif index == len(temp) - 1 or np.isnan(temp[index + 1]):
            array_out[index] = (3 * temp[index] + temp[index - 1]) / 4
        else:
            array_out[index] = (temp[index + 1] + 2 * temp[index] + temp[index - 1]) / 4

    return array_out


def _smooth121_frequency(spectra: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Apply 1-2-1 smoothing along the frequency dimension."""
    _, _, nwave = spectra.shape
    zero_frequency = np.where(freq == 0)[0]
    if zero_frequency.size == 0:
        return spectra

    zero_index = int(zero_frequency[0])
    for wave_index in range(nwave):
        for variable_index in range(4):
            spectra[variable_index, zero_index + 1 :, wave_index] = _smooth121_1D(
                spectra[variable_index, zero_index + 1 :, wave_index]
            )
    return spectra


def _get_symm_asymm(values: np.ndarray, lat: np.ndarray, mode: str = "symm") -> np.ndarray:
    """Extract the symmetric or antisymmetric meridional component."""
    _, nlat, _ = values.shape

    if mode == "symm":
        component = values[:, lat[:] >= 0, :]
        if len(lat) % 2 == 1:
            for lat_index in range(nlat // 2 + 1):
                component[:, lat_index, :] = 0.5 * (
                    values[:, lat_index, :] + values[:, nlat - lat_index - 1, :]
                )
        else:
            for lat_index in range(nlat // 2):
                component[:, lat_index, :] = 0.5 * (
                    values[:, lat_index, :] + values[:, nlat - lat_index - 1, :]
                )
        return component

    if mode in {"asymm", "anti-symm"}:
        component = values[:, lat[:] > 0, :]
        for lat_index in range(nlat // 2):
            component[:, lat_index, :] = 0.5 * (
                values[:, lat_index, :] - values[:, nlat - lat_index - 1, :]
            )
        return component

    raise ValueError(f"Invalid mode: {mode}. Must be 'symm' or 'asymm'.")


def _cross_spectrum_segment(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Compute one FFT segment of the cross-spectrum."""
    ntim, _, nlon = xx.shape

    xx = np.transpose(xx, axes=[1, 2, 0])
    yy = np.transpose(yy, axes=[1, 2, 0])

    xfft = np.fft.rfft2(xx, axes=(1, 2))
    yfft = np.fft.rfft2(yy, axes=(1, 2))

    xfft = np.transpose(xfft, axes=[2, 0, 1]) / (ntim * nlon)
    yfft = np.transpose(yfft, axes=[2, 0, 1]) / (ntim * nlon)

    xfft = np.fft.fftshift(xfft, axes=(2,))
    yfft = np.fft.fftshift(yfft, axes=(2,))

    power_x = np.average(np.square(np.abs(xfft)), axis=1)
    power_y = np.average(np.square(np.abs(yfft)), axis=1)
    cross = np.conj(xfft) * yfft
    co_spectrum = np.average(np.real(cross), axis=1)
    quad_spectrum = np.average(np.imag(cross), axis=1)

    power_x = power_x[:, ::-1]
    power_y = power_y[:, ::-1]
    co_spectrum = co_spectrum[:, ::-1]
    quad_spectrum = quad_spectrum[:, ::-1]

    nfreq = (ntim + 1) // 2 if ntim % 2 == 1 else ntim // 2 + 1
    nwave = nlon if nlon % 2 == 1 else nlon + 1
    output = np.zeros([8, nfreq, nwave], dtype="double")

    if nlon % 2 == 1:
        output[0, :nfreq, :nlon] = power_x
        output[1, :nfreq, :nlon] = power_y
        output[2, :nfreq, :nlon] = co_spectrum
        output[3, :nfreq, :nlon] = quad_spectrum
    else:
        output[0, :nfreq, 1 : nlon + 1] = power_x
        output[1, :nfreq, 1 : nlon + 1] = power_y
        output[2, :nfreq, 1 : nlon + 1] = co_spectrum
        output[3, :nfreq, 1 : nlon + 1] = quad_spectrum
        output[:, :, 0] = output[:, :, nlon]
    return output


def _compute_coherence_phase(spectra: np.ndarray) -> np.ndarray:
    """Compute coherence, phase, and vector components from averaged spectra."""
    power_x = spectra[0, :, :]
    power_y = spectra[1, :, :]
    co_spectrum = spectra[2, :, :]
    quad_spectrum = spectra[3, :, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        power_y_safe = np.where(power_y == 0, np.nan, power_y)
        coherence_sq = (np.square(co_spectrum) + np.square(quad_spectrum)) / (power_x * power_y_safe)
        phase = np.arctan2(quad_spectrum, co_spectrum)
        norm = np.sqrt(np.square(quad_spectrum) + np.square(co_spectrum))
        norm_safe = np.where(norm == 0, np.nan, norm)
        vector_x = -quad_spectrum / norm_safe
        vector_y = co_spectrum / norm_safe

    spectra[4, :, :] = coherence_sq
    spectra[5, :, :] = phase
    spectra[6, :, :] = vector_x
    spectra[7, :, :] = vector_y
    return spectra


def calculate_cross_spectrum(
    X: Union[xr.DataArray, np.ndarray],
    Y: Union[xr.DataArray, np.ndarray],
    *,
    segLen: int = 96,
    segOverLap: int = -65,
    symmetry: str = "symm",
    return_xarray: bool = True,
    normalize_by_reference: bool = False,
    latent_heat: float = 2.5e6,
) -> Dict[str, Union[np.ndarray, xr.DataArray, float]]:
    """Calculate a wavenumber-frequency cross-spectrum."""
    is_xarray = isinstance(X, xr.DataArray)
    if is_xarray:
        if not isinstance(Y, xr.DataArray):
            raise ValueError("X and Y must both be xarray.DataArray or numpy.ndarray.")
        lat = X.lat.values
        x_values = X.values
        y_values = Y.values
    else:
        if isinstance(Y, xr.DataArray):
            raise ValueError("X and Y must both be xarray.DataArray or numpy.ndarray.")
        x_values = np.asarray(X)
        y_values = np.asarray(Y)
        lat = np.linspace(-14, 14, x_values.shape[1])

    if x_values.shape != y_values.shape:
        raise ValueError(f"X and Y must have the same shape: {x_values.shape} vs {y_values.shape}")

    ntim, _, nlon = x_values.shape
    y_component = _get_symm_asymm(y_values, lat, symmetry)
    x_component = _get_symm_asymm(x_values, lat, symmetry)

    y_component = signal.detrend(y_component, 0)
    x_component = signal.detrend(x_component, 0)
    window = signal.windows.tukey(segLen, 0.10, True)

    nfreq = (segLen + 1) // 2 if segLen % 2 == 1 else segLen // 2 + 1
    nwave = nlon if nlon % 2 == 1 else nlon + 1
    spectra = np.zeros([8, nfreq, nwave], dtype="double")
    wave = np.arange(-int(nwave / 2), int(nwave / 2) + 1, 1.0)
    freq = (
        np.linspace(0, 0.5 * (segLen - 1) / segLen, num=nfreq)
        if segLen % 2 == 1
        else np.linspace(0, 0.5, num=nfreq)
    )
    zero_frequency = np.where(freq == 0.0)[0]

    n_segments = 0
    start = 0
    while start + segLen <= ntim:
        stop = start + segLen
        xx = x_component[start:stop, :, :] * window[:, np.newaxis, np.newaxis]
        yy = y_component[start:stop, :, :] * window[:, np.newaxis, np.newaxis]
        spectra_segment = _cross_spectrum_segment(xx, yy)
        spectra_segment[:, zero_frequency, :] = np.nan
        spectra_segment = _smooth121_frequency(spectra_segment, freq)
        spectra = spectra + spectra_segment
        n_segments += 1
        start = stop + segOverLap - 1

    if n_segments == 0:
        raise ValueError(f"Input length is too short for cross-spectrum: ntim={ntim}, segLen={segLen}")

    spectra = spectra / n_segments
    spectra = _compute_coherence_phase(spectra)

    if normalize_by_reference:
        extended = np.zeros([10, nfreq, nwave], dtype="double")
        extended[:8, :, :] = spectra
        power_x = spectra[0, :, :]
        co_spectrum = spectra[2, :, :]
        quad_spectrum = spectra[3, :, :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            power_x_safe = np.where(power_x == 0, np.nan, power_x)
            extended[8, :, :] = co_spectrum / (latent_heat * power_x_safe)
            extended[9, :, :] = quad_spectrum / (latent_heat * power_x_safe)

        spectra = extended

    dof = 2.667 * n_segments
    prob = np.array([0.80, 0.85, 0.90, 0.925, 0.95, 0.99])
    prob_coh2 = 1 - np.power(1 - prob, 1.0 / (0.5 * dof - 1))

    result: Dict[str, Union[np.ndarray, xr.DataArray, float]] = {
        "freq": freq,
        "wave": wave,
        "nseg": n_segments,
        "dof": dof,
        "p": prob,
        "prob_coh2": prob_coh2,
    }

    if return_xarray:
        if normalize_by_reference:
            component_names = [
                "PX",
                "PY",
                "CXY",
                "QXY",
                "COH2",
                "PHAS",
                "V1",
                "V2",
                "NORM_CXY_REAL",
                "NORM_CXY_IMAG",
            ]
        else:
            component_names = ["PX", "PY", "CXY", "QXY", "COH2", "PHAS", "V1", "V2"]
        result["STC"] = xr.DataArray(
            spectra,
            dims=("component", "frequency", "wavenumber"),
            coords={"component": component_names, "frequency": freq, "wavenumber": wave},
            attrs={"nseg": n_segments, "dof": dof, "normalized": normalize_by_reference},
        )
    else:
        result["STC"] = spectra
    return result


def quick_cross_spectrum(
    X: xr.DataArray,
    Y: xr.DataArray,
    *,
    remove_annual: bool = True,
    **kwargs: object,
) -> Dict[str, Union[np.ndarray, xr.DataArray, float]]:
    """Convenience wrapper that removes a day-of-year climatology before analysis."""
    if remove_annual:
        x_anomaly = X.groupby("time.dayofyear") - X.groupby("time.dayofyear").mean()
        y_anomaly = Y.groupby("time.dayofyear") - Y.groupby("time.dayofyear").mean()
    else:
        x_anomaly = X
        y_anomaly = Y
    return calculate_cross_spectrum(x_anomaly, y_anomaly, **kwargs)


__all__ = [
    "CrossSpectrumConfig",
    "calculate_cross_spectrum",
    "quick_cross_spectrum",
    "remove_annual_cycle",
    "nan_to_value_by_interp_3D",
]
