"""Phase-analysis helpers for Kelvin-wave composites."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt
import time


def butter_lowpass_filter(data: np.ndarray, cutoff_freq: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth low-pass filter to a one-dimensional series."""
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_freq / nyquist
    b_values, a_values = butter(order, normalized_cutoff, btype="lowpass")
    return filtfilt(b_values, a_values, data)


def remove_10d_from_daily_data(
    data: np.ndarray,
    *,
    fs: float = 1.0,
    cutoff: float = 1 / 10,
    order: int = 4,
) -> np.ndarray:
    """Low-pass filter a ``(time, lat, lon)`` daily array to remove sub-10-day signals."""
    nt, nlat, nlon = data.shape
    filtered = np.empty_like(data)
    for lat_index in range(nlat):
        for lon_index in range(nlon):
            filtered[:, lat_index, lon_index] = butter_lowpass_filter(
                data[:, lat_index, lon_index],
                cutoff_freq=cutoff,
                fs=fs,
                order=order,
            )
    return filtered


def remove_clm(data: xr.DataArray, *, fs: float = 1.0, cutoff: float = 1 / 10, order: int = 4) -> xr.DataArray:
    """Remove the mean climatology and return the anomaly field."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("Input must be an xarray.DataArray.")
    if "time" not in data.dims:
        raise ValueError("Input data must contain the 'time' dimension.")

    climatology = data.mean("time")
    anomalies = data - climatology
    return xr.DataArray(
        data=anomalies.data,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs,
        name="lowpass_filtered_anomaly",
    )


def find_local_extrema(values: np.ndarray) -> np.ndarray:
    """Find local minima and maxima in ``(time, lon)`` data."""
    nt, nlon = values.shape
    output = np.full((nt, nlon), np.nan, dtype=np.float32)

    previous_values = values[:-2, :]
    current_values = values[1:-1, :]
    next_values = values[2:, :]

    local_min = (current_values <= previous_values) & (current_values <= next_values)
    local_max = (current_values >= previous_values) & (current_values >= next_values)

    output[1:-1, :] = np.where(local_min, 1, output[1:-1, :])
    output[1:-1, :] = np.where(local_max, -1, output[1:-1, :])
    return output


def find_peak_influence_range(
    peak_idx: int,
    peak_value: float,
    zero_idx: np.ndarray,
    peak_indices: np.ndarray,
    values: np.ndarray,
    value_std: float,
    n_std: float,
) -> Tuple[int, int]:
    """Return the time range influenced by a significant peak."""
    if np.abs(peak_value) < value_std * n_std:
        return peak_idx, peak_idx

    distance_to_zero = zero_idx - peak_idx
    positive_distance = np.where(distance_to_zero >= 0, distance_to_zero, np.inf)
    if np.all(np.isinf(positive_distance)):
        right_index = peak_idx
    else:
        right_index = zero_idx[np.argmin(positive_distance)]
        next_peaks = peak_indices[peak_indices > peak_idx]
        if next_peaks.size > 0:
            right_index = min(right_index, next_peaks.min() - 1)

    negative_distance = np.where(distance_to_zero < 0, distance_to_zero, -np.inf)
    if np.all(negative_distance == -np.inf):
        left_index = peak_idx
    else:
        left_index = zero_idx[np.argmax(negative_distance)] + 1
        previous_peaks = peak_indices[peak_indices < peak_idx]
        if previous_peaks.size > 0:
            left_index = max(left_index, previous_peaks.max() + 1)

    return int(left_index), int(right_index)


def process_single_longitude(
    lon_index: int,
    values: np.ndarray,
    local_min_max_id: np.ndarray,
    value_std: float,
    n_std: float,
) -> np.ndarray:
    """Assign the influence range of significant peaks for one longitude."""
    nt = values.shape[0]
    peak_output = np.full(nt, np.nan, dtype=np.float32)
    zero_idx = np.where(values[:-1, lon_index] * values[1:, lon_index] <= 0)[0]
    peak_idx = np.where(np.abs(local_min_max_id[:, lon_index]) == 1)[0]

    if len(peak_idx) == 0:
        return peak_output

    for idx in peak_idx:
        peak_value = values[idx, lon_index]
        if np.abs(peak_value) < value_std * n_std:
            continue
        left_index, right_index = find_peak_influence_range(
            peak_idx=idx,
            peak_value=peak_value,
            zero_idx=zero_idx,
            peak_indices=peak_idx,
            values=values[:, lon_index],
            value_std=value_std,
            n_std=n_std,
        )
        peak_output[left_index : right_index + 1] = peak_value
    return peak_output


def optimize_peak_detection(
    values: np.ndarray,
    kelvin_ref: xr.DataArray,
    value_std: float,
    *,
    Nstd: float = 1.0,
    use_parallel: bool = True,
    n_jobs: int = -1,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Detect local extrema and assign significant-peak influence ranges."""
    nt, nlon = values.shape
    start_time = time.time()
    local_min_max_id = find_local_extrema(values)
    print(f"[sqrt] Local-extrema detection time: {time.time() - start_time:.2f}s")

    extrema = xr.DataArray(
        data=local_min_max_id,
        dims=kelvin_ref.dims,
        coords=kelvin_ref.coords,
        name="local_extrema",
    )

    assign_start = time.time()
    if use_parallel:
        peak_columns = Parallel(n_jobs=n_jobs)(
            delayed(process_single_longitude)(lon_index, values, local_min_max_id, value_std, Nstd)
            for lon_index in range(nlon)
        )
        peak_values = np.column_stack(peak_columns)
    else:
        peak_values = np.full((nt, nlon), np.nan, dtype=np.float32)
        for lon_index in range(nlon):
            peak_values[:, lon_index] = process_single_longitude(
                lon_index,
                values,
                local_min_max_id,
                value_std,
                Nstd,
            )
    print(f"[sqrt] Peak-influence assignment time: {time.time() - assign_start:.2f}s")

    peak_da = xr.DataArray(
        data=peak_values,
        dims=kelvin_ref.dims,
        coords=kelvin_ref.coords,
        name="peak_influence",
    )
    return peak_da, extrema


def meridional_projection(
    inputdata: xr.DataArray,
    lat: np.ndarray,
    *,
    lat_0: float = 9.0,
    lat_tropics: float = 10.0,
    omega: int = 0,
) -> xr.DataArray:
    """Project a ``(time, lat, lon)`` field meridionally using a Gaussian weight."""
    weights = np.exp(-(lat / (2 * lat_0)) ** 2)
    if omega == 0:
        omega_mask = np.where(np.abs(lat) > lat_tropics, 0, 1)
    else:
        omega_mask = np.where(np.abs(lat) > lat_tropics, 1, 0)
    filter_weights = weights * omega_mask
    weight_sum = np.sum(filter_weights)

    if inputdata.ndim != 3 or inputdata.dims != ("time", "lat", "lon"):
        raise ValueError("inputdata must be a (time, lat, lon) DataArray.")

    transposed = np.transpose(inputdata.data, (0, 2, 1))
    if np.sum(np.isnan(transposed)) == 0:
        projected = np.inner(transposed, filter_weights) / weight_sum
    else:
        nt, nlon, _ = transposed.shape
        expanded_weights = np.tile(filter_weights, (nt, nlon, 1))
        masked_weights = np.ma.array(expanded_weights, mask=np.isnan(transposed))
        projected = np.nansum(transposed * masked_weights, axis=2) / np.nansum(masked_weights, axis=2)

    return xr.DataArray(
        data=projected,
        dims=("time", "lon"),
        coords={"time": inputdata.time, "lon": inputdata.lon},
        name="meridional_projection",
    )


def calculate_kelvin_phase(
    kelvin_filtered: xr.DataArray,
    peak_values: xr.DataArray,
    *,
    correct_phase: bool = True,
) -> xr.DataArray:
    """Calculate Kelvin-wave phase from filtered values and peak envelopes."""
    values = kelvin_filtered.data
    peak_data = peak_values.data
    nt, nlon = values.shape

    enhancement_decay = np.full((nt, nlon), np.nan)
    enhancement_decay[1:-1, :] = np.where(
        (values[1:-1, :] > values[0:-2, :])
        & (values[1:-1, :] < values[2:, :])
        & (~np.isnan(peak_data[1:-1, :])),
        0,
        np.nan,
    )
    enhancement_decay[1:-1, :] = np.where(
        (values[1:-1, :] < values[0:-2, :])
        & (values[1:-1, :] > values[2:, :])
        & (~np.isnan(peak_data[1:-1, :])),
        1,
        enhancement_decay[1:-1, :],
    )

    peak_amplitude = np.abs(peak_data)
    normalized = np.divide(
        values,
        peak_amplitude,
        out=np.full(values.shape, np.nan, dtype=np.float64),
        where=peak_amplitude > 0,
    )
    normalized = np.clip(normalized, -1.0, 1.0)
    phase = np.arcsin(normalized)
    phase = np.where(np.isfinite(peak_data), phase, np.nan)

    if correct_phase:
        phase_corrected = phase.copy()
        decay_negative = (enhancement_decay == 1) & (peak_data <= 0)
        decay_positive = (enhancement_decay == 1) & (peak_data >= 0)
        phase_corrected[decay_negative] = -np.pi - phase[decay_negative]
        phase_corrected[decay_positive] = np.pi - phase[decay_positive]
        phase = -phase_corrected

    return xr.DataArray(data=phase, dims=kelvin_filtered.dims, coords=kelvin_filtered.coords, name="kelvin_phase")


def phase_composite(
    data: xr.DataArray,
    phase: xr.DataArray,
    *,
    n_bins: Optional[int] = None,
    phase_range: Tuple[float, float] = (-np.pi, np.pi),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite data by phase bins."""
    from scipy import stats as scipy_stats

    if n_bins is None:
        delta_phase = 1 / np.pi
        n_bins = int((phase_range[1] - phase_range[0]) / delta_phase)
    else:
        delta_phase = (phase_range[1] - phase_range[0]) / n_bins

    bins = np.linspace(
        phase_range[0] - delta_phase / 2,
        phase_range[1] + delta_phase / 2,
        n_bins + 1,
    )
    phase_bin = 0.5 * (bins[:-1] + bins[1:])
    phase_flat = phase.values.flatten()
    data_flat = data.values.flatten()
    mask = ~np.isnan(phase_flat) & ~np.isnan(data_flat)

    counts = scipy_stats.binned_statistic(phase_flat[mask], data_flat[mask], statistic="count", bins=bins).statistic
    means = scipy_stats.binned_statistic(phase_flat[mask], data_flat[mask], statistic="mean", bins=bins).statistic
    return phase_bin, means, counts


def lag_composite(
    data: xr.DataArray,
    phase: xr.DataArray,
    lon: np.ndarray,
    *,
    lon_ref: float = 180.0,
    nlag: int = 10,
    phase_threshold: float = -np.pi / 2,
    tolerance: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create lag composites relative to a reference longitude and phase threshold."""
    nt, nlon = data.shape
    lon_index = np.argwhere(lon == lon_ref).squeeze()
    phase_ref = phase.data[:, lon_index]
    it_max = np.argwhere(np.abs(phase_ref - phase_threshold) < tolerance).squeeze()
    it_max = np.where(((it_max <= nlag) | (it_max >= nt - nlag)), np.nan, it_max)
    it_max = np.delete(it_max, np.isnan(it_max) == 1).astype("int")

    composite = np.full([2 * nlag + 1, nlon], np.nan)
    if it_max.size == 0:
        return np.arange(-nlag, nlag + 1, 1), composite, it_max

    composite[nlag, :] = np.nanmean(data.data[it_max, :], 0)
    for lag_index in range(1, nlag + 1):
        composite[nlag - lag_index, :] = np.nanmean(data.data[it_max - lag_index, :], 0)
        composite[nlag + lag_index, :] = np.nanmean(data.data[it_max + lag_index, :], 0)

    return np.arange(-nlag, nlag + 1, 1), composite, it_max


def save_composite_to_netcdf(
    output_path: str,
    pr_ano_comp: np.ndarray,
    pr_kw_comp: np.ndarray,
    lon: np.ndarray,
    nlag: int,
) -> Path:
    """Save lag composites to a NetCDF file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tlag = np.arange(-nlag, nlag + 1)
    lont, tlon = np.meshgrid(lon, tlag)

    dataset = xr.Dataset(
        {
            "pr_ano_comp": (("tlag", "lon"), pr_ano_comp),
            "pr_kw_comp": (("tlag", "lon"), pr_kw_comp),
            "lont": (("tlag", "lon"), lont),
            "tlon": (("tlag", "lon"), tlon),
        },
        coords={"tlag": tlag, "lon": lon},
    )
    dataset["tlag"].attrs.update({"standard_name": "lag_time", "units": "lag (time steps from maximum)"})
    dataset["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    dataset["pr_ano_comp"].attrs.update({"units": "mm/day", "long_name": "Precipitation anomaly composite"})
    dataset["pr_kw_comp"].attrs.update({"units": "mm/day", "long_name": "Kelvin-wave filtered precipitation composite"})
    dataset.to_netcdf(output)
    return output


def composite_kw_phase(
    kelvin: xr.DataArray,
    pr_ori: xr.DataArray,
    lon: np.ndarray,
    model_name: str,
    output_dir: str,
    *,
    lon_ref: float = 180.0,
    nlag: int = 10,
    Nstd: float = 1.0,
    debug_plot: bool = False,
) -> None:
    """Run a Kelvin-wave phase-composite workflow and save its products."""
    import os

    lat = kelvin.lat.values
    kelvin_ref = meridional_projection(kelvin, lat)
    pr_ori_eq = meridional_projection(pr_ori, lat)
    pr_anomaly = remove_clm(pr_ori)
    pr_anomaly_eq = meridional_projection(pr_anomaly, lat)

    values = kelvin_ref.data
    value_std = np.nanstd(values)
    peak_values, _ = optimize_peak_detection(values, kelvin_ref, value_std, Nstd=Nstd)
    phase = calculate_kelvin_phase(kelvin_ref, peak_values)
    phase_bin, pr_kw_phase_mean, _ = phase_composite(kelvin_ref, phase)

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f"{model_name}_precip_kw_{lon_ref}.npz"),
        KW_filtered_pr=kelvin_ref,
        lon=lon,
        phase_bin=phase_bin,
        phase_correct=phase,
        pr_kw=pr_kw_phase_mean,
    )

    tlag, pr_kw_comp, it_max = lag_composite(kelvin_ref, phase, lon, lon_ref=lon_ref, nlag=nlag)
    pr_ano_comp = np.full_like(pr_kw_comp, np.nan)
    if it_max.size > 0:
        pr_ano_comp[nlag, :] = np.nanmean(pr_anomaly_eq.data[it_max, :], axis=0)
        for lag_index in range(1, nlag + 1):
            pr_ano_comp[nlag - lag_index, :] = np.nanmean(pr_anomaly_eq.data[it_max - lag_index, :], axis=0)
            pr_ano_comp[nlag + lag_index, :] = np.nanmean(pr_anomaly_eq.data[it_max + lag_index, :], axis=0)

    output_file = os.path.join(output_dir, f"{model_name}_kw_composite_lag_lon_prano_prkw_history_{lon_ref}.nc")
    save_composite_to_netcdf(output_file, pr_ano_comp, pr_kw_comp, lon, nlag)

    if model_name != "GPCP":
        pr_kw_plot = pr_kw_comp * 86400
    else:
        pr_kw_plot = pr_kw_comp

    strong_points = np.where(pr_kw_plot >= 1.5, pr_kw_plot, np.nan)
    strong_index = np.argwhere(~np.isnan(strong_points))
    if strong_index.size == 0:
        return

    x_values = tlag[strong_index[:, 0]]
    y_values = lon[strong_index[:, 1]]
    slope, _ = np.polyfit(x_values, y_values, deg=1)
    phase_speed = slope * 111000.0 / (24 * 3600.0)

    np.savez(
        os.path.join(output_dir, f"{model_name}_kw_zwnum_freq_Cp_ave_from_precip_lag_regression_history_{lon_ref}.npz"),
        Cp_ave=phase_speed,
    )

    if debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(phase_bin, pr_kw_phase_mean)
        plt.title(f"{model_name} KW phase composite")
        plt.show()


__all__ = [
    "butter_lowpass_filter",
    "remove_10d_from_daily_data",
    "remove_clm",
    "find_local_extrema",
    "find_peak_influence_range",
    "process_single_longitude",
    "optimize_peak_detection",
    "meridional_projection",
    "calculate_kelvin_phase",
    "phase_composite",
    "lag_composite",
    "save_composite_to_netcdf",
    "composite_kw_phase",
]
