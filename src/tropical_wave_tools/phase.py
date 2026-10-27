"""Phase-analysis helpers for Kelvin-wave composites."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt

LOGGER = logging.getLogger(__name__)


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
    _nt, nlat, nlon = data.shape
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
    """Remove the time mean and return the anomaly field.

    ``fs``, ``cutoff`` and ``order`` are accepted for backward compatibility but unused: no
    low-pass filter is applied here (see ``remove_10d_from_daily_data`` for that).
    """
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
        name="anomaly",
    )


def find_local_extrema(values: np.ndarray) -> np.ndarray:
    """Find local minima (``+1``) and maxima (``-1``) in ``(time, lon)`` data.

    A flat run of equal values counts as one extremum (its first sample), so a plateau no
    longer produces several adjacent "peaks" that cut each other's influence range, and a
    constant series has none.
    """
    nt, nlon = values.shape
    output = np.full((nt, nlon), np.nan, dtype=np.float32)

    previous_values = values[:-2, :]
    current_values = values[1:-1, :]
    next_values = values[2:, :]

    local_min = (current_values < previous_values) & (current_values <= next_values)
    local_max = (current_values > previous_values) & (current_values >= next_values)

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
) -> tuple[int, int]:
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
    peak_output = np.full(nt, np.nan, dtype=np.float64)  # float32 would cost ~1e-4 rad near crests
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
) -> tuple[xr.DataArray, xr.DataArray]:
    """Detect local extrema and assign significant-peak influence ranges."""
    nt, nlon = values.shape
    start_time = time.time()
    local_min_max_id = find_local_extrema(values)
    LOGGER.debug("Local-extrema detection time: %.2fs", time.time() - start_time)

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
        peak_values = np.full((nt, nlon), np.nan, dtype=np.float64)
        for lon_index in range(nlon):
            peak_values[:, lon_index] = process_single_longitude(
                lon_index,
                values,
                local_min_max_id,
                value_std,
                Nstd,
            )
    LOGGER.debug("Peak-influence assignment time: %.2fs", time.time() - assign_start)

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
    n_bins: int | None = None,
    phase_range: tuple[float, float] = (-np.pi, np.pi),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite data by phase bins.

    The ``n_bins`` bins have equal width ``(hi - lo) / n_bins`` and the returned centres are
    ``lo + (i + 0.5) * width``. For a full ``2*pi`` range the phase is periodic, so ``+pi`` and
    ``-pi`` are the same phase and fall in the same bin; values outside a partial range are
    ignored. (Earlier versions spanned ``2*pi + width`` with ``n_bins`` bins, which made the bins
    wider than ``2*pi / n_bins`` and split the ``+-pi`` phase over two half-populated end bins.)
    The default ``n_bins`` is 19.
    """
    from scipy import stats as scipy_stats

    low, high = phase_range
    if n_bins is None:
        n_bins = int((high - low) / (1 / np.pi))
    bins = np.linspace(low, high, n_bins + 1)
    phase_bin = 0.5 * (bins[:-1] + bins[1:])

    phase_flat = phase.values.flatten().astype(float)
    data_flat = data.values.flatten().astype(float)
    mask = ~np.isnan(phase_flat) & ~np.isnan(data_flat)
    phase_flat, data_flat = phase_flat[mask], data_flat[mask]
    if np.isclose(high - low, 2.0 * np.pi):
        phase_flat = low + np.mod(phase_flat - low, high - low)  # fold +pi onto -pi

    counts = scipy_stats.binned_statistic(phase_flat, data_flat, statistic="count", bins=bins).statistic
    means = scipy_stats.binned_statistic(phase_flat, data_flat, statistic="mean", bins=bins).statistic
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create lag composites relative to a reference longitude and phase threshold.

    ``lon_ref`` is snapped to the nearest grid longitude (an error is raised if it is farther
    than one grid step away). The reference times are those where the phase at that longitude
    is within ``tolerance`` of ``phase_threshold``; with the default ``-pi/2`` these are the
    crests of the filtered series. Times closer than ``nlag`` samples to either end are skipped.
    """
    nt, nlon = data.shape
    lon_values = np.asarray(lon, dtype=float)
    lon_index = int(np.argmin(np.abs(lon_values - lon_ref)))
    grid_step = float(np.median(np.abs(np.diff(lon_values)))) if lon_values.size > 1 else 0.0
    if abs(lon_values[lon_index] - lon_ref) > max(grid_step, 1.0e-6):
        raise ValueError(f"`lon_ref`={lon_ref} is outside the longitude grid ({lon_values.min()}..{lon_values.max()}).")
    phase_ref = phase.data[:, lon_index]
    candidates = np.atleast_1d(np.argwhere(np.abs(phase_ref - phase_threshold) < tolerance).squeeze())
    it_max = candidates[(candidates > nlag) & (candidates < nt - nlag)].astype(int)

    composite = np.full([2 * nlag + 1, nlon], np.nan)
    if it_max.size == 0:
        return np.arange(-nlag, nlag + 1, 1), composite, it_max

    composite[nlag, :] = np.nanmean(data.data[it_max, :], 0)
    for lag_index in range(1, nlag + 1):
        composite[nlag - lag_index, :] = np.nanmean(data.data[it_max - lag_index, :], 0)
        composite[nlag + lag_index, :] = np.nanmean(data.data[it_max + lag_index, :], 0)

    return np.arange(-nlag, nlag + 1, 1), composite, it_max


def trace_composite_ridge(
    composite: np.ndarray,
    lags: np.ndarray,
    lon: np.ndarray,
    *,
    lon_ref: float,
    threshold: float,
    window_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow the single strongest crest of a lag-longitude composite outward from ``lon_ref``.

    Starting from the maximum within ``window_deg`` of ``lon_ref`` at lag 0, the crest is followed
    to larger and smaller lags: at each step the maximum within ``window_deg`` of the position
    predicted from the previous step is taken, and tracing stops when it falls below ``threshold``.
    Longitudes are treated as circular and returned unwrapped (continuous). Other simultaneous
    crests, for example the repeats of a wave with zonal wavenumber 3, are ignored.

    Returns ``(ridge_lags, ridge_lons)``, both empty if lag 0 has no crest above ``threshold``.
    """
    lags = np.asarray(lags)
    lon = np.asarray(lon, dtype=float)
    centre_index = int(np.argmin(np.abs(lags)))

    def circular_offset(centre: float) -> np.ndarray:
        return (lon - centre + 180.0) % 360.0 - 180.0

    def peak_near(row: np.ndarray, centre: float) -> tuple[float, float] | None:
        masked = np.where(np.abs(circular_offset(centre)) <= window_deg, row, np.nan)
        if not np.isfinite(masked).any():
            return None
        index = int(np.nanargmax(masked))
        if masked[index] < threshold:
            return None
        return float(centre + circular_offset(centre)[index]), float(masked[index])

    start = peak_near(composite[centre_index], lon_ref)
    if start is None:
        return np.array([], dtype=lags.dtype), np.array([], dtype=float)

    track: dict[int, float] = {centre_index: start[0]}
    for direction in (+1, -1):
        previous, shift = start[0], 0.0
        index = centre_index + direction
        while 0 <= index < lags.size:
            found = peak_near(composite[index], previous + shift)
            if found is None:
                break
            shift = found[0] - previous  # longitude change per step, used as the next prediction
            previous = found[0]
            track[index] = found[0]
            index += direction
    indices = np.array(sorted(track))
    return lags[indices], np.array([track[i] for i in indices])


def estimate_phase_speed(
    composite: np.ndarray,
    lags: np.ndarray,
    lon: np.ndarray,
    *,
    lon_ref: float,
    threshold: float,
    method: str = "ridge",
    window_deg: float = 30.0,
    min_points: int = 3,
) -> float:
    """Zonal phase speed (m/s, positive eastward) of a lag-longitude composite, NaN if undefined.

    ``method="ridge"`` fits a straight line to the crest followed by :func:`trace_composite_ridge`
    and is reliable when the composite holds several simultaneous crests. ``method="threshold_fit"``
    is the historical estimator: one straight line through *all* points above ``threshold``, which
    is only meaningful for a single propagating ridge. Lags are in days (one step = one day).
    """
    lags = np.asarray(lags, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if method == "ridge":
        ridge_lags, ridge_lons = trace_composite_ridge(
            composite, lags, lon, lon_ref=lon_ref, threshold=threshold, window_deg=window_deg
        )
        if ridge_lags.size < min_points:
            return float("nan")
        slope = np.polyfit(ridge_lags.astype(float), ridge_lons, deg=1)[0]
    elif method == "threshold_fit":
        strong = np.argwhere(composite >= threshold)
        if strong.shape[0] < 2:
            return float("nan")
        slope = np.polyfit(lags[strong[:, 0]], lon[strong[:, 1]], deg=1)[0]
    else:
        raise ValueError("`method` must be 'ridge' or 'threshold_fit'.")
    return float(slope * 111000.0 / (24 * 3600.0))


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
    to_mm_per_day: float | None = None,
    strong_threshold: float = 1.5,
    speed_method: str = "ridge",
    ridge_window_deg: float = 30.0,
) -> None:
    """Run a Kelvin-wave phase-composite workflow and save its products.

    ``to_mm_per_day`` converts the filtered field to mm/day for the phase-speed fit; the default
    (``None``) keeps the historical behaviour of 86400 (kg m-2 s-1 input) except for
    ``model_name == "GPCP"``, which is assumed to be in mm/day already. Points of the lag-longitude
    composite above ``strong_threshold`` (mm/day) are used to estimate the phase speed ``Cp_ave``
    with :func:`estimate_phase_speed`. The default ``speed_method="ridge"`` follows the single
    strongest crest from ``lon_ref`` (``ridge_window_deg`` per step) and is robust to several
    simultaneous crests; ``"threshold_fit"`` is the historical straight-line fit through all points
    above the threshold, meaningful only for one propagating ridge.
    """
    import os

    lat = kelvin.lat.values
    kelvin_ref = meridional_projection(kelvin, lat)
    meridional_projection(pr_ori, lat)
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

    factor = to_mm_per_day if to_mm_per_day is not None else (1.0 if model_name == "GPCP" else 86400.0)
    pr_kw_plot = pr_kw_comp * factor

    phase_speed = estimate_phase_speed(
        pr_kw_plot,
        tlag,
        lon,
        lon_ref=lon_ref,
        threshold=strong_threshold,
        method=speed_method,
        window_deg=ridge_window_deg,
    )
    if not np.isfinite(phase_speed):
        return

    np.savez(
        os.path.join(output_dir, f"{model_name}_kw_zwnum_freq_Cp_ave_from_precip_lag_regression_history_{lon_ref}.npz"),
        Cp_ave=phase_speed,
        speed_method=speed_method,
    )

    if debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(phase_bin, pr_kw_phase_mean)
        plt.title(f"{model_name} KW phase composite")
        plt.show()


__all__ = [
    "butter_lowpass_filter",
    "calculate_kelvin_phase",
    "composite_kw_phase",
    "estimate_phase_speed",
    "find_local_extrema",
    "find_peak_influence_range",
    "lag_composite",
    "meridional_projection",
    "optimize_peak_detection",
    "phase_composite",
    "process_single_longitude",
    "remove_10d_from_daily_data",
    "remove_clm",
    "save_composite_to_netcdf",
    "trace_composite_ridge",
]
