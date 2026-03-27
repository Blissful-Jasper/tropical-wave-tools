"""Wheeler-Kiladis spectral analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy import fft, signal

from tropical_wave_tools.config import SpectralConfig
from tropical_wave_tools.io import load_dataarray, save_dataset
from tropical_wave_tools.preprocessing import (
    build_wk_decomposition_layout,
    detrend_with_mean,
    remove_annual_cycle_fft,
    smooth_121,
)


@dataclass
class WKAnalysisResult:
    """Container for WK spectral analysis output."""

    power_symmetric: xr.DataArray
    power_antisymmetric: xr.DataArray
    background: xr.DataArray

    @property
    def frequency(self) -> np.ndarray:
        """Return the frequency coordinate."""
        return self.power_symmetric["frequency"].values

    @property
    def wavenumber(self) -> np.ndarray:
        """Return the wavenumber coordinate."""
        return self.power_symmetric["wavenumber"].values

    def to_dataset(self) -> xr.Dataset:
        """Convert the result to a serializable dataset."""
        return xr.Dataset(
            {
                "power_symmetric": self.power_symmetric,
                "power_antisymmetric": self.power_antisymmetric,
                "background": self.background,
            }
        )


class WKSpectralAnalysis:
    """Stateful WK analysis pipeline mirroring the original project."""

    def __init__(self, config: Optional[SpectralConfig] = None):
        self.config = config or SpectralConfig()
        self.raw_data: Optional[xr.DataArray] = None
        self.processed_data: Optional[xr.DataArray] = None
        self.result: Optional[WKAnalysisResult] = None

    def load_data(
        self,
        *,
        data: Optional[xr.DataArray] = None,
        data_path: Optional[Union[str, Path]] = None,
        variable: str = "olr",
        lat_range: Tuple[float, float] = (-15.0, 15.0),
        time_range: Optional[Tuple[str, str]] = None,
    ) -> WKSpectralAnalysis:
        """Load data from a DataArray or a NetCDF file."""
        if data is None and data_path is None:
            raise ValueError("Either `data` or `data_path` must be provided.")
        if data is not None:
            loaded = data.sortby("lat").transpose("time", "lat", "lon")
            if time_range is not None:
                loaded = loaded.sel(time=slice(*time_range))
            loaded = loaded.sel(lat=slice(*lat_range))
        else:
            loaded = load_dataarray(
                data_path,
                variable=variable,
                lat_range=lat_range,
                time_range=time_range,
            )
        self.raw_data = loaded
        return self

    def preprocess(self) -> WKSpectralAnalysis:
        """Detrend, remove low frequencies, and build the legacy WK layout."""
        if self.raw_data is None:
            raise ValueError("No input data loaded.")
        detrended = detrend_with_mean(self.raw_data)
        filtered = remove_annual_cycle_fft(
            detrended,
            samples_per_day=self.config.samples_per_day,
            freq_cutoff=self.config.resolved_freq_cutoff,
        )
        self.processed_data = build_wk_decomposition_layout(filtered)
        return self

    def compute_spectrum(self) -> WKSpectralAnalysis:
        """Compute the windowed WK power spectrum."""
        if self.processed_data is None:
            raise ValueError("Call preprocess() before compute_spectrum().")

        ntim, nlat, nlon = self.processed_data.shape
        spd = self.config.samples_per_day
        n_day_win = self.config.window_size_days
        n_day_skip = self.config.window_skip_days
        n_samp_win = int(n_day_win * spd)
        n_samp_skip = int(n_day_skip * spd)

        if ntim < n_samp_win:
            raise ValueError(
                f"Input time length ({ntim}) is shorter than the analysis window ({n_samp_win})."
            )

        n_window = int((ntim - n_samp_win) / (n_samp_skip + n_samp_win)) + 1
        if n_window <= 0:
            raise ValueError("No valid analysis windows were found.")

        sumpower = np.zeros((n_samp_win, nlat, nlon), dtype=np.float64)
        start = 0
        stop = n_samp_win

        for _ in range(n_window):
            data_window = self.processed_data[start:stop, :, :]
            window_values = signal.detrend(data_window.values, axis=0)
            taper = signal.windows.tukey(n_samp_win, self.config.tukey_alpha, sym=True)
            window_values *= taper[:, np.newaxis, np.newaxis]
            power = fft.fft2(window_values, axes=(0, 2)) / (nlon * n_samp_win)
            sumpower += np.abs(power) ** 2
            start = stop + n_samp_skip
            stop = start + n_samp_win

        sumpower /= float(n_window)

        if nlon % 2 == 0:
            wavenumber = fft.fftshift(fft.fftfreq(nlon) * nlon)[1:]
            sumpower = fft.fftshift(sumpower, axes=2)[:, :, nlon:0:-1]
        else:
            wavenumber = fft.fftshift(fft.fftfreq(nlon) * nlon)
            sumpower = fft.fftshift(sumpower, axes=2)[:, :, ::-1]

        frequency = fft.fftshift(fft.fftfreq(n_samp_win, d=1.0 / spd))[n_samp_win // 2 :]
        sumpower = fft.fftshift(sumpower, axes=0)[n_samp_win // 2 :, :, :]

        power_symmetric = np.array(2.0 * sumpower[:, nlat // 2 :, :].sum(axis=1), copy=True)
        power_antisymmetric = np.array(2.0 * sumpower[:, : nlat // 2, :].sum(axis=1), copy=True)
        background = np.array(sumpower.sum(axis=1), copy=True)

        power_symmetric[0, :] = np.nan
        power_antisymmetric[0, :] = np.nan
        background[0, :] = np.nan

        coords = {"frequency": frequency, "wavenumber": wavenumber}
        self.result = WKAnalysisResult(
            power_symmetric=xr.DataArray(power_symmetric, dims=("frequency", "wavenumber"), coords=coords),
            power_antisymmetric=xr.DataArray(
                power_antisymmetric,
                dims=("frequency", "wavenumber"),
                coords=coords,
            ),
            background=xr.DataArray(background, dims=("frequency", "wavenumber"), coords=coords),
        )
        return self

    def smooth_background(self, *, wave_limit: Optional[int] = None) -> WKSpectralAnalysis:
        """Smooth the background spectrum using the original 1-2-1 logic."""
        if self.result is None:
            raise ValueError("Call compute_spectrum() before smooth_background().")

        background = self.result.background.values.copy()
        wave_cap = self.config.wave_limit if wave_limit is None else wave_limit
        wave_indices = np.where(np.abs(self.result.wavenumber) <= wave_cap)[0]

        for idx, freq in enumerate(self.result.frequency):
            if freq < 0.1:
                n_smooth = 5
            elif freq < 0.2:
                n_smooth = 10
            elif freq < 0.3:
                n_smooth = 20
            else:
                n_smooth = 40

            for _ in range(n_smooth):
                background[idx, wave_indices] = smooth_121(background[idx, wave_indices])

        for wavenumber_index in wave_indices:
            for _ in range(10):
                background[:, wavenumber_index] = smooth_121(background[:, wavenumber_index])

        self.result = WKAnalysisResult(
            power_symmetric=self.result.power_symmetric,
            power_antisymmetric=self.result.power_antisymmetric,
            background=xr.DataArray(
                background,
                dims=self.result.background.dims,
                coords=self.result.background.coords,
            ),
        )
        return self

    def save(self, output_path: Union[str, Path]) -> Path:
        """Serialize the current result to NetCDF."""
        if self.result is None:
            raise ValueError("No result is available to save.")
        return save_dataset(self.result.to_dataset(), output_path)


def analyze_wk_spectrum(
    data: xr.DataArray,
    *,
    config: Optional[SpectralConfig] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> WKAnalysisResult:
    """High-level WK analysis returning a structured result object."""
    analysis = WKSpectralAnalysis(config)
    analysis.load_data(data=data)
    analysis.preprocess()
    analysis.compute_spectrum()
    analysis.smooth_background()
    if output_path is not None:
        analysis.save(output_path)
    if analysis.result is None:
        raise RuntimeError("Unexpected empty WK analysis result.")
    return analysis.result


def calculate_wk_spectrum(
    data: xr.DataArray,
    *,
    window_days: int = 96,
    skip_days: int = 30,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compatibility helper matching the original package return signature.

    It returns ``(power_symmetric, power_antisymmetric, background)``.
    """
    config = SpectralConfig(window_size_days=window_days, window_skip_days=skip_days)
    result = analyze_wk_spectrum(data, config=config, output_path=output_path)
    return result.power_symmetric, result.power_antisymmetric, result.background
