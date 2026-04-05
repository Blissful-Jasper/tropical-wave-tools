"""Wave-filter classes ported from the original project with cleaner boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy import fft, signal

from tropical_wave_tools.config import DEFAULT_WAVE_SPECS, WaveSpec
from tropical_wave_tools.exceptions import UnknownWaveError
from tropical_wave_tools.io import load_dataarray
from tropical_wave_tools.preprocessing import extract_low_harmonics


class WaveFilter:
    """Legacy-style WK filter for extracting equatorial wave signals."""

    beta: float = 2.28e-11
    earth_radius: float = 6.37e6

    def __init__(self) -> None:
        self.wave_specs: dict[str, WaveSpec] = dict(DEFAULT_WAVE_SPECS)

    def extract_low_harmonics(
        self,
        data: xr.DataArray,
        *,
        n_harm: int = 3,
        dim: str = "dayofyear",
    ) -> xr.DataArray:
        """Compatibility wrapper for climatology smoothing."""
        return extract_low_harmonics(data, n_harm=n_harm, dim=dim)

    def add_wave_param(
        self,
        wave_name: str,
        *,
        period_days: Tuple[Optional[float], Optional[float]],
        wavenumber: Tuple[int, int],
        equivalent_depth: Tuple[Optional[float], Optional[float]] = (None, None),
        meridional_mode: Optional[int] = None,
        dispersion_family: str = "none",
    ) -> None:
        """Register a new wave band."""
        self.wave_specs[wave_name.lower()] = WaveSpec(
            period_days=period_days,
            wavenumber=wavenumber,
            equivalent_depth=equivalent_depth,
            meridional_mode=meridional_mode,
            dispersion_family=dispersion_family,
        )

    def get_available_waves(self) -> List[str]:
        """Return the currently registered wave names."""
        return sorted(self.wave_specs)

    def get_wave_params(self, wave_name: str) -> WaveSpec:
        """Return one wave specification."""
        wave_key = wave_name.lower()
        if wave_key not in self.wave_specs:
            raise UnknownWaveError(
                f"Unknown wave '{wave_name}'. Available waves: {', '.join(self.get_available_waves())}"
            )
        return self.wave_specs[wave_key]

    @staticmethod
    def _internal_wave_name(wave_name: str) -> str:
        """Map public aliases to the internal dispersion-branch name."""
        wave_key = wave_name.lower()
        if wave_key in {"eig", "eig0", "ig"}:
            return "ig0"
        return wave_key

    def _kf_filter(
        self,
        in_data: Union[xr.DataArray, np.ndarray],
        *,
        lon: np.ndarray,
        obs_per_day: int,
        period_days: Tuple[Optional[float], Optional[float]],
        wavenumber: Tuple[int, int],
        equivalent_depth: Tuple[Optional[float], Optional[float]],
        wave_name: str,
        meridional_mode: Optional[int] = None,
    ) -> Union[xr.DataArray, np.ndarray]:
        """
        Apply the Wheeler-Kiladis wavenumber-frequency filter on ``(time, lon)`` data.

        This closely follows the original implementation to preserve result semantics.
        """
        is_xarray = isinstance(in_data, xr.DataArray)
        data_np = in_data.values if is_xarray else np.asarray(in_data)
        time_dim, lon_dim = data_np.shape

        wrap_flag = np.isclose((lon[0] + 360.0) % 360.0, lon[-1] % 360.0)
        if wrap_flag:
            data_np = data_np[:, 1:]

        data_np = signal.detrend(data_np, axis=0)
        data_np = signal.windows.tukey(time_dim, alpha=0.05)[:, np.newaxis] * data_np

        fft_data = fft.rfft2(data_np, axes=(1, 0))
        fft_data[:, 1:] = fft_data[:, -1:0:-1]

        period_min, period_max = period_days
        k_min, k_max = wavenumber
        h_min, h_max = equivalent_depth

        freq_dim = fft_data.shape[0]
        k_dim = fft_data.shape[1]
        j_min = int(time_dim / (period_max * obs_per_day)) if period_max else 0
        j_max = int(time_dim / (period_min * obs_per_day)) if period_min else freq_dim
        j_max = min(j_max, freq_dim)

        if k_min < 0:
            i_min = max(k_dim + k_min, k_dim // 2)
        else:
            i_min = min(k_min, k_dim // 2)

        if k_max < 0:
            i_max = max(k_dim + k_max, k_dim // 2)
        else:
            i_max = min(k_max, k_dim // 2)

        if j_min > 0:
            fft_data[:j_min, :] = 0
        if j_max < freq_dim - 1:
            fft_data[j_max + 1 :, :] = 0
        if i_min < i_max:
            if i_min > 0:
                fft_data[:, :i_min] = 0
            if i_max < k_dim - 1:
                fft_data[:, i_max + 1 :] = 0

        spc = 24.0 * 3600.0 / (2.0 * np.pi * obs_per_day)
        c = (
            np.sqrt(9.8 * np.array([h_min, h_max], dtype=float))
            if h_min is not None and h_max is not None
            else np.array([np.nan, np.nan])
        )

        for i in range(k_dim):
            k = (i - k_dim if i > k_dim // 2 else i) / self.earth_radius
            freq = np.array([0.0, float(freq_dim)])

            if wave_name == "kelvin" and np.all(np.isfinite(c)):
                freq = k * c
            elif wave_name == "er" and np.all(np.isfinite(c)):
                freq = -self.beta * k / (k**2 + 3 * self.beta / c)
            elif wave_name in {"mrg", "ig0", "eig", "eig0", "ig"} and np.all(np.isfinite(c)):
                if k == 0:
                    freq = np.sqrt(self.beta * c)
                elif k > 0:
                    freq = k * c * (0.5 + 0.5 * np.sqrt(1 + 4 * self.beta / (k**2 * c)))
                else:
                    freq = k * c * (0.5 - 0.5 * np.sqrt(1 + 4 * self.beta / (k**2 * c)))
            elif wave_name in {"ig1", "wig", "ig2"} and np.all(np.isfinite(c)):
                default_mode = 2 if wave_name == "ig2" else 1
                mode_number = int(meridional_mode or default_mode)
                freq = np.sqrt((2 * mode_number + 1) * self.beta * c + (k**2 * c**2))

            j_min_wave = int(np.floor(freq[0] * spc * time_dim)) if np.isfinite(freq[0]) else 0
            j_max_wave = int(np.ceil(freq[1] * spc * time_dim)) if np.isfinite(freq[1]) else freq_dim
            j_min_wave = min(j_min_wave, freq_dim)
            j_max_wave = max(j_max_wave, 0)

            fft_data[:j_min_wave, i] = 0
            if j_max_wave < freq_dim:
                fft_data[j_max_wave + 1 :, i] = 0

        fft_data[:, 1:] = fft_data[:, -1:0:-1]
        filtered = np.real(fft.irfft2(fft_data, axes=(1, 0), s=(lon_dim, time_dim)))

        if is_xarray:
            output = in_data.copy(data=filtered)
            output.attrs.update(
                {
                    "wavenumber": wavenumber,
                    "period": period_days,
                    "depth": equivalent_depth,
                    "waveName": wave_name,
                }
            )
            return output.transpose("time", "lon")

        return filtered

    def extract_wave_signal(
        self,
        data: xr.DataArray,
        *,
        wave_name: str = "kelvin",
        obs_per_day: int = 1,
        use_parallel: bool = True,
        n_jobs: int = -1,
        n_harm: int = 3,
    ) -> xr.DataArray:
        """Remove the annual cycle and extract one wave component."""
        spec = self.get_wave_params(wave_name)
        climatology = data.groupby("time.dayofyear").mean(dim="time")
        climatology_fit = self.extract_low_harmonics(climatology, n_harm=n_harm)
        anomaly = (data.groupby("time.dayofyear") - climatology_fit).transpose("time", "lat", "lon")

        lon = anomaly.lon.values

        def _filter_one_latitude(lat_index: int) -> np.ndarray:
            lat_slice = anomaly.isel(lat=lat_index)
            filtered = self._kf_filter(
                lat_slice.values if use_parallel else lat_slice,
                lon=lon,
                obs_per_day=obs_per_day,
                period_days=spec.period_days,
                wavenumber=spec.wavenumber,
                equivalent_depth=spec.equivalent_depth,
                wave_name=self._internal_wave_name(wave_name),
                meridional_mode=spec.meridional_mode,
            )
            return np.asarray(filtered)

        if use_parallel:
            filtered = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_filter_one_latitude)(index) for index in range(len(anomaly.lat))
            )
        else:
            filtered = [_filter_one_latitude(index) for index in range(len(anomaly.lat))]

        filtered_array = np.stack(filtered, axis=1)
        return xr.DataArray(
            filtered_array,
            coords=anomaly.coords,
            dims=anomaly.dims,
            attrs={
                "long_name": f"{wave_name.title()} Wave Component",
                "units": data.attrs.get("units", "unknown"),
                "wavenumber": spec.wavenumber,
                "period": spec.period_days,
                "depth": spec.equivalent_depth,
                "waveName": wave_name.lower(),
            },
        )


class CCKWFilter:
    """Refactored high-level wave filter that wraps the legacy band-pass core."""

    def __init__(
        self,
        *,
        ds: Union[str, Path, xr.DataArray, xr.Dataset],
        var: Optional[str] = None,
        sel_dict: Optional[Dict[str, slice]] = None,
        wave_name: Optional[str] = None,
        units: Optional[str] = None,
        spd: int = 1,
        n_workers: int = 4,
        verbose: bool = True,
        n_harm: int = 3,
    ) -> None:
        self.ds = ds
        self.var = var
        self.sel_dict = sel_dict or {}
        self.wave_name = wave_name.lower() if wave_name else None
        self.units = units
        self.spd = spd
        self.n_workers = n_workers
        self.verbose = verbose
        self.n_harm = n_harm

        self.data: Optional[xr.DataArray] = None
        self.anomaly: Optional[xr.DataArray] = None
        self.filtered_data: Optional[np.ndarray] = None
        self.filter_note: Optional[str] = None
        self.spec: Optional[WaveSpec] = None

    def __repr__(self) -> str:
        return (
            "CCKWFilter("
            f"wave_name={self.wave_name!r}, spd={self.spd}, "
            f"n_workers={self.n_workers}, n_harm={self.n_harm})"
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _resolve_wave_spec(self) -> WaveSpec:
        if self.wave_name is None:
            raise ValueError("`wave_name` must be provided.")
        if self.wave_name not in DEFAULT_WAVE_SPECS:
            raise UnknownWaveError(
                f"Unknown wave '{self.wave_name}'. Available waves: {', '.join(sorted(DEFAULT_WAVE_SPECS))}"
            )
        self.spec = DEFAULT_WAVE_SPECS[self.wave_name]
        return self.spec

    def load_data(self) -> xr.DataArray:
        """Load and subset the source data."""
        if isinstance(self.ds, xr.DataArray):
            data = self.ds
        elif isinstance(self.ds, xr.Dataset):
            variable_name = self.var or next(iter(self.ds.data_vars))
            data = self.ds[variable_name]
        else:
            data = load_dataarray(self.ds, variable=self.var)

        data = data.sortby("lat").transpose("time", "lat", "lon")
        if self.sel_dict:
            data = data.sel(**self.sel_dict)
        self.data = data
        return data

    def _nyquist_frequency(self) -> float:
        return 0.5 * float(self.spd)

    def _is_resolvable(self) -> bool:
        if self.spec is None:
            self._resolve_wave_spec()
        if self.spec is None:
            raise RuntimeError("Wave specification was not resolved.")

        period_min, period_max = self.spec.period_days
        fmin = None if period_max is None else 1.0 / period_max
        fmax = None if period_min is None else 1.0 / period_min
        nyquist = self._nyquist_frequency()

        if fmin is not None and fmin >= nyquist - 1.0e-12:
            return False
        if fmax is not None and fmax > nyquist + 1.0e-12:
            return False
        return True

    def detrend_data(self) -> xr.DataArray:
        """Build the anomaly field used by the filter."""
        if self.data is None:
            self.load_data()
        self._resolve_wave_spec()
        if self.data is None:
            raise RuntimeError("Input data failed to load.")

        legacy_filter = WaveFilter()
        climatology = self.data.groupby("time.dayofyear").mean(dim="time")
        climatology_fit = legacy_filter.extract_low_harmonics(climatology, n_harm=self.n_harm)
        self.anomaly = (self.data.groupby("time.dayofyear") - climatology_fit).transpose(
            "time",
            "lat",
            "lon",
        )

        self.filter_note = None
        if not self._is_resolvable():
            self.filter_note = (
                f"Wave {self.wave_name} is not resolvable for samples_per_day={self.spd}; "
                "returned zeros to avoid Nyquist-edge artefacts."
            )
        return self.anomaly

    def _filter_one_latitude(self, lat_index: int) -> np.ndarray:
        if self.anomaly is None or self.spec is None:
            raise RuntimeError("Call detrend_data() before filtering.")
        if not self._is_resolvable():
            return np.zeros((self.anomaly.sizes["time"], self.anomaly.sizes["lon"]), dtype=np.float64)

        legacy_filter = WaveFilter()
        wave_name = WaveFilter._internal_wave_name(str(self.wave_name))
        lat_slice = self.anomaly.isel(lat=lat_index)
        filtered = legacy_filter._kf_filter(
            lat_slice.values,
            lon=self.anomaly.lon.values,
            obs_per_day=self.spd,
            period_days=self.spec.period_days,
            wavenumber=self.spec.wavenumber,
            equivalent_depth=self.spec.equivalent_depth,
            wave_name=wave_name,
            meridional_mode=self.spec.meridional_mode,
        )
        return np.asarray(filtered)

    def apply_filter(self) -> np.ndarray:
        """Apply the filter across latitude."""
        if self.anomaly is None:
            self.detrend_data()
        if self.anomaly is None:
            raise RuntimeError("Anomaly field was not prepared.")

        if self.n_workers == 1:
            filtered = [self._filter_one_latitude(index) for index in range(len(self.anomaly.lat))]
        else:
            filtered = Parallel(n_jobs=self.n_workers, prefer="threads")(
                delayed(self._filter_one_latitude)(index)
                for index in range(len(self.anomaly.lat))
            )
        self.filtered_data = np.stack(filtered, axis=1)
        return self.filtered_data

    def create_output(self) -> xr.DataArray:
        """Convert the filtered array back to an xarray object."""
        if self.filtered_data is None:
            raise ValueError("No filtered data are available.")
        if self.data is None or self.spec is None:
            raise RuntimeError("Missing filter state.")

        attrs = {
            "long_name": f"{self.wave_name} wave filtered data",
            "min_equiv_depth": self.spec.equivalent_depth[0],
            "max_equiv_depth": self.spec.equivalent_depth[1],
            "min_wavenumber": self.spec.wavenumber[0],
            "max_wavenumber": self.spec.wavenumber[1],
            "min_period": self.spec.period_days[0],
            "max_period": self.spec.period_days[1],
            "units": self.units or self.data.attrs.get("units", "unknown"),
            "filter_method": "NCL-aligned Wheeler-Kiladis kf_filter",
            "samples_per_day": self.spd,
            "annual_cycle_harmonics": self.n_harm,
            "waveName": self.wave_name,
        }
        if self.filter_note is not None:
            attrs["note"] = self.filter_note

        return xr.DataArray(
            self.filtered_data,
            coords={"time": self.data.time, "lat": self.data.lat, "lon": self.data.lon},
            dims=("time", "lat", "lon"),
            attrs=attrs,
        )

    def process(self) -> xr.DataArray:
        """Run the complete filter pipeline."""
        self._log(f"Processing wave filter for: {self.wave_name}")
        self.load_data()
        self.detrend_data()
        self.apply_filter()
        output = self.create_output()
        self._log("Wave filtering finished.")
        return output


def filter_wave_signal(
    data: xr.DataArray,
    *,
    wave_name: str,
    method: str = "cckw",
    obs_per_day: int = 1,
    n_harm: int = 3,
    n_workers: int = 4,
    n_jobs: int = -1,
    use_parallel: bool = True,
) -> xr.DataArray:
    """Convenience function for one-off wave filtering."""
    if method == "legacy":
        legacy_filter = WaveFilter()
        return legacy_filter.extract_wave_signal(
            data,
            wave_name=wave_name,
            obs_per_day=obs_per_day,
            use_parallel=use_parallel,
            n_jobs=n_jobs,
            n_harm=n_harm,
        )
    if method == "cckw":
        cckw_filter = CCKWFilter(
            ds=data,
            wave_name=wave_name,
            units=data.attrs.get("units"),
            spd=obs_per_day,
            n_workers=n_workers,
            n_harm=n_harm,
            verbose=False,
        )
        return cckw_filter.process()
    raise ValueError("`method` must be either 'legacy' or 'cckw'.")
