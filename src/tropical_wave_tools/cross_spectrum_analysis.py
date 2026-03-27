"""High-level workflows for multi-experiment cross-spectrum analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import gc

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tropical_wave_tools.cross_spectrum import calculate_cross_spectrum, remove_annual_cycle
from tropical_wave_tools.io import load_dataarray
from tropical_wave_tools.plotting import get_cckw_envelope_curve

try:  # pragma: no cover - optional dependency
    import dask
except ImportError:  # pragma: no cover - optional dependency
    dask = None

try:  # pragma: no cover - optional dependency
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


class MemoryMonitor:
    """Memory monitor with graceful fallback when psutil is unavailable."""

    def __init__(self) -> None:
        self.process = psutil.Process() if psutil is not None else None

    def get_memory_info(self) -> Dict[str, float]:
        """Return process and system memory information."""
        if self.process is None or psutil is None:
            return {
                "rss_gb": float("nan"),
                "vms_gb": float("nan"),
                "percent": float("nan"),
                "available_gb": float("nan"),
                "total_gb": float("nan"),
            }
        mem = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        virtual_mem = psutil.virtual_memory()
        return {
            "rss_gb": mem.rss / 1024**3,
            "vms_gb": mem.vms / 1024**3,
            "percent": mem_percent,
            "available_gb": virtual_mem.available / 1024**3,
            "total_gb": virtual_mem.total / 1024**3,
        }

    def print_memory_status(self, label: str = "") -> Dict[str, float]:
        """Print and return memory information."""
        info = self.get_memory_info()
        print(f"Memory status {('- ' + label) if label else ''}: {info}")
        return info


def load_netcdf_data(
    file_path: Union[str, Path],
    *,
    variable: Optional[str] = None,
    chunks: Optional[Dict[str, int]] = None,
    verbose: bool = True,
) -> xr.DataArray:
    """Load one NetCDF file using the project-standard I/O path."""
    if verbose:
        print(f"Loading: {file_path}")
    return load_dataarray(file_path, variable=variable, chunks=chunks)


def load_multiple_experiments(
    variable_name: str,
    experiments: Sequence[str],
    data_dir: Union[str, Path],
    *,
    file_pattern: str = "{var}_{exp}_2deg_interp.nc",
    chunks: Optional[Dict[str, int]] = None,
    scale_factor: float = 1.0,
    verbose: bool = True,
) -> Dict[str, xr.DataArray]:
    """Load one variable across multiple experiments."""
    data_dict: Dict[str, xr.DataArray] = {}
    for experiment in experiments:
        file_path = Path(data_dir) / file_pattern.format(var=variable_name, exp=experiment)
        try:
            data = load_netcdf_data(file_path, variable=variable_name, chunks=chunks, verbose=verbose)
            if scale_factor != 1.0:
                data = data * scale_factor
            data_dict[str(experiment)] = data
        except Exception as exc:
            if verbose:
                print(f"Failed to load {experiment}: {exc}")
    return data_dict


def preprocess_data_with_mask(
    data1: xr.DataArray,
    data2: xr.DataArray,
    *,
    mask: Optional[xr.DataArray] = None,
    remove_annual: bool = True,
    fill_value: float = 0.0,
    verbose: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Apply masking, NaN cleanup, and optional annual-cycle removal."""
    if remove_annual:
        data1 = data1.groupby("time.dayofyear") - data1.groupby("time.dayofyear").mean()
        data2 = data2.groupby("time.dayofyear") - data2.groupby("time.dayofyear").mean()

    if mask is not None:
        mask_float = mask.astype(float)
        data1 = data1 * mask_float
        data2 = data2 * mask_float

    data1 = xr.where(np.isinf(data1), np.nan, data1).fillna(fill_value)
    data2 = xr.where(np.isinf(data2), np.nan, data2).fillna(fill_value)

    if remove_annual:
        data1 = remove_annual_cycle(data1)
        data2 = remove_annual_cycle(data2)

    if verbose:
        print(f"Preprocessed data shapes: {data1.shape}, {data2.shape}")
    return data1, data2


def _compute_many(*arrays: xr.DataArray) -> Tuple[xr.DataArray, ...]:
    """Compute lazy arrays if dask is available."""
    if dask is not None:
        return dask.compute(*arrays)
    return tuple(array.load() for array in arrays)


def compute_cross_spectrum_for_experiments(
    data1_dict: Dict[str, xr.DataArray],
    data2_dict: Dict[str, xr.DataArray],
    *,
    experiments: Sequence[str],
    mask: Optional[xr.DataArray] = None,
    seg_length: int = 96,
    seg_overlap: int = -65,
    symmetry: str = "symm",
    memory_monitor: Optional[MemoryMonitor] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Compute cross-spectra for multiple experiments."""
    results: Dict[str, Dict[str, object]] = {}
    for experiment in experiments:
        if experiment not in data1_dict or experiment not in data2_dict:
            continue
        if memory_monitor is not None:
            memory_monitor.print_memory_status(f"start {experiment}")

        data1_raw = data1_dict[experiment]
        data2_raw = data2_dict[experiment]
        if mask is not None:
            data1_raw = data1_raw.where(mask, drop=True)
            data2_raw = data2_raw.where(mask, drop=True)

        data1_prepped, data2_prepped = preprocess_data_with_mask(data1_raw, data2_raw, mask=None, verbose=verbose)
        data1_computed, data2_computed = _compute_many(data1_prepped, data2_prepped)
        result = calculate_cross_spectrum(
            data1_computed,
            data2_computed,
            segLen=seg_length,
            segOverLap=seg_overlap,
            symmetry=symmetry,
            return_xarray=True,
        )
        results[str(experiment)] = {
            "STC": result["STC"],
            "freq": result["freq"],
            "wave": result["wave"],
            "nseg": result["nseg"],
            "dof": result["dof"],
            "p": result["p"],
            "prob_coh2": result["prob_coh2"],
        }
        del data1_prepped, data2_prepped, data1_computed, data2_computed
        gc.collect()
    return results


def plot_cross_spectrum_panel(
    results: Dict[str, Dict[str, object]],
    *,
    experiments: Sequence[str],
    exp_titles: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (16, 8),
    dpi: int = 300,
    cmap: str = "viridis",
    contour_levels: Optional[np.ndarray] = None,
    vector_scale: float = 30.0,
    vector_skip: int = 2,
    xlim: Tuple[float, float] = (-15, 15),
    ylim: Tuple[float, float] = (0.0, 0.5),
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot coherence-squared panels for multiple experiments."""
    if exp_titles is None:
        exp_titles = [str(experiment).upper() for experiment in experiments]
    if contour_levels is None:
        contour_levels = np.linspace(0.2, 0.8, 21)

    figure, axes = plt.subplots(1, len(experiments), figsize=figsize, dpi=dpi)
    if len(experiments) == 1:
        axes = np.array([axes])

    contourf = None
    for index, (experiment, title, axis) in enumerate(zip(experiments, exp_titles, axes)):
        if experiment not in results:
            continue
        stc = results[experiment]["STC"]
        wave = results[experiment]["wave"]
        freq = results[experiment]["freq"]
        threshold = float(np.asarray(results[experiment]["prob_coh2"]).max())
        coh2 = stc.sel(component="COH2")
        masked = coh2.where(coh2 >= threshold)
        contourf = masked.plot.contourf(
            ax=axis,
            cmap=cmap,
            levels=contour_levels,
            add_colorbar=False,
            add_labels=False,
            extend="neither",
        )

        wave_sub = wave[::vector_skip]
        freq_sub = freq[::vector_skip]
        u_sub = stc.sel(component="V1").values[::vector_skip, ::vector_skip]
        v_sub = stc.sel(component="V2").values[::vector_skip, ::vector_skip]
        coh2_sub = stc.sel(component="COH2").values[::vector_skip, ::vector_skip]
        mask = coh2_sub < threshold
        axis.quiver(
            wave_sub,
            freq_sub,
            np.where(mask, np.nan, u_sub),
            np.where(mask, np.nan, v_sub),
            scale=vector_scale,
            headwidth=4,
            headlength=5,
            width=0.004,
            alpha=0.8,
        )

        kw_x, kw_y = get_cckw_envelope_curve()
        axis.plot(kw_x[0], kw_y[0], "red", linewidth=1.5)
        axis.set_title(f"({chr(97 + index)}) {title}", loc="left")
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_xlabel("Zonal wavenumber")
        axis.set_ylabel("Frequency (1/day)")
        axis.axvline(0, linestyle=":", color="k", linewidth=1)
        if verbose:
            print(f"{experiment}: threshold={threshold:.4f}")

    if contourf is not None:
        colorbar = figure.colorbar(contourf, ax=axes, orientation="horizontal", pad=0.15, aspect=40, shrink=0.8)
        colorbar.set_label("Coherence Squared")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    return figure, axes


def analyze_cross_spectrum(
    var1_name: str,
    var2_name: str,
    experiments: Sequence[str],
    data_dir: Union[str, Path],
    *,
    mask: Optional[xr.DataArray] = None,
    file_pattern: str = "{var}_{exp}_2deg_interp.nc",
    var1_scale: float = 1.0,
    var2_scale: float = 1.0,
    chunks: Optional[Dict[str, int]] = None,
    seg_length: int = 96,
    seg_overlap: int = -65,
    symmetry: str = "symm",
    output_dir: Optional[Union[str, Path]] = None,
    plot_params: Optional[Dict[str, object]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, object]], Optional[Tuple[plt.Figure, np.ndarray]]]:
    """One-stop cross-spectrum workflow from data loading to plotting."""
    memory_monitor = MemoryMonitor()
    if verbose:
        memory_monitor.print_memory_status("start")

    var1_data = load_multiple_experiments(
        var1_name,
        experiments,
        data_dir,
        file_pattern=file_pattern,
        chunks=chunks,
        scale_factor=var1_scale,
        verbose=verbose,
    )
    var2_data = load_multiple_experiments(
        var2_name,
        experiments,
        data_dir,
        file_pattern=file_pattern,
        chunks=chunks,
        scale_factor=var2_scale,
        verbose=verbose,
    )

    results = compute_cross_spectrum_for_experiments(
        var1_data,
        var2_data,
        experiments=experiments,
        mask=mask,
        seg_length=seg_length,
        seg_overlap=seg_overlap,
        symmetry=symmetry,
        memory_monitor=memory_monitor,
        verbose=verbose,
    )

    figure_axes = None
    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        plot_kwargs = plot_params or {}
        output_path = output_root / f"cross_spectrum_{var1_name}_{var2_name}_{symmetry}.png"
        figure_axes = plot_cross_spectrum_panel(
            results,
            experiments=experiments,
            output_path=output_path,
            verbose=verbose,
            **plot_kwargs,
        )

    if verbose:
        memory_monitor.print_memory_status("done")
    return results, figure_axes


__all__ = [
    "MemoryMonitor",
    "load_netcdf_data",
    "load_multiple_experiments",
    "preprocess_data_with_mask",
    "compute_cross_spectrum_for_experiments",
    "plot_cross_spectrum_panel",
    "analyze_cross_spectrum",
]
