"""Plotting utilities for spectral and filter outputs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tropical_wave_tools.matsuno import matsuno_modes_wk
from tropical_wave_tools.spectral import WKAnalysisResult

try:
    import cartopy.crs as ccrs
except ImportError:  # pragma: no cover - optional dependency
    ccrs = None


def get_cckw_envelope_curve(
    *,
    he: Optional[Sequence[float]] = None,
    fmax: Optional[Sequence[float]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return polygon coordinates for the CCKW envelope."""
    he = list(he) if he is not None else [8.0, 25.0, 90.0]
    fmax = list(fmax) if fmax is not None else [1 / 3, 1 / 2.25, 0.5]

    gravity = 9.8
    earth_radius = 6371e3
    seconds_per_day = 86400.0
    wave_number_max = 14.0
    period_max_days = 20.0

    c_min = (gravity * he[0]) ** 0.5 / (2.0 * np.pi * earth_radius) * seconds_per_day
    c_max = (gravity * he[-1]) ** 0.5 / (2.0 * np.pi * earth_radius) * seconds_per_day

    x_curve = np.array(
        [
            2.0,
            1.0 / period_max_days / c_min,
            wave_number_max,
            wave_number_max,
            fmax[0] / c_max,
            2.0,
            2.0,
        ]
    )
    y_curve = np.array(
        [
            1.0 / period_max_days,
            1.0 / period_max_days,
            wave_number_max * c_min,
            fmax[0],
            fmax[0],
            2.0 * c_max,
            1.0 / period_max_days,
        ]
    )
    return [x_curve], [y_curve]


def save_figure(figure: plt.Figure, path: Union[str, Path], *, dpi: int = 200) -> Path:
    """Save a figure and return its output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def plot_time_series(
    data: xr.DataArray,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a simple time series."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 4), dpi=150)
    else:
        figure = ax.figure
    data.plot(ax=ax)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time")
    if save_path is not None:
        save_figure(figure, save_path)
    return figure, ax


def plot_latlon_field(
    data: xr.DataArray,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    levels: Optional[int] = None,
    use_cartopy: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a latitude-longitude field with optional cartopy support."""
    if use_cartopy:
        if ccrs is None:
            raise ImportError("cartopy is required when `use_cartopy=True`.")
        if ax is None:
            figure, ax = plt.subplots(
                figsize=(8, 4),
                dpi=150,
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
        else:
            figure = ax.figure
        data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels)
        ax.coastlines()
    else:
        if ax is None:
            figure, ax = plt.subplots(figsize=(8, 4), dpi=150)
        else:
            figure = ax.figure
        data.plot(ax=ax, cmap=cmap, levels=levels)
    if title:
        ax.set_title(title)
    if save_path is not None:
        save_figure(figure, save_path)
    return figure, ax


def plot_wk_spectrum(
    result: WKAnalysisResult,
    *,
    max_wn: int = 15,
    max_freq: float = 0.5,
    add_matsuno_lines: bool = True,
    equivalent_depths: Sequence[float] = (8.0, 25.0, 90.0),
    cpd_lines: Sequence[float] = (3.0, 6.0, 30.0),
    cmap: str = "RdBu_r",
    levels: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot normalized symmetric and antisymmetric WK spectra."""
    if levels is None:
        levels = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

    symmetric_norm = result.power_symmetric / result.background
    antisymmetric_norm = result.power_antisymmetric / result.background

    symmetric_plot = symmetric_norm.sel(
        frequency=slice(0.0, max_freq),
        wavenumber=slice(-max_wn, max_wn),
    )
    antisymmetric_plot = antisymmetric_norm.sel(
        frequency=slice(0.0, max_freq),
        wavenumber=slice(-max_wn, max_wn),
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200, constrained_layout=True)

    image = symmetric_plot.plot.contourf(
        ax=axes[0],
        levels=levels,
        cmap=cmap,
        add_colorbar=False,
        extend="neither",
    )
    antisymmetric_plot.plot.contourf(
        ax=axes[1],
        levels=levels,
        cmap=cmap,
        add_colorbar=False,
        extend="neither",
    )

    contour_levels = levels[levels >= 1.1]
    symmetric_plot.plot.contour(
        ax=axes[0],
        levels=contour_levels,
        colors="k",
        linewidths=0.5,
        add_labels=False,
    )
    antisymmetric_plot.plot.contour(
        ax=axes[1],
        levels=contour_levels,
        colors="k",
        linewidths=0.5,
        add_labels=False,
    )

    titles = ("Symmetric Component", "Antisymmetric Component")
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.axvline(0.0, linestyle="--", color="k", linewidth=0.5)
        axis.set_xlim((-max_wn, max_wn))
        axis.set_ylim((0.0, max_freq))
        axis.set_xlabel("Zonal Wavenumber")
        axis.set_ylabel("Frequency (CPD)")

        for period_days in cpd_lines:
            frequency = 1.0 / period_days
            if frequency <= max_freq:
                axis.axhline(frequency, color="k", linestyle=":", linewidth=0.5)
                axis.text(
                    -max_wn + 1,
                    frequency + 0.01,
                    f"{int(period_days)}d",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )

    if add_matsuno_lines:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The iteration is not making good progress",
                category=RuntimeWarning,
            )
            matsuno_modes = matsuno_modes_wk(he=tuple(equivalent_depths), n=(1,), max_wn=max_wn)
        kw_x, kw_y = get_cckw_envelope_curve()
        for axis in axes:
            for depth, frame in matsuno_modes.items():
                axis.plot(frame.index, frame[f"Kelvin(he={depth}m)"], color="k", linewidth=0.8)
                axis.plot(frame.index, frame[f"ER(n=1,he={depth}m)"], color="k", linewidth=0.8)
            axis.plot(kw_x[0], kw_y[0], color="green", linewidth=1.2, zorder=5)

    figure.colorbar(image, ax=axes, orientation="horizontal", shrink=0.6, aspect=30, pad=0.1)

    if save_path is not None:
        save_figure(figure, save_path)

    return figure, axes


def plot_spatial_std_comparison(
    legacy_std: xr.DataArray,
    cckw_std: xr.DataArray,
    *,
    wave_name: str,
    title_suffix: str = "",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot legacy-vs-CCKW time-standard-deviation maps."""
    diff = legacy_std - cckw_std
    vmax = float(np.nanmax([legacy_std.max().item(), cckw_std.max().item()]))
    dmax = float(np.nanmax(np.abs(diff.values)))
    if not np.isfinite(dmax) or dmax == 0.0:
        dmax = 1.0e-8

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180, constrained_layout=True)

    legacy_std.plot.contourf(
        ax=axes[0],
        levels=21,
        cmap="Spectral_r",
        vmin=0.0,
        vmax=vmax,
        add_colorbar=True,
    )
    axes[0].set_title(f"Legacy WaveFilter STD\n{wave_name.upper()}")

    cckw_std.plot.contourf(
        ax=axes[1],
        levels=21,
        cmap="Spectral_r",
        vmin=0.0,
        vmax=vmax,
        add_colorbar=True,
    )
    axes[1].set_title(f"CCKWFilter STD\n{wave_name.upper()}")

    diff.plot.contourf(
        ax=axes[2],
        levels=21,
        cmap="RdBu_r",
        vmin=-dmax,
        vmax=dmax,
        add_colorbar=True,
    )
    axes[2].set_title("Difference\nLegacy - CCKW")

    for axis in axes:
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")

    figure.suptitle(f"Spatial STD Comparison for {wave_name.upper()} {title_suffix}".strip())
    if save_path is not None:
        save_figure(figure, save_path)
    return figure, axes
