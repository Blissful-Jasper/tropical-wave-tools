"""Wheeler-Kiladis spectrum plots and the dispersion-curve overlays they use."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from tropical_wave_tools.config import DEFAULT_WAVE_SPECS
from tropical_wave_tools.matsuno import matsuno_modes_wk
from tropical_wave_tools.plotting._style import (
    _apply_axes_style,
    _style_colorbar,
    get_publication_cmap,
    save_figure,
    scientific_plot_style,
)
from tropical_wave_tools.spectral import WKAnalysisResult


def get_cckw_envelope_curve(
    *,
    he: Sequence[float] | None = None,
    fmax: Sequence[float] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
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


def _plot_dispersion_guides(
    ax: plt.Axes,
    *,
    families: Sequence[str],
    max_wn: int,
    max_freq: float,
    equivalent_depths: Sequence[float],
) -> None:
    """Overlay representative Matsuno dispersion curves for major wave families."""
    if not equivalent_depths:
        return

    family_columns = {
        "Kelvin": ("Kelvin(he={depth}m)", "#0f766e"),
        "ER n=1": ("ER(n=1,he={depth}m)", "#c2410c"),
        "WIG n=1": ("WIG(n=1,he={depth}m)", "#2563eb"),
        "MRG": ("MRG(he={depth}m)", "#7c3aed"),
        "EIG n=0": ("EIG(n=0,he={depth}m)", "#b91c1c"),
        "EIG n=1": ("EIG(n=1,he={depth}m)", "#0f4c81"),
        "WIG n=2": ("WIG(n=2,he={depth}m)", "#1d4ed8"),
        "EIG n=2": ("EIG(n=2,he={depth}m)", "#8b5cf6"),
    }
    max_mode = 2 if any("n=2" in label for label in families) else 1
    matsuno_curves = matsuno_modes_wk(
        he=tuple(float(depth) for depth in equivalent_depths),
        n=tuple(range(1, max_mode + 1)),
        max_wn=max_wn,
    )

    for label in families:
        if label not in family_columns:
            continue
        column_template, color = family_columns[label]
        middle_depth = float(equivalent_depths[len(equivalent_depths) // 2])
        visible_curves: list[tuple[float, np.ndarray, np.ndarray]] = []

        for depth in equivalent_depths:
            curve_depth = float(depth)
            matsuno_frame = matsuno_curves[curve_depth]
            column = column_template.format(depth=curve_depth)
            if column not in matsuno_frame:
                continue

            curve = matsuno_frame[column].astype(float)
            valid = np.isfinite(curve.values) & (curve.values >= 0.0) & (curve.values <= max_freq)
            if not np.any(valid):
                continue
            visible_curves.append((curve_depth, matsuno_frame.index.values[valid], curve.values[valid]))

        if not visible_curves:
            continue

        visible_depths = np.asarray([curve_depth for curve_depth, _, _ in visible_curves], dtype=float)
        preferred_depth = middle_depth if np.any(np.isclose(visible_depths, middle_depth)) else visible_curves[0][0]

        for curve_depth, x_values, y_values in visible_curves:
            is_reference = np.isclose(curve_depth, preferred_depth)
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=1.45 if is_reference else 0.9,
                alpha=0.95 if is_reference else 0.55,
                label=label if is_reference else None,
            )


def _wave_frequency_bounds(wave_name: str) -> tuple[float, float]:
    """Convert one wave's period range into frequency bounds."""
    spec = DEFAULT_WAVE_SPECS[wave_name.lower()]
    period_min, period_max = spec.period_days
    if period_min is None or period_max is None:
        raise ValueError(f"Wave {wave_name} does not define closed period bounds.")
    return (1.0 / float(period_max), 1.0 / float(period_min))


def _add_wk_filter_box(
    ax: plt.Axes,
    *,
    wave_name: str,
    color: str,
    label: str,
    linewidth: float = 1.4,
) -> Rectangle:
    """Overlay one WK filter box based on the configured wave bounds."""
    spec = DEFAULT_WAVE_SPECS[wave_name.lower()]
    freq_min, freq_max = _wave_frequency_bounds(wave_name)
    k_min, k_max = spec.wavenumber
    rectangle = Rectangle(
        (float(k_min), float(freq_min)),
        float(k_max - k_min),
        float(freq_max - freq_min),
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle="--",
        alpha=0.95,
        label=label,
        zorder=6,
    )
    ax.add_patch(rectangle)
    return rectangle


def _set_wk_legend(
    ax: plt.Axes,
    *,
    allowed_labels: Sequence[str],
    rename_map: dict[str, str] | None = None,
    **legend_kwargs: object,
) -> None:
    """Keep only the requested WK legend entries and rename them if needed."""
    rename_map = rename_map or {}
    handles, labels = ax.get_legend_handles_labels()
    selected_handles: list[object] = []
    selected_labels: list[str] = []
    seen: set[str] = set()

    for handle, label in zip(handles, labels):
        mapped = rename_map.get(label, label)
        if mapped not in allowed_labels or mapped in seen:
            continue
        selected_handles.append(handle)
        selected_labels.append(mapped)
        seen.add(mapped)

    if not selected_handles:
        return

    legend = ax.legend(selected_handles, selected_labels, **legend_kwargs)
    legend.get_frame().set_edgecolor("#d4d4d8")


def plot_wk_spectrum(
    result: WKAnalysisResult,
    *,
    max_wn: int = 15,
    max_freq: float = 0.5,
    add_matsuno_lines: bool = True,
    equivalent_depths: Sequence[float] = (12.0, 25.0, 50.0),
    kelvin_band_depths: Sequence[float] = (8.0, 25.0, 90.0),
    cpd_lines: Sequence[float] = (3.0, 6.0, 30.0),
    cmap: str | LinearSegmentedColormap = "wk_power",
    levels: np.ndarray | None = None,
    colorbar_orientation: str = "vertical",
    annotate_filter_boxes: bool = True,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot normalized symmetric and antisymmetric WK spectra."""
    with scientific_plot_style():
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        if levels is None:
            levels = np.array([1.0, 1.15, 1.3, 1.5, 1.8, 2.2, 2.8, 3.6])

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

        figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), dpi=230, constrained_layout=True)
        figure.patch.set_facecolor("white")

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
            colors="white",
            linewidths=0.5,
            add_labels=False,
        )
        antisymmetric_plot.plot.contour(
            ax=axes[1],
            levels=contour_levels,
            colors="white",
            linewidths=0.5,
            add_labels=False,
        )

        titles = ("Symmetric Component", "Antisymmetric Component")
        for axis, title in zip(axes, titles):
            axis.set_title(title)
            axis.axvline(0.0, linestyle="--", color="#64748b", linewidth=0.75)
            axis.set_xlim((-max_wn, max_wn))
            axis.set_ylim((0.0, max_freq))
            axis.set_xlabel("Zonal Wavenumber")
            axis.set_ylabel("Frequency (CPD)")
            _apply_axes_style(axis, grid=True)

            for period_days in cpd_lines:
                frequency = 1.0 / period_days
                if frequency <= max_freq:
                    axis.axhline(frequency, color="#94a3b8", linestyle=":", linewidth=0.6)
                    axis.text(
                        -max_wn + 1,
                        frequency + 0.01,
                        f"{int(period_days)}d",
                        fontsize=8,
                        color="#475569",
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )

        if add_matsuno_lines:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The iteration is not making good progress",
                    category=RuntimeWarning,
                )
                _plot_dispersion_guides(
                    axes[0],
                    families=("Kelvin", "ER n=1"),
                    max_wn=max_wn,
                    max_freq=max_freq,
                    equivalent_depths=equivalent_depths,
                )
                _plot_dispersion_guides(
                    axes[1],
                    families=("MRG", "EIG n=0"),
                    max_wn=max_wn,
                    max_freq=max_freq,
                    equivalent_depths=equivalent_depths,
                )

            kw_x, kw_y = get_cckw_envelope_curve(he=kelvin_band_depths)
            axes[0].plot(
                kw_x[0],
                kw_y[0],
                color="#16a34a",
                linewidth=1.2,
                alpha=0.9,
                zorder=5,
            )

        if annotate_filter_boxes:
            _add_wk_filter_box(axes[0], wave_name="mjo", color="#7c3aed", label="MJO", linewidth=1.55)
            _add_wk_filter_box(axes[1], wave_name="td", color="#ef4444", label="TD", linewidth=1.55)

        _set_wk_legend(
            axes[0],
            allowed_labels=("Kelvin", "ER", "MJO"),
            rename_map={"ER n=1": "ER"},
            loc="upper left",
            fontsize=7.6,
            frameon=True,
            framealpha=0.92,
            borderpad=0.35,
            labelspacing=0.28,
        )
        _set_wk_legend(
            axes[1],
            allowed_labels=("MRG", "TD", "EIG"),
            rename_map={"EIG n=0": "EIG"},
            loc="upper right",
            fontsize=7.6,
            frameon=True,
            framealpha=0.92,
            borderpad=0.35,
            labelspacing=0.28,
        )

        cbar = figure.colorbar(
            image,
            ax=axes,
            orientation=colorbar_orientation,
            shrink=0.72 if colorbar_orientation == "horizontal" else 0.94,
            aspect=32 if colorbar_orientation == "horizontal" else 28,
            pad=0.1 if colorbar_orientation == "horizontal" else 0.025,
        )
        cbar.set_label("Normalized power")
        _style_colorbar(cbar)

        if save_path is not None:
            save_figure(figure, save_path)

        return figure, axes
