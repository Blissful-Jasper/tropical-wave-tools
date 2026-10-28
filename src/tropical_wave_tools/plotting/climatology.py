"""Monthly cycles, longitude-month heatmaps, variance-fraction cycles and annual trends."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from tropical_wave_tools.plotting._style import (
    _add_panel_label,
    _apply_axes_style,
    _filled_levels,
    _is_weak_signal,
    _nice_step,
    _normalize_cycle_values,
    _resolve_extend,
    _style_colorbar,
    _wave_title_name,
    get_publication_cmap,
    save_figure,
    scientific_plot_style,
)


def plot_monthly_cycle(
    months: Sequence[int],
    series: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a monthly cycle for multiple variables."""
    colors = ("#0f766e", "#9a3412", "#334155", "#7c3aed")
    with scientific_plot_style():
        figure, axis = plt.subplots(figsize=(7.6, 4.2), dpi=210, constrained_layout=True)
        for index, (label, values) in enumerate(series.items()):
            axis.plot(
                months,
                values,
                marker="o",
                linewidth=2.0,
                markersize=4.2,
                color=colors[index % len(colors)],
                label=label,
            )
        axis.set_xlim(1, 12)
        axis.set_xticks(np.arange(1, 13))
        axis.set_xlabel("Month")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(loc="upper right", fontsize=8)
        _apply_axes_style(axis, grid=True)
        axis.ticklabel_format(style="plain", axis="y", useOffset=False)
        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axis


def plot_case05_seasonal_variance_cycles(
    tropical_mean: xr.DataArray,
    tropical_std: xr.DataArray,
    *,
    wave_order: Sequence[str] = ("kelvin", "er", "mrg", "td"),
    ylabel: str = "Variance contribution [%]",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a Fig. 12 style seasonal cycle of variance fractions for four waves."""
    with scientific_plot_style():
        figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), dpi=220, constrained_layout=False)
        axes = np.atleast_1d(axes).reshape(2, 2)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        for panel_index, (axis, wave_name) in enumerate(zip(axes.flat, wave_order)):
            mean_values = np.asarray(tropical_mean.sel(wave=wave_name).values, dtype=float)
            std_values = np.asarray(tropical_std.sel(wave=wave_name).values, dtype=float)
            upper = np.nanmax(mean_values + std_values)
            if not np.isfinite(upper) or upper <= 0.0:
                upper = 1.0
            tick_step = _nice_step(upper, target_steps=5, integer=False)
            ylim = max(tick_step, tick_step * np.ceil(1.08 * upper / tick_step))
            months = np.arange(1, 13)
            axis.fill_between(
                months,
                np.clip(mean_values - std_values, 0.0, None),
                mean_values + std_values,
                color="#cbd5e1",
                alpha=0.24,
                zorder=1,
            )
            axis.errorbar(
                months,
                mean_values,
                yerr=std_values,
                fmt="o-",
                color="#111827",
                ecolor="#94a3b8",
                elinewidth=0.9,
                capsize=2.0,
                capthick=0.8,
                linewidth=1.45,
                markersize=3.4,
                markerfacecolor="white",
                markeredgewidth=0.95,
                zorder=3,
            )
            axis.set_title(_wave_title_name(wave_name).replace(" Wave", ""))
            axis.set_xlim(1, 12)
            axis.set_ylim(0.0, ylim)
            axis.set_xticks(months)
            axis.set_xticklabels(month_labels, fontsize=7.8)
            axis.set_yticks(np.arange(0.0, ylim + 0.5 * tick_step, tick_step))
            axis.set_ylabel(ylabel)
            axis.grid(True, color="#cbd5e1", alpha=0.42, linewidth=0.55, linestyle=":")
            _apply_axes_style(axis, grid=False)
            _add_panel_label(axis, chr(65 + panel_index))

        figure.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.09, hspace=0.28, wspace=0.18)
        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_case05_regional_variance_cycles(
    regional_mean: xr.DataArray,
    regional_std: xr.DataArray,
    *,
    region_order: Sequence[str],
    wave_order: Sequence[str] = ("kelvin", "er", "mrg", "td"),
    region_labels: dict[str, str] | None = None,
    ylabel: str = "Variance contribution [%]",
    note_text: str = "Monthly mean ± 1σ\n20°S–20°N regional mean\nVariance fraction",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a Fig. 14 style regional seasonal-cycle comparison across waves."""
    with scientific_plot_style():
        figure, axes = plt.subplots(4, 2, figsize=(12.2, 11.2), dpi=220, constrained_layout=False)
        axes = np.atleast_1d(axes).reshape(4, 2)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        wave_styles = {
            "kelvin": {"color": "#0f766e", "marker": "o"},
            "er": {"color": "#1d4ed8", "marker": "s"},
            "mrg": {"color": "#c2410c", "marker": "^"},
            "td": {"color": "#111827", "marker": "D"},
        }

        legend_handles = []
        legend_labels = []
        for panel_index, region_name in enumerate(region_order):
            axis = axes.flat[panel_index]
            panel_max = 0.0
            for wave_name in wave_order:
                mean_values = np.asarray(regional_mean.sel(region=region_name, wave=wave_name).values, dtype=float)
                std_values = np.asarray(regional_std.sel(region=region_name, wave=wave_name).values, dtype=float)
                style = wave_styles[wave_name]
                handle = axis.errorbar(
                    np.arange(1, 13),
                    mean_values,
                    yerr=std_values,
                    fmt=f"{style['marker']}-",
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=0.72,
                    alpha=0.92,
                    capsize=1.8,
                    capthick=0.7,
                    linewidth=1.2,
                    markersize=3.0,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    zorder=3,
                )
                if panel_index == 0:
                    legend_handles.append(handle[0])
                    legend_labels.append(_wave_title_name(wave_name).replace(" Wave", ""))
                panel_max = max(panel_max, float(np.nanmax(mean_values + std_values)))

            tick_step = _nice_step(panel_max if panel_max > 0 else 1.0, target_steps=4, integer=False)
            ylim = max(tick_step, tick_step * np.ceil(1.08 * max(panel_max, tick_step) / tick_step))
            axis.set_title(region_labels.get(region_name, region_name) if region_labels else region_name)
            axis.set_xlim(1, 12)
            axis.set_ylim(0.0, ylim)
            axis.set_xticks(np.arange(1, 13))
            axis.set_xticklabels(month_labels, fontsize=7.3)
            axis.set_yticks(np.arange(0.0, ylim + 0.5 * tick_step, tick_step))
            axis.set_ylabel(ylabel)
            axis.grid(True, color="#cbd5e1", alpha=0.42, linewidth=0.55, linestyle=":")
            _apply_axes_style(axis, grid=False)
            _add_panel_label(axis, chr(65 + panel_index))

        legend_axis = axes.flat[len(region_order)]
        legend_axis.set_axis_off()
        legend = legend_axis.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            frameon=True,
            ncol=1,
            fontsize=9.0,
            borderpad=0.6,
            handlelength=2.1,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#cbd5e1")
        legend_axis.text(
            0.02,
            0.34,
            note_text,
            ha="left",
            va="top",
            fontsize=8.8,
            color="#334155",
            transform=legend_axis.transAxes,
        )
        for axis in axes.flat[len(region_order) + 1 :]:
            axis.set_axis_off()

        figure.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.07, hspace=0.42, wspace=0.2)
        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_monthly_longitude_heatmap(
    data: xr.DataArray,
    *,
    title: str,
    cmap: str | LinearSegmentedColormap = "seasonal",
    colorbar_label: str = "Projected monthly RMS",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a month-longitude heatmap for seasonal wave evolution."""
    with scientific_plot_style():
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        figure, axis = plt.subplots(figsize=(9.4, 4.3), dpi=220, constrained_layout=True)
        vmax = float(data.quantile(0.98).item())
        vmin = float(data.quantile(0.02).item())
        if not np.isfinite(vmin):
            vmin = float(np.nanmin(data.values))
        if not np.isfinite(vmax):
            vmax = float(np.nanmax(data.values))
        image = data.plot.contourf(
            ax=axis,
            x="lon",
            y="month",
            cmap=cmap,
            levels=_filled_levels(vmin, vmax),
            add_colorbar=False,
        )
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Month")
        axis.set_yticks(np.arange(1, 13))
        _apply_axes_style(axis, grid=False)
        cbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.08, shrink=0.84, aspect=36)
        cbar.set_label(colorbar_label)
        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axis


def plot_wave_monthly_cycle_comparison(
    monthly_cycles: dict[str, dict[str, np.ndarray]],
    *,
    wave_names: Sequence[str],
    ncols: int = 3,
    monthly_significance: dict[str, dict[str, xr.DataArray]] | None = None,
    normalize_each_series: bool = False,
    normalization: str = "annual_mean",
    weak_signal_threshold: float | None = 0.1,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot monthly RMS annual cycles for multiple waves.

    A panel whose series all peak below ``weak_signal_threshold`` is annotated "Weak signal at daily
    sampling"; pass ``None`` to switch the annotation off.
    """
    with scientific_plot_style():
        n_panels = len(wave_names)
        nrows = int(np.ceil(n_panels / ncols))
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.7 * ncols, 3.2 * nrows),
            dpi=220,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        months = np.arange(1, 13)
        palette = {"OLR": "#0f766e", "U850": "#c2410c", "V850": "#334155"}

        for panel_index, (axis, wave_name) in enumerate(zip(axes.flat, wave_names)):
            series = monthly_cycles[wave_name]
            for label, values in series.items():
                raw_values = np.asarray(values, dtype=float)
                plot_values = raw_values
                if normalize_each_series:
                    plot_values, _ = _normalize_cycle_values(raw_values, method=normalization)
                axis.plot(
                    months,
                    plot_values,
                    marker="o",
                    linewidth=1.8,
                    markersize=3.8,
                    color=palette.get(label, "#7c3aed"),
                    label=label,
                )
                if monthly_significance is not None and wave_name in monthly_significance and label in monthly_significance[wave_name]:
                    pvalues = np.asarray(monthly_significance[wave_name][label].values, dtype=float)
                    significant = np.isfinite(pvalues) & (pvalues < 0.05)
                    if significant.any():
                        axis.scatter(
                            months[significant],
                            plot_values[significant],
                            s=18,
                            color=palette.get(label, "#7c3aed"),
                            edgecolor="#0f172a",
                            linewidth=0.4,
                            zorder=4,
                        )
            if normalize_each_series:
                axis.axhline(1.0, color="#94a3b8", linewidth=0.9, linestyle="--")
            cycle_title = "relative annual cycle" if normalize_each_series else "annual cycle"
            axis.set_title(f"{wave_name.upper()} {cycle_title}")
            axis.set_xlim(1, 12)
            axis.set_xticks(np.arange(1, 13, 2))
            axis.set_xlabel("Month")
            axis.set_ylabel("Relative RMS amplitude" if normalize_each_series else "Projected RMS")
            _apply_axes_style(axis, grid=True)
            axis.ticklabel_format(style="plain", axis="y", useOffset=False)
            _add_panel_label(axis, chr(65 + panel_index))
            if panel_index == 0:
                axis.legend(loc="upper right", fontsize=7.8)
                notes: list[str] = []
                if normalize_each_series:
                    notes.append("Normalized by annual mean")
                if monthly_significance is not None:
                    notes.append("Filled markers: p < 0.05")
                if notes:
                    axis.text(
                        0.02,
                        0.03,
                        "\n".join(notes),
                        transform=axis.transAxes,
                        fontsize=7.1,
                        color="#475569",
                    )
            if weak_signal_threshold is not None and all(
                _is_weak_signal(values, threshold=weak_signal_threshold) for values in series.values()
            ):
                axis.text(
                    0.5,
                    0.5,
                    "Weak signal\nat daily sampling",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#475569",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "pad": 2.0},
                )

        for axis in axes.flat[n_panels:]:
            axis.set_visible(False)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_wave_monthly_longitude_comparison(
    monthly_longitude_fields: Sequence[xr.DataArray],
    *,
    wave_names: Sequence[str],
    ncols: int = 3,
    cmap: str | LinearSegmentedColormap = "seasonal",
    colorbar_label: str = "Projected monthly RMS",
    significance_fields: Sequence[xr.DataArray] | None = None,
    weak_signal_threshold: float | None = 0.1,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot month-longitude seasonal evolution for multiple waves.

    ``significance_fields`` are p-values on (month, lon) in either dimension order; cells with
    p < 0.05 are hatched. ``weak_signal_threshold`` works as in :func:`plot_wave_monthly_cycle_comparison`.
    """
    if len(monthly_longitude_fields) != len(wave_names):
        raise ValueError("`monthly_longitude_fields` and `wave_names` must have the same length.")

    with scientific_plot_style():
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        n_panels = len(wave_names)
        nrows = int(np.ceil(n_panels / ncols))
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.8 * ncols, 3.0 * nrows),
            dpi=220,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        vmax = max(float(field.quantile(0.98).item()) for field in monthly_longitude_fields)
        vmin = min(float(field.quantile(0.02).item()) for field in monthly_longitude_fields)
        shared_levels = _filled_levels(vmin, vmax)
        extend = _resolve_extend(None, list(monthly_longitude_fields), shared_levels)
        image = None

        for panel_index, (axis, wave_name, field) in enumerate(zip(axes.flat, wave_names, monthly_longitude_fields)):
            image = field.plot.contourf(
                ax=axis,
                x="lon",
                y="month",
                cmap=cmap,
                levels=shared_levels,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                extend=extend,
            )
            if significance_fields is not None:
                significance = significance_fields[panel_index].transpose("month", "lon")
                significant_mask = xr.where(significance < 0.05, 1.0, np.nan)
                axis.contourf(
                    significance["lon"],
                    significance["month"],
                    significant_mask,
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=[".."],
                )
            axis.set_title(f"{wave_name.upper()} seasonal evolution")
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Month")
            axis.set_yticks(np.arange(1, 13, 2))
            _apply_axes_style(axis, grid=False)
            _add_panel_label(axis, chr(65 + panel_index))
            if panel_index == 0 and significance_fields is not None:
                axis.text(
                    0.02,
                    0.03,
                    "Hatching: p < 0.05",
                    transform=axis.transAxes,
                    fontsize=7.1,
                    color="#475569",
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2},
                )
            if weak_signal_threshold is not None and _is_weak_signal(field, threshold=weak_signal_threshold):
                axis.text(
                    0.5,
                    0.5,
                    "Weak signal\nat daily sampling",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#475569",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "pad": 2.0},
                )

        for axis in axes.flat[n_panels:]:
            axis.set_visible(False)

        if image is not None:
            cbar = figure.colorbar(
                image,
                ax=[axis for index, axis in enumerate(axes.flat) if index < n_panels],
                orientation="horizontal",
                pad=0.06,
                shrink=0.84,
                aspect=40,
            )
            cbar.set_label(colorbar_label)
            _style_colorbar(cbar)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_wave_annual_trend_comparison(
    annual_series: dict[str, xr.DataArray],
    *,
    wave_names: Sequence[str],
    ncols: int = 3,
    trend_pvalues: dict[str, float] | None = None,
    weak_signal_threshold: float | None = 0.1,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot annual RMS time series with fitted linear trends for multiple waves.

    ``weak_signal_threshold`` works as in :func:`plot_wave_monthly_cycle_comparison`.
    """
    with scientific_plot_style():
        n_panels = len(wave_names)
        nrows = int(np.ceil(n_panels / ncols))
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.8 * ncols, 3.1 * nrows),
            dpi=220,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).reshape(nrows, ncols)

        for panel_index, (axis, wave_name) in enumerate(zip(axes.flat, wave_names)):
            series = annual_series[wave_name]
            years = np.asarray(series["year"].values, dtype=float)
            values = np.asarray(series.values, dtype=float)
            valid = np.isfinite(years) & np.isfinite(values)
            axis.plot(years[valid], values[valid], color="#0f766e", linewidth=1.8, marker="o", markersize=3.6)
            if valid.sum() >= 2:
                slope, intercept = np.polyfit(years[valid], values[valid], deg=1)
                axis.plot(
                    years[valid],
                    slope * years[valid] + intercept,
                    color="#c2410c",
                    linewidth=1.4,
                    linestyle="--",
                )
                pvalue = None if trend_pvalues is None else trend_pvalues.get(wave_name)
                significance = ""
                if pvalue is not None and np.isfinite(pvalue):
                    significance = f" (p={pvalue:.3f})"
                    if pvalue < 0.05:
                        significance += " *"
                title = f"{wave_name.upper()} annual RMS trend\n{slope * 10.0:+.2f} per decade{significance}"
            else:
                title = f"{wave_name.upper()} annual RMS trend"
            axis.set_title(title)
            axis.set_xlabel("Year")
            axis.set_ylabel("Projected RMS")
            _apply_axes_style(axis, grid=True)
            _add_panel_label(axis, chr(65 + panel_index))
            if weak_signal_threshold is not None and _is_weak_signal(
                values[valid] if valid.any() else values, threshold=weak_signal_threshold
            ):
                axis.text(
                    0.5,
                    0.5,
                    "Weak signal\nat daily sampling",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#475569",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "pad": 2.0},
                )

        for axis in axes.flat[n_panels:]:
            axis.set_visible(False)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes
