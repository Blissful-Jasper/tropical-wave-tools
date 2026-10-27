"""Time series, Hovmoller and lag-longitude plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from tropical_wave_tools.plotting._geo import (
    _format_longitude_label,
)
from tropical_wave_tools.plotting._style import (
    _add_panel_label,
    _apply_axes_style,
    _continuous_symmetric_levels_from_data,
    _field_label,
    _filled_levels,
    _integer_levels_from_data,
    _nice_step,
    _style_colorbar,
    _symmetric_limit,
    get_publication_cmap,
    save_figure,
    scientific_plot_style,
)
from tropical_wave_tools.plotting._wind import (
    _overlay_anomaly_contours,
)


def plot_time_series(
    data: xr.DataArray,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a simple time series."""
    with scientific_plot_style():
        if ax is None:
            figure, ax = plt.subplots(figsize=(8, 4), dpi=170)
        else:
            figure = ax.figure
        data.plot(ax=ax, color="#0f766e", linewidth=1.7)
        if title:
            ax.set_title(title)
        ax.set_xlabel("Time")
        _apply_axes_style(ax, grid=True)
        if save_path is not None:
            save_figure(figure, save_path)
        return figure, ax


def plot_hovmoller_comparison(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    left_title: str = "Raw anomaly",
    right_title: str = "Filtered signal",
    x: str = "lon",
    y: str = "time",
    cmap: str | LinearSegmentedColormap = "wave_diverging",
    quantile: float = 0.98,
    colorbar_label: str = "Amplitude",
    colorbar_orientation: str = "vertical",
    integer_colorbar: bool = False,
    colorbar_extend: str = "neither",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot two Hovmoller panels with a shared symmetric color scale."""
    with scientific_plot_style():
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("`plot_hovmoller_comparison` expects two 2D DataArray inputs.")

        left_scale = float(np.abs(left).quantile(quantile).item())
        right_scale = float(np.abs(right).quantile(quantile).item())
        vmax = max(left_scale, right_scale)
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0

        figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=210, constrained_layout=True)
        if integer_colorbar:
            contour_levels, colorbar_ticks = _integer_levels_from_data([left.values, right.values], symmetric=True)
            vmax = float(max(abs(contour_levels[0]), abs(contour_levels[-1])))
        else:
            contour_levels = _filled_levels(-vmax, vmax, count=23)
            colorbar_ticks = None
        image = left.plot.contourf(
            ax=axes[0],
            x=x,
            y=y,
            cmap=cmap,
            levels=contour_levels,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            add_colorbar=False,
            extend=colorbar_extend,
        )
        right.plot.contourf(
            ax=axes[1],
            x=x,
            y=y,
            cmap=cmap,
            levels=contour_levels,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            add_colorbar=False,
            extend=colorbar_extend,
        )

        for index, (axis, title) in enumerate(zip(axes, (left_title, right_title))):
            axis.set_title(title)
            axis.set_xlabel("Longitude" if x == "lon" else x)
            axis.set_ylabel("Time" if y == "time" else y)
            _apply_axes_style(axis, grid=False)
            _add_panel_label(axis, chr(65 + index))

        cbar = figure.colorbar(
            image,
            ax=axes,
            orientation=colorbar_orientation,
            shrink=0.8 if colorbar_orientation == "horizontal" else 0.92,
            aspect=34 if colorbar_orientation == "horizontal" else 28,
            pad=0.09 if colorbar_orientation == "horizontal" else 0.025,
        )
        cbar.set_label(colorbar_label)
        _style_colorbar(cbar, ticks=colorbar_ticks, integer=integer_colorbar)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_hovmoller_triptych(
    fields: Sequence[xr.DataArray],
    *,
    titles: Sequence[str],
    cmaps: Sequence[str | LinearSegmentedColormap] | None = None,
    colorbar_labels: Sequence[str] | None = None,
    x: str = "lon",
    y: str = "time",
    quantile: float = 0.98,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot three Hovmoller panels with publication-style colorbars."""
    if len(fields) != len(titles):
        raise ValueError("`fields` and `titles` must have the same length.")
    if cmaps is None:
        cmaps = ("olr_diverging", "wave_diverging", "wave_diverging")
    if colorbar_labels is None:
        colorbar_labels = ("OLR anomaly", "Zonal wind", "Meridional wind")

    with scientific_plot_style():
        n_panels = len(fields)
        figure, axes = plt.subplots(1, n_panels, figsize=(4.9 * n_panels, 5.1), dpi=220, constrained_layout=True)
        axes = np.atleast_1d(axes)

        for index, (axis, data, title, cmap_name, label) in enumerate(
            zip(axes, fields, titles, cmaps, colorbar_labels)
        ):
            vmax = _symmetric_limit(data, quantile=quantile)
            image = data.plot.contourf(
                ax=axis,
                x=x,
                y=y,
                cmap=get_publication_cmap(str(cmap_name)) if isinstance(cmap_name, str) else cmap_name,
                levels=_filled_levels(-vmax, vmax, count=23),
                norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                add_colorbar=False,
            )
            axis.set_title(title)
            axis.set_xlabel("Longitude" if x == "lon" else x)
            axis.set_ylabel("Time" if y == "time" else y)
            _apply_axes_style(axis, grid=False)
            cbar = figure.colorbar(image, ax=axis, orientation="vertical", pad=0.02, shrink=0.9, aspect=28)
            cbar.set_label(label)
            _add_panel_label(axis, chr(65 + index))

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_paper_style_hovmoller(
    shading: xr.DataArray,
    contours: xr.DataArray,
    *,
    title: str,
    base_point_label: str,
    cmap: str | LinearSegmentedColormap = "paper_hovmoller_diverging",
    figsize: tuple[float, float] = (7.2, 5.0),
    shading_label: str = "Regressed convection proxy (W m$^{-2}$)",
    shading_quantile: float = 0.985,
    shading_level_count: int = 37,
    shading_range_scale: float = 1.0,
    shading_min_vmax: float | None = None,
    contour_target_steps: int = 6,
    contour_quantile: float = 0.98,
    xticks: Sequence[float] | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a Lubis & Jacobi style lagged-regression Hovmoller panel."""
    if not {"lag", "lon"}.issubset(shading.dims) or not {"lag", "lon"}.issubset(contours.dims):
        raise ValueError("`shading` and `contours` must contain lag/lon dimensions.")

    with scientific_plot_style():
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        figure, axis = plt.subplots(figsize=figsize, dpi=220, constrained_layout=False)

        shading_levels, shading_ticks, shading_vmax = _continuous_symmetric_levels_from_data(
            shading,
            quantile=shading_quantile,
            count=shading_level_count,
            range_scale=shading_range_scale,
            minimum_vmax=shading_min_vmax,
        )
        image = shading.plot.contourf(
            ax=axis,
            x="lon",
            y="lag",
            cmap=cmap,
            levels=shading_levels,
            norm=TwoSlopeNorm(vmin=-shading_vmax, vcenter=0.0, vmax=shading_vmax),
            add_colorbar=False,
            extend="neither",
        )

        contour_limit = _symmetric_limit(contours, quantile=contour_quantile)
        contour_step = _nice_step(2.0 * contour_limit, target_steps=contour_target_steps, integer=False)
        contour_limit = max(contour_step, contour_step * np.ceil(contour_limit / contour_step))
        contour_levels = np.arange(-contour_limit, contour_limit + 0.5 * contour_step, contour_step)
        negative_levels = contour_levels[contour_levels < 0.0]
        positive_levels = contour_levels[contour_levels > 0.0]
        if negative_levels.size:
            contours.plot.contour(
                ax=axis,
                x="lon",
                y="lag",
                levels=negative_levels,
                colors="#334155",
                linewidths=0.85,
                linestyles="--",
                add_colorbar=False,
                zorder=3,
            )
        if positive_levels.size:
            contours.plot.contour(
                ax=axis,
                x="lon",
                y="lag",
                levels=positive_levels,
                colors="#334155",
                linewidths=0.85,
                linestyles="-",
                add_colorbar=False,
                zorder=3,
            )
        if np.nanmin(contours.values) < 0.0 < np.nanmax(contours.values):
            contours.plot.contour(
                ax=axis,
                x="lon",
                y="lag",
                levels=[0.0],
                colors="#111827",
                linewidths=1.0,
                linestyles="-",
                add_colorbar=False,
                zorder=3,
            )

        axis.invert_yaxis()
        axis.axhline(0.0, color="#64748b", linewidth=0.65, linestyle=":", zorder=2)
        axis.set_title("")
        axis.set_title(title, loc="left", fontsize=11.2, pad=8.0)
        axis.text(
            0.995,
            0.995,
            base_point_label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.3,
            color="#334155",
            bbox={
                "facecolor": "white",
                "alpha": 0.86,
                "edgecolor": "#cbd5e1",
                "pad": 1.0,
            },
        )
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Lag (days)")
        axis.grid(True, color="#cbd5e1", alpha=0.42, linewidth=0.55, linestyle=":")
        _apply_axes_style(axis, grid=False)

        if xticks is not None:
            axis.set_xticks(list(xticks))
            axis.set_xticklabels([_format_longitude_label(value) for value in xticks])

        cbar_axis = figure.add_axes([0.15, 0.085, 0.72, 0.028])
        cbar = figure.colorbar(
            image,
            cax=cbar_axis,
            orientation="horizontal",
        )
        cbar.set_label(shading_label)
        _style_colorbar(cbar, ticks=shading_ticks, scientific_integer=False)

        figure.subplots_adjust(left=0.1, right=0.985, top=0.88, bottom=0.235)
        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axis


def plot_lag_longitude_evolution(
    lagged_equatorial_field: xr.DataArray,
    *,
    title: str,
    cmap: str | LinearSegmentedColormap = "olr_diverging",
    integer_colorbar: bool = False,
    colorbar_extend: str = "neither",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot lag-versus-longitude evolution for a composite field."""
    with scientific_plot_style():
        if not {"lag", "lon"}.issubset(lagged_equatorial_field.dims):
            raise ValueError("`lagged_equatorial_field` must contain lag/lon dimensions.")
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        vmax = _symmetric_limit(lagged_equatorial_field, quantile=0.99)
        figure, axis = plt.subplots(figsize=(10.0, 4.8), dpi=220, constrained_layout=True)
        if integer_colorbar:
            contour_levels, colorbar_ticks = _integer_levels_from_data(lagged_equatorial_field, symmetric=True)
            vmax = float(max(abs(contour_levels[0]), abs(contour_levels[-1])))
        else:
            contour_levels = _filled_levels(-vmax, vmax, count=25)
            colorbar_ticks = None
        image = lagged_equatorial_field.plot.contourf(
            ax=axis,
            x="lon",
            y="lag",
            cmap=cmap,
            levels=contour_levels,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            add_colorbar=False,
            extend=colorbar_extend,
        )
        _overlay_anomaly_contours(axis, lagged_equatorial_field, x="lon", y="lag")
        axis.axhline(0.0, color="#334155", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Lag (days)")
        _apply_axes_style(axis, grid=False)
        cbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.08, shrink=0.84, aspect=36)
        cbar.set_label(_field_label("Composite amplitude", lagged_equatorial_field))
        _style_colorbar(cbar, ticks=colorbar_ticks, integer=integer_colorbar)
        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axis
