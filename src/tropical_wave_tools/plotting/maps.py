"""Lat-lon fields, spatial standard deviation, wind diagnostics and horizontal-structure plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from tropical_wave_tools.plotting._geo import (
    _add_map_guides,
    _default_map_projection,
    _focus_longitude_center,
    _lat_lon_last,
    _prepare_projected_field,
    _set_map_extent_from_field,
    _subset_longitude_window,
    ccrs,
)
from tropical_wave_tools.plotting._style import (
    _add_panel_label,
    _add_shared_colorbar,
    _apply_axes_style,
    _continuous_symmetric_levels_from_data,
    _field_label,
    _filled_levels,
    _integer_levels_from_data,
    _is_weak_signal,
    _resolve_extend,
    _scientific_integer_levels_from_data,
    _signed_levels_from_data,
    _std_label,
    _style_colorbar,
    _symmetric_limit,
    get_publication_cmap,
    save_figure,
    scientific_plot_style,
)
from tropical_wave_tools.plotting._wind import (
    _add_quiver_key,
    _add_reference_vector_legend,
    _journal_quiver_kwargs,
    _overlay_anomaly_contours,
    _plot_zonal_wind_contours,
    _prefer_zonal_wind_contours,
    _quiver_reference_value,
)


def plot_latlon_field(
    data: xr.DataArray,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cmap: str | LinearSegmentedColormap = "olr_diverging",
    levels: int | None = None,
    use_cartopy: bool | None = None,
    colorbar_label: str | None = None,
    colorbar_orientation: str = "horizontal",
    integer_colorbar: bool = False,
    zero_floor_colorbar: bool = False,
    target_steps: int = 8,
    level_count: int = 19,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a latitude-longitude field with optional cartopy support."""
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        data = _lat_lon_last(data)
        data_min = float(np.nanmin(data.values))
        data_max = float(np.nanmax(data.values))
        colorbar_ticks = None
        if levels is None:
            if integer_colorbar:
                levels, colorbar_ticks = _integer_levels_from_data(
                    data,
                    symmetric=data_min < 0.0 < data_max,
                    zero_floor=zero_floor_colorbar,
                    target_steps=target_steps,
                )
            elif data_min < 0.0 < data_max:
                symmetric_limit = _symmetric_limit(data)
                levels = _filled_levels(-symmetric_limit, symmetric_limit, count=level_count)
            else:
                low = float(data.quantile(0.02).item())
                high = float(data.quantile(0.98).item())
                if not np.isfinite(low):
                    low = data_min
                if not np.isfinite(high):
                    high = data_max
                levels = _filled_levels(low, high, count=level_count)
        if use_cartopy:
            projection, data_crs = _default_map_projection()
            projected = _prepare_projected_field(data)
            if ax is None:
                figure, ax = plt.subplots(
                    figsize=(9.2, 4.8),
                    dpi=190,
                    subplot_kw={"projection": projection},
                )
            else:
                figure = ax.figure
            plot_kwargs = {
                "ax": ax,
                "transform": data_crs,
                "cmap": cmap,
                "levels": levels,
                "add_colorbar": False,
            }
            if data_min < 0.0 < data_max:
                symmetric_limit = (
                    float(max(abs(levels[0]), abs(levels[-1])))
                    if integer_colorbar and np.ndim(levels) > 0
                    else _symmetric_limit(data)
                )
                plot_kwargs["norm"] = TwoSlopeNorm(vmin=-symmetric_limit, vcenter=0.0, vmax=symmetric_limit)
            image = projected.plot.contourf(
                **plot_kwargs,
            )
            if "lon" in projected.coords and "lat" in projected.coords:
                _set_map_extent_from_field(ax, projected, data_crs)
            _add_map_guides(ax, draw_labels=True)
        else:
            if ax is None:
                figure, ax = plt.subplots(figsize=(8.4, 4.4), dpi=180)
            else:
                figure = ax.figure
            plot_kwargs = {
                "ax": ax,
                "cmap": cmap,
                "levels": levels,
                "add_colorbar": False,
            }
            if data_min < 0.0 < data_max:
                symmetric_limit = (
                    float(max(abs(levels[0]), abs(levels[-1])))
                    if integer_colorbar and np.ndim(levels) > 0
                    else _symmetric_limit(data)
                )
                plot_kwargs["norm"] = TwoSlopeNorm(vmin=-symmetric_limit, vcenter=0.0, vmax=symmetric_limit)
            image = data.plot.contourf(
                **plot_kwargs,
            )
        cbar = figure.colorbar(
            image,
            ax=ax,
            orientation=colorbar_orientation,
            shrink=0.84,
            pad=0.08 if colorbar_orientation == "horizontal" else 0.03,
            aspect=34 if colorbar_orientation == "horizontal" else 28,
        )
        if colorbar_label:
            cbar.set_label(colorbar_label)
        _style_colorbar(cbar, ticks=colorbar_ticks, integer=integer_colorbar)
        if title:
            ax.set_title(title)
        _apply_axes_style(ax, grid=not use_cartopy)
        if save_path is not None:
            save_figure(figure, save_path)
        return figure, ax


def plot_spatial_std_comparison(
    legacy_std: xr.DataArray,
    cckw_std: xr.DataArray,
    *,
    wave_name: str,
    title_suffix: str = "",
    use_cartopy: bool | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot legacy-vs-CCKW time-standard-deviation maps."""
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        legacy_std, cckw_std = _lat_lon_last(legacy_std), _lat_lon_last(cckw_std)
        diff = legacy_std - cckw_std
        vmax = float(np.nanmax([legacy_std.max().item(), cckw_std.max().item()]))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        dmax = float(np.nanmax(np.abs(diff.values)))
        if not np.isfinite(dmax) or dmax == 0.0:
            dmax = 1.0e-8

        subplot_kwargs = {}
        if use_cartopy:
            projection, data_crs = _default_map_projection()
            subplot_kwargs["projection"] = projection

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(15.8, 5.3),
            dpi=210,
            constrained_layout=True,
            subplot_kw=subplot_kwargs or None,
        )

        legacy_plot_data = _prepare_projected_field(legacy_std) if use_cartopy else legacy_std
        cckw_plot_data = _prepare_projected_field(cckw_std) if use_cartopy else cckw_std
        diff_plot_data = _prepare_projected_field(diff) if use_cartopy else diff

        common_kwargs = {
            "levels": 21,
            "vmin": 0.0,
            "vmax": vmax,
            "add_colorbar": False,
        }
        diff_kwargs = {
            "levels": 21,
            "vmin": -dmax,
            "vmax": dmax,
            "add_colorbar": False,
        }
        if use_cartopy:
            common_kwargs["transform"] = data_crs
            diff_kwargs["transform"] = data_crs

        image_main = legacy_plot_data.plot.contourf(
            ax=axes[0],
            cmap=get_publication_cmap("wave_std"),
            **common_kwargs,
        )
        cckw_plot_data.plot.contourf(
            ax=axes[1],
            cmap=get_publication_cmap("wave_std"),
            **common_kwargs,
        )
        image_diff = diff_plot_data.plot.contourf(
            ax=axes[2],
            cmap=get_publication_cmap("wave_diverging"),
            **diff_kwargs,
        )

        axes[0].set_title(f"Legacy WaveFilter STD\n{wave_name.upper()}")
        axes[1].set_title(f"CCKWFilter STD\n{wave_name.upper()}")
        axes[2].set_title("Difference\nLegacy - CCKW")

        for index, axis in enumerate(axes):
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            if use_cartopy:
                _set_map_extent_from_field(axis, legacy_plot_data, data_crs)
                _add_map_guides(axis, draw_labels=True, label_left=index == 0, label_bottom=True)
            else:
                _apply_axes_style(axis, grid=False)
            _add_panel_label(axis, chr(65 + index))

        cbar_main = figure.colorbar(
            image_main,
            ax=axes[:2],
            orientation="horizontal",
            shrink=0.78,
            aspect=34,
            pad=0.08,
        )
        cbar_main.set_label("Standard deviation")
        cbar_diff = figure.colorbar(
            image_diff,
            ax=axes[2],
            orientation="horizontal",
            shrink=0.78,
            aspect=34,
            pad=0.08,
        )
        cbar_diff.set_label("Legacy - CCKW")

        figure.suptitle(
            f"Spatial STD Comparison for {wave_name.upper()} {title_suffix}".strip(),
            y=0.985,
            fontsize=12.5,
        )
        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_spatial_std_triptych(
    olr_std: xr.DataArray,
    u_std: xr.DataArray,
    v_std: xr.DataArray,
    *,
    titles: Sequence[str] = ("Filtered OLR STD", "Filtered U850 STD", "Filtered V850 STD"),
    use_cartopy: bool | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a projected OLR/U850/V850 standard-deviation triptych."""
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        olr_std, u_std, v_std = _lat_lon_last(olr_std), _lat_lon_last(u_std), _lat_lon_last(v_std)
        subplot_kwargs = {}
        if use_cartopy:
            projection, data_crs = _default_map_projection()
            subplot_kwargs["projection"] = projection

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(15.8, 5.4),
            dpi=210,
            constrained_layout=True,
            subplot_kw=subplot_kwargs or None,
        )

        plot_data = [
            _prepare_projected_field(olr_std) if use_cartopy else olr_std,
            _prepare_projected_field(u_std) if use_cartopy else u_std,
            _prepare_projected_field(v_std) if use_cartopy else v_std,
        ]
        cmaps = (
            get_publication_cmap("wave_std"),
            get_publication_cmap("wind_std"),
            get_publication_cmap("wind_std"),
        )
        labels = (
            _std_label(olr_std),
            _std_label(u_std),
            _std_label(v_std),
        )

        for index, (axis, data, title, cmap, label) in enumerate(zip(axes, plot_data, titles, cmaps, labels)):
            vmax = float(np.nanmax(data.values))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            kwargs = {
                "ax": axis,
                "cmap": cmap,
                "levels": _filled_levels(0.0, vmax),
                "vmin": 0.0,
                "vmax": vmax,
                "add_colorbar": False,
            }
            if use_cartopy:
                kwargs["transform"] = data_crs
            image = data.plot.contourf(**kwargs)
            axis.set_title(title)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            if use_cartopy:
                _set_map_extent_from_field(axis, data, data_crs)
                _add_map_guides(axis, draw_labels=True, label_left=index == 0, label_bottom=True)
            else:
                _apply_axes_style(axis, grid=False)
            _add_panel_label(axis, chr(65 + index))
            cbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.08, shrink=0.84, aspect=30)
            cbar.set_label(label)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_wave_spatial_comparison(
    fields: Sequence[xr.DataArray],
    *,
    titles: Sequence[str],
    colorbar_label: str = "Filtered OLR STD",
    colorbar_orientation: str = "vertical",
    use_cartopy: bool | None = None,
    ncols: int = 2,
    cmap: str | LinearSegmentedColormap = "wave_std",
    integer_colorbar: bool = False,
    target_steps: int = 8,
    level_count: int = 19,
    weak_signal_threshold: float | None = 0.1,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a grid comparing filtered spatial distributions across multiple waves.

    A panel whose peak is below ``weak_signal_threshold`` (in the units of the fields) is annotated
    "Weak signal at daily sampling"; pass ``None`` to switch the annotation off, for example for
    fields in units where 0.1 is not small.
    """
    if len(fields) != len(titles):
        raise ValueError("`fields` and `titles` must have the same length.")
    fields = [_lat_lon_last(field) for field in fields]

    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        n_panels = len(fields)
        nrows = int(np.ceil(n_panels / ncols))
        subplot_kwargs = {}
        if use_cartopy:
            projection, data_crs = _default_map_projection()
            subplot_kwargs["projection"] = projection
        figure_width = 7.8 if ncols == 1 else 5.15 * ncols
        figure_height = (1.85 * nrows + 0.35) if (use_cartopy and ncols == 1) else 3.0 * nrows
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figure_width, figure_height),
            dpi=220,
            constrained_layout=True,
            subplot_kw=subplot_kwargs or None,
        )
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        vmax = max(float(np.nanmax(field.values)) for field in fields)
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        if integer_colorbar:
            contour_levels, colorbar_ticks = _integer_levels_from_data(
                fields,
                symmetric=False,
                zero_floor=True,
                target_steps=target_steps,
            )
            vmax = float(contour_levels[-1])
        else:
            contour_levels = _filled_levels(0.0, vmax, count=level_count)
            colorbar_ticks = None
        extend = _resolve_extend(None, fields, contour_levels)
        image = None

        for panel_index, (axis, field, title) in enumerate(zip(axes.flat, fields, titles)):
            row, col = divmod(panel_index, ncols)
            plot_data = _prepare_projected_field(field) if use_cartopy else field
            kwargs = {
                "ax": axis,
                "cmap": cmap,
                "levels": contour_levels,
                "vmin": 0.0,
                "vmax": vmax,
                "add_colorbar": False,
                "extend": extend,
            }
            if use_cartopy:
                kwargs["transform"] = data_crs
            image = plot_data.plot.contourf(**kwargs)
            axis.set_title(title)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            if use_cartopy:
                _set_map_extent_from_field(axis, plot_data, data_crs)
                _add_map_guides(axis, draw_labels=True, label_left=col == 0, label_bottom=row == nrows - 1)
            else:
                _apply_axes_style(axis, grid=False)
            _add_panel_label(axis, chr(65 + panel_index))
            if weak_signal_threshold is not None and _is_weak_signal(field, threshold=weak_signal_threshold):
                axis.text(
                    0.5,
                    0.5,
                    "Weak signal\nat daily sampling",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.1,
                    color="#475569",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "pad": 2.0},
                )

        spare_axes = list(axes.flat[n_panels:])
        auxiliary_axis: plt.Axes | None = None
        if spare_axes and colorbar_orientation == "horizontal" and nrows > 1:
            auxiliary_axis = spare_axes[0]
            auxiliary_axis.set_visible(True)
            auxiliary_axis.set_axis_off()
            spare_axes = spare_axes[1:]
        for axis in spare_axes:
            axis.set_visible(False)

        if image is not None:
            panel_axes = [axis for index, axis in enumerate(axes.flat) if index < n_panels]
            if colorbar_orientation == "vertical" and ncols == 1:
                colorbar_pad = 0.01
                colorbar_shrink = 0.84
                colorbar_aspect = 40
                colorbar_fraction = 0.03
            else:
                colorbar_pad = 0.045 if colorbar_orientation == "horizontal" else 0.015
                colorbar_shrink = 0.86 if colorbar_orientation == "horizontal" else 0.94
                colorbar_aspect = 42 if colorbar_orientation == "horizontal" else 34
                colorbar_fraction = 0.055 if colorbar_orientation == "horizontal" else 0.038
            _add_shared_colorbar(
                figure,
                image,
                panel_axes,
                label=colorbar_label,
                orientation=colorbar_orientation,
                ticks=colorbar_ticks,
                integer=integer_colorbar,
                pad=colorbar_pad,
                shrink=colorbar_shrink,
                aspect=colorbar_aspect,
                fraction=colorbar_fraction,
            )

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_wind_diagnostics_panel(
    divergence: xr.DataArray,
    vorticity: xr.DataArray,
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
    *,
    titles: Sequence[str] = ("Low-level divergence", "Low-level vorticity"),
    quiver_stride: int = 4,
    quiver_scale: float | None = 55.0,
    integer_colorbar: bool = True,
    colorbar_extend: str | None = None,
    use_cartopy: bool | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot divergence and vorticity maps with wind vectors.

    ``colorbar_extend=None`` (default) extends the colorbars when values fall outside the scale.
    """
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        divergence, vorticity = _lat_lon_last(divergence), _lat_lon_last(vorticity)
        zonal_wind, meridional_wind = _lat_lon_last(zonal_wind), _lat_lon_last(meridional_wind)
        subplot_kwargs = {}
        if use_cartopy:
            projection, data_crs = _default_map_projection()
            subplot_kwargs["projection"] = projection
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(13.25, 5.15),
            dpi=220,
            constrained_layout=False,
            subplot_kw=subplot_kwargs or None,
        )

        diagnostics = (
            (divergence, get_publication_cmap("wave_diverging"), _field_label("Divergence", divergence)),
            (vorticity, get_publication_cmap("olr_diverging"), _field_label("Vorticity", vorticity)),
        )
        quiver_reference = _quiver_reference_value(zonal_wind, meridional_wind)
        last_quiver = None
        for index, (axis, title, (field, cmap, label)) in enumerate(zip(axes, titles, diagnostics)):
            if integer_colorbar:
                contour_levels, colorbar_ticks, vmax = _scientific_integer_levels_from_data(field)
            else:
                vmax = _symmetric_limit(field, quantile=0.99)
                contour_levels = _filled_levels(-vmax, vmax)
                colorbar_ticks = None
            plot_data = _prepare_projected_field(field) if use_cartopy else field
            kwargs = {
                "ax": axis,
                "cmap": cmap,
                "levels": contour_levels,
                "norm": TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                "add_colorbar": False,
                "extend": _resolve_extend(colorbar_extend, field, contour_levels),
            }
            if use_cartopy:
                kwargs["transform"] = data_crs
            image = plot_data.plot.contourf(**kwargs)
            _overlay_anomaly_contours(
                axis,
                field,
                use_cartopy=use_cartopy,
                data_crs=data_crs if use_cartopy else None,
            )

            u_sub = zonal_wind.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
            v_sub = meridional_wind.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
            lon2d, lat2d = np.meshgrid(u_sub.lon.values, u_sub.lat.values)
            quiver_kwargs = _journal_quiver_kwargs(quiver_scale=quiver_scale, width=0.0020)
            if use_cartopy:
                last_quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, transform=data_crs, **quiver_kwargs)
                _set_map_extent_from_field(axis, plot_data, data_crs)
                _add_map_guides(axis, draw_labels=True, label_left=index == 0, label_bottom=True)
            else:
                last_quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, **quiver_kwargs)
                _apply_axes_style(axis, grid=False)
            axis.set_title(title)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            _add_panel_label(axis, chr(65 + index))
            cbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.12, shrink=0.82, aspect=34)
            cbar.set_label(label)
            _style_colorbar(cbar, ticks=colorbar_ticks, scientific_integer=integer_colorbar)
        if last_quiver is not None:
            legend_axis = figure.add_axes([0.895, 0.29, 0.09, 0.15])
            _add_reference_vector_legend(legend_axis, reference=quiver_reference)
        figure.subplots_adjust(left=0.055, right=0.88, top=0.9, bottom=0.205, wspace=0.1)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_horizontal_structure(
    olr_field: xr.DataArray,
    u_field: xr.DataArray,
    v_field: xr.DataArray,
    *,
    title: str | None = None,
    quiver_stride: int = 5,
    quiver_scale: float | None = 55.0,
    colorbar_orientation: str = "horizontal",
    integer_colorbar: bool = False,
    colorbar_extend: str | None = None,
    wind_overlay: str = "auto",
    use_cartopy: bool | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one horizontal structure map with OLR shading and wind vectors.

    ``colorbar_extend=None`` (default) extends the colorbar when values fall outside the scale.
    """
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        olr_field, u_field, v_field = _lat_lon_last(olr_field), _lat_lon_last(u_field), _lat_lon_last(v_field)
        vmax = _symmetric_limit(olr_field)
        cmap = get_publication_cmap("olr_diverging")
        if integer_colorbar:
            contour_levels, colorbar_ticks = _integer_levels_from_data(olr_field, symmetric=True)
            vmax = float(max(abs(contour_levels[0]), abs(contour_levels[-1])))
        else:
            contour_levels = _filled_levels(-vmax, vmax)
            colorbar_ticks = None
        extend = _resolve_extend(colorbar_extend, olr_field, contour_levels)

        if use_cartopy:
            projection, data_crs = _default_map_projection()
            olr_plot = _prepare_projected_field(olr_field)
            figure, axis = plt.subplots(figsize=(10.4, 4.9), dpi=220, constrained_layout=True, subplot_kw={"projection": projection})
            image = olr_plot.plot.contourf(
                ax=axis,
                transform=data_crs,
                cmap=cmap,
                levels=contour_levels,
                norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                add_colorbar=False,
                extend=extend,
            )
            _set_map_extent_from_field(axis, olr_plot, data_crs)
            _add_map_guides(axis, draw_labels=True)
        else:
            figure, axis = plt.subplots(figsize=(9.6, 4.7), dpi=220, constrained_layout=True)
            image = olr_field.plot.contourf(
                ax=axis,
                cmap=cmap,
                levels=contour_levels,
                norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                add_colorbar=False,
                extend=extend,
            )
            _apply_axes_style(axis, grid=False)
        _overlay_anomaly_contours(
            axis,
            olr_field,
            use_cartopy=use_cartopy,
            data_crs=data_crs if use_cartopy else None,
        )

        use_u_contours = wind_overlay == "u_contours" or (
            wind_overlay == "auto" and _prefer_zonal_wind_contours(u_field, v_field)
        )
        quiver = None
        if use_u_contours:
            _plot_zonal_wind_contours(axis, u_field, use_cartopy=use_cartopy, data_crs=data_crs if use_cartopy else None)
            axis.text(
                0.985,
                0.98,
                "U850 contours\nsolid + / dashed -",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.8,
                color="#0f172a",
                bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "#cbd5e1", "pad": 1.8},
            )
        else:
            quiver_reference = _quiver_reference_value(u_field, v_field)
            u_sub = u_field.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
            v_sub = v_field.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
            lon2d, lat2d = np.meshgrid(u_sub.lon.values, u_sub.lat.values)
            quiver_kwargs = _journal_quiver_kwargs(quiver_scale=quiver_scale, width=0.00175)
            if use_cartopy:
                quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, transform=data_crs, **quiver_kwargs)
            else:
                quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, **quiver_kwargs)

        axis.set_title(title or "Wave horizontal structure")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        cbar = figure.colorbar(
            image,
            ax=axis,
            orientation=colorbar_orientation,
            pad=0.08 if colorbar_orientation == "horizontal" else 0.025,
            shrink=0.78 if colorbar_orientation == "horizontal" else 0.92,
            aspect=34 if colorbar_orientation == "horizontal" else 30,
        )
        cbar.set_label(_field_label("Filtered OLR", olr_field))
        _style_colorbar(cbar, ticks=colorbar_ticks, integer=integer_colorbar)
        if quiver is not None:
            _add_quiver_key(axis, quiver, x=0.9, y=-0.16, reference=quiver_reference)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axis


def plot_wave_horizontal_structure_comparison(
    olr_fields: Sequence[xr.DataArray],
    zonal_winds: Sequence[xr.DataArray],
    meridional_winds: Sequence[xr.DataArray],
    *,
    titles: Sequence[str],
    ncols: int = 2,
    quiver_stride: int = 6,
    colorbar_orientation: str = "vertical",
    quiver_scale: float | None = 55.0,
    use_cartopy: bool | None = None,
    weak_signal_threshold: float | None = 0.1,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare lag-0 horizontal structures for multiple waves in one figure.

    ``weak_signal_threshold`` works as in :func:`plot_wave_spatial_comparison`.
    """
    if not (len(olr_fields) == len(zonal_winds) == len(meridional_winds) == len(titles)):
        raise ValueError("All input sequences must have the same length.")
    olr_fields = [_lat_lon_last(field) for field in olr_fields]
    zonal_winds = [_lat_lon_last(field) for field in zonal_winds]
    meridional_winds = [_lat_lon_last(field) for field in meridional_winds]

    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        n_panels = len(olr_fields)
        nrows = int(np.ceil(n_panels / ncols))
        subplot_kwargs = {}
        if use_cartopy:
            projection, data_crs = _default_map_projection()
            subplot_kwargs["projection"] = projection

        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.25 * ncols, 3.05 * nrows),
            dpi=220,
            constrained_layout=True,
            subplot_kw=subplot_kwargs or None,
        )
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        vmax = max(_symmetric_limit(field) for field in olr_fields)
        shared_levels = _filled_levels(-vmax, vmax)
        extend = _resolve_extend(None, olr_fields, shared_levels)
        image = None
        quiver_reference = _quiver_reference_value(
            xr.concat(list(zonal_winds), dim="panel"),
            xr.concat(list(meridional_winds), dim="panel"),
        )
        last_quiver = None

        for panel_index, (axis, olr_field, zonal_wind, meridional_wind, title) in enumerate(
            zip(axes.flat, olr_fields, zonal_winds, meridional_winds, titles)
        ):
            row, col = divmod(panel_index, ncols)
            if use_cartopy:
                plot_data = _prepare_projected_field(olr_field)
                image = plot_data.plot.contourf(
                    ax=axis,
                    transform=data_crs,
                    cmap=get_publication_cmap("olr_diverging"),
                    levels=shared_levels,
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
                    extend=extend,
                )
                _set_map_extent_from_field(axis, plot_data, data_crs)
                _add_map_guides(axis, draw_labels=True, label_left=col == 0, label_bottom=row == nrows - 1)
            else:
                image = olr_field.plot.contourf(
                    ax=axis,
                    cmap=get_publication_cmap("olr_diverging"),
                    levels=shared_levels,
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
                    extend=extend,
                )
                _apply_axes_style(axis, grid=False)
            _overlay_anomaly_contours(
                axis,
                olr_field,
                use_cartopy=use_cartopy,
                data_crs=data_crs if use_cartopy else None,
            )

            u_sub = zonal_wind.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
            v_sub = meridional_wind.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
            lon2d, lat2d = np.meshgrid(u_sub.lon.values, u_sub.lat.values)
            quiver_kwargs = _journal_quiver_kwargs(quiver_scale=quiver_scale, width=0.0016)
            if use_cartopy:
                last_quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, transform=data_crs, **quiver_kwargs)
            else:
                last_quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, **quiver_kwargs)
            axis.set_title(title)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            _add_panel_label(axis, chr(65 + panel_index))
            if weak_signal_threshold is not None and _is_weak_signal(olr_field, threshold=weak_signal_threshold):
                axis.text(
                    0.5,
                    0.5,
                    "Weak signal\nat daily sampling",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.1,
                    color="#475569",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "pad": 2.0},
                )

        spare_axes = list(axes.flat[n_panels:])
        auxiliary_axis: plt.Axes | None = None
        if spare_axes and colorbar_orientation == "horizontal" and nrows > 1:
            auxiliary_axis = spare_axes[0]
            auxiliary_axis.set_visible(True)
            auxiliary_axis.set_axis_off()
            spare_axes = spare_axes[1:]
        for axis in spare_axes:
            axis.set_visible(False)

        if image is not None:
            visible_axes = [axis for index, axis in enumerate(axes.flat) if index < n_panels]
            _add_shared_colorbar(
                figure,
                image,
                visible_axes,
                label=_field_label("Filtered OLR", olr_fields[0]),
                orientation=colorbar_orientation,
            )
        visible_axes = [axis for index, axis in enumerate(axes.flat) if index < n_panels]
        if visible_axes and last_quiver is not None:
            _add_quiver_key(visible_axes[-1], last_quiver, x=0.93, y=-0.14, reference=quiver_reference)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_lagged_horizontal_structure(
    olr_lagged: xr.DataArray,
    u_lagged: xr.DataArray,
    v_lagged: xr.DataArray,
    *,
    lags: Sequence[int] | None = None,
    ncols: int = 3,
    quiver_stride: int = 6,
    quiver_scale: float | None = 55.0,
    colorbar_orientation: str = "horizontal",
    integer_colorbar: bool = False,
    colorbar_extend: str | None = None,
    olr_quantile: float = 0.98,
    olr_level_count: int = 19,
    olr_range_scale: float = 1.0,
    olr_min_vmax: float | None = None,
    wind_overlay: str = "auto",
    quiver_width: float = 0.0016,
    quiver_headwidth: float = 2.9,
    quiver_headlength: float = 3.7,
    quiver_headaxislength: float = 3.4,
    focus_longitude: bool = False,
    focus_center_lon: float | None = None,
    focus_half_width: float | None = None,
    suptitle: str | None = None,
    panel_title_template: str = "Lag {lag:+d} d",
    use_cartopy: bool | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot lagged horizontal composites with OLR shading and low-level wind vectors.

    ``colorbar_extend=None`` (default) extends the shared colorbar when values fall outside the
    scale (the scale spans ``olr_quantile`` of ``|OLR|``, so the strongest signals are painted with
    the end colours instead of being left blank).
    """
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        olr_lagged, u_lagged, v_lagged = _lat_lon_last(olr_lagged), _lat_lon_last(u_lagged), _lat_lon_last(v_lagged)
        if lags is None:
            lags = [int(value) for value in olr_lagged["lag"].values]
        lagged_olr = olr_lagged.sel(lag=list(lags))
        lagged_u = u_lagged.sel(lag=list(lags))
        lagged_v = v_lagged.sel(lag=list(lags))
        if focus_longitude:
            center_lon = _focus_longitude_center(lagged_olr.sel(lag=0)) if focus_center_lon is None else float(focus_center_lon)
            half_width = 90.0 if focus_half_width is None else float(focus_half_width)
            lagged_olr = _subset_longitude_window(lagged_olr, center_lon=center_lon, half_width=half_width)
            lagged_u = _subset_longitude_window(lagged_u, center_lon=center_lon, half_width=half_width)
            lagged_v = _subset_longitude_window(lagged_v, center_lon=center_lon, half_width=half_width)
        vmax = _symmetric_limit(lagged_olr, quantile=olr_quantile)
        if integer_colorbar:
            olr_levels, colorbar_ticks = _integer_levels_from_data(lagged_olr, symmetric=True)
            vmax = float(max(abs(olr_levels[0]), abs(olr_levels[-1])))
        else:
            olr_levels, colorbar_ticks, vmax = _continuous_symmetric_levels_from_data(
                lagged_olr,
                quantile=olr_quantile,
                count=olr_level_count,
                target_tick_steps=8,
                range_scale=olr_range_scale,
                minimum_vmax=olr_min_vmax,
            )
        extend = _resolve_extend(colorbar_extend, lagged_olr, olr_levels)
        n_panels = len(lags)
        nrows = int(np.ceil(n_panels / ncols))

        subplot_kwargs = {}
        if use_cartopy:
            projection, data_crs = _default_map_projection()
            subplot_kwargs["projection"] = projection
        if nrows == 1 and ncols >= 4:
            figure_width = 3.05 * ncols
            figure_height = 2.45
        else:
            figure_width = 9.0 if ncols == 1 else 5.25 * ncols
            figure_height = (2.85 * nrows + 0.6) if (use_cartopy and ncols == 1) else 3.05 * nrows
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figure_width, figure_height),
            dpi=220,
            constrained_layout=False,
            subplot_kw=subplot_kwargs or None,
        )
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        image = None
        use_u_contours = wind_overlay == "u_contours" or (
            wind_overlay == "auto" and _prefer_zonal_wind_contours(lagged_u, lagged_v)
        )
        quiver_reference = _quiver_reference_value(lagged_u, lagged_v)
        u_contour_levels = _signed_levels_from_data(lagged_u, target_steps=8) if use_u_contours else None
        last_quiver = None

        for panel_index, (axis, lag) in enumerate(zip(axes.flat, lags)):
            row, col = divmod(panel_index, ncols)
            olr_panel = lagged_olr.sel(lag=lag)
            u_panel = lagged_u.sel(lag=lag)
            v_panel = lagged_v.sel(lag=lag)
            if use_cartopy:
                olr_plot = _prepare_projected_field(olr_panel)
                image = olr_plot.plot.contourf(
                    ax=axis,
                    transform=data_crs,
                    cmap=get_publication_cmap("olr_diverging"),
                    levels=olr_levels,
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
                    extend=extend,
                )
                _set_map_extent_from_field(axis, olr_plot, data_crs)
                _add_map_guides(axis, draw_labels=True, label_left=col == 0, label_bottom=row == nrows - 1)
            else:
                image = olr_panel.plot.contourf(
                    ax=axis,
                    cmap=get_publication_cmap("olr_diverging"),
                    levels=olr_levels,
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
                    extend=extend,
                )
                _apply_axes_style(axis, grid=False)
            _overlay_anomaly_contours(
                axis,
                olr_panel,
                use_cartopy=use_cartopy,
                data_crs=data_crs if use_cartopy else None,
            )

            if use_u_contours:
                _plot_zonal_wind_contours(
                    axis,
                    u_panel,
                    levels=u_contour_levels,
                    use_cartopy=use_cartopy,
                    data_crs=data_crs if use_cartopy else None,
                )
                if panel_index == 0:
                    axis.text(
                        0.985,
                        0.98,
                        "U850 contours\nsolid + / dashed -",
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7.8,
                        color="#0f172a",
                        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "#cbd5e1", "pad": 1.8},
                    )
            else:
                u_sub = u_panel.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
                v_sub = v_panel.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
                lon2d, lat2d = np.meshgrid(u_sub.lon.values, u_sub.lat.values)
                quiver_kwargs = _journal_quiver_kwargs(
                    quiver_scale=quiver_scale,
                    width=quiver_width,
                    headwidth=quiver_headwidth,
                    headlength=quiver_headlength,
                    headaxislength=quiver_headaxislength,
                )
                if use_cartopy:
                    last_quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, transform=data_crs, **quiver_kwargs)
                else:
                    last_quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, **quiver_kwargs)
            axis.set_title(panel_title_template.format(lag=int(lag)))
            if use_cartopy:
                axis.set_xlabel("")
                axis.set_ylabel("")
            else:
                axis.set_xlabel("Longitude" if row == nrows - 1 else "")
                axis.set_ylabel("Latitude")
            _add_panel_label(axis, chr(65 + panel_index))

        spare_axes = list(axes.flat[n_panels:])
        auxiliary_axis: plt.Axes | None = None
        if spare_axes and colorbar_orientation == "horizontal" and nrows > 1:
            auxiliary_axis = spare_axes[0]
            auxiliary_axis.set_visible(True)
            auxiliary_axis.set_axis_off()
            spare_axes = spare_axes[1:]
        for axis in spare_axes:
            axis.set_visible(False)

        if image is not None:
            panel_axes = [axis for index, axis in enumerate(axes.flat) if index < n_panels]
            if auxiliary_axis is not None:
                panel_boxes = [axis.get_position() for axis in panel_axes]
                left = min(box.x0 for box in panel_boxes)
                right = max(box.x1 for box in panel_boxes)
                bottom = min(box.y0 for box in panel_boxes)
                cbar_left = left + 0.18 * (right - left)
                cbar_width = 0.56 * (right - left)
                cbar_bottom = max(0.055, bottom - 0.075)
                cax = figure.add_axes([cbar_left, cbar_bottom, cbar_width, 0.022])
                cbar = figure.colorbar(
                    image,
                    cax=cax,
                    orientation=colorbar_orientation,
                )
            else:
                cbar = figure.colorbar(
                    image,
                    ax=panel_axes,
                    orientation=colorbar_orientation,
                    pad=(0.04 if nrows > 1 else 0.06) if colorbar_orientation == "horizontal" else 0.02,
                    shrink=0.76 if (colorbar_orientation == "horizontal" and nrows > 1) else (0.8 if colorbar_orientation == "horizontal" else 0.94),
                    aspect=36 if colorbar_orientation == "horizontal" else 32,
                )
            cbar.set_label(_field_label("Filtered OLR", olr_lagged))
            _style_colorbar(cbar, ticks=colorbar_ticks, integer=integer_colorbar)
            if last_quiver is not None:
                if auxiliary_axis is not None:
                    _add_reference_vector_legend(auxiliary_axis, reference=quiver_reference)
                elif colorbar_orientation == "horizontal" and nrows > 1:
                    _add_quiver_key(panel_axes[-1], last_quiver, x=0.8, y=0.1, reference=quiver_reference)
                else:
                    _add_quiver_key(panel_axes[-1], last_quiver, x=0.93, y=-0.14, reference=quiver_reference)
        if suptitle:
            figure.suptitle(suptitle, y=0.965, fontsize=13.0, fontweight="semibold")
        if nrows == 1 and ncols >= 4:
            figure.subplots_adjust(left=0.04, right=0.995, top=0.8, bottom=0.23, wspace=0.045)
        else:
            figure.subplots_adjust(left=0.055, right=0.99, top=0.92, bottom=0.19, hspace=0.09, wspace=0.06)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_wave_evolution_comparison(
    olr_lagged_fields: Sequence[xr.DataArray],
    zonal_lagged_fields: Sequence[xr.DataArray],
    meridional_lagged_fields: Sequence[xr.DataArray],
    *,
    wave_names: Sequence[str],
    lags: Sequence[int] = (-6, 0, 6),
    quiver_stride: int = 6,
    colorbar_orientation: str = "vertical",
    quiver_scale: float | None = 55.0,
    use_cartopy: bool | None = None,
    weak_signal_threshold: float | None = 0.1,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare OLR-wind horizontal evolution across multiple waves and lags.

    ``weak_signal_threshold`` works as in :func:`plot_wave_spatial_comparison`.
    """
    if not (
        len(olr_lagged_fields)
        == len(zonal_lagged_fields)
        == len(meridional_lagged_fields)
        == len(wave_names)
    ):
        raise ValueError("All input sequences must have the same length.")
    olr_lagged_fields = [_lat_lon_last(field) for field in olr_lagged_fields]
    zonal_lagged_fields = [_lat_lon_last(field) for field in zonal_lagged_fields]
    meridional_lagged_fields = [_lat_lon_last(field) for field in meridional_lagged_fields]

    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        nrows = len(wave_names)
        ncols = len(lags)
        subplot_kwargs = {}
        if use_cartopy:
            projection, data_crs = _default_map_projection()
            subplot_kwargs["projection"] = projection
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.15 * ncols, 3.0 * nrows),
            dpi=220,
            constrained_layout=True,
            subplot_kw=subplot_kwargs or None,
        )
        axes = np.atleast_2d(axes)
        vmax = max(_symmetric_limit(field.sel(lag=list(lags))) for field in olr_lagged_fields)
        shared_levels = _filled_levels(-vmax, vmax)
        extend = _resolve_extend(None, [field.sel(lag=list(lags)) for field in olr_lagged_fields], shared_levels)
        image = None
        quiver_reference = _quiver_reference_value(
            xr.concat(list(zonal_lagged_fields), dim="panel"),
            xr.concat(list(meridional_lagged_fields), dim="panel"),
        )
        last_quiver = None

        for row, (wave_name, olr_lagged, zonal_lagged, meridional_lagged) in enumerate(
            zip(wave_names, olr_lagged_fields, zonal_lagged_fields, meridional_lagged_fields)
        ):
            for col, lag in enumerate(lags):
                axis = axes[row, col]
                olr_panel = olr_lagged.sel(lag=lag)
                u_panel = zonal_lagged.sel(lag=lag)
                v_panel = meridional_lagged.sel(lag=lag)

                if use_cartopy:
                    plot_data = _prepare_projected_field(olr_panel)
                    image = plot_data.plot.contourf(
                        ax=axis,
                        transform=data_crs,
                        cmap=get_publication_cmap("olr_diverging"),
                        levels=shared_levels,
                        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                        add_colorbar=False,
                        extend=extend,
                    )
                    _set_map_extent_from_field(axis, plot_data, data_crs)
                    _add_map_guides(axis, draw_labels=True, label_left=col == 0, label_bottom=row == nrows - 1)
                else:
                    image = olr_panel.plot.contourf(
                        ax=axis,
                        cmap=get_publication_cmap("olr_diverging"),
                        levels=shared_levels,
                        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                        add_colorbar=False,
                        extend=extend,
                    )
                    _apply_axes_style(axis, grid=False)
                _overlay_anomaly_contours(
                    axis,
                    olr_panel,
                    use_cartopy=use_cartopy,
                    data_crs=data_crs if use_cartopy else None,
                )

                u_sub = u_panel.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
                v_sub = v_panel.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
                lon2d, lat2d = np.meshgrid(u_sub.lon.values, u_sub.lat.values)
                quiver_kwargs = _journal_quiver_kwargs(quiver_scale=quiver_scale, width=0.00155)
                if use_cartopy:
                    last_quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, transform=data_crs, **quiver_kwargs)
                else:
                    last_quiver = axis.quiver(lon2d, lat2d, u_sub.values, v_sub.values, **quiver_kwargs)

                axis.set_title(f"{wave_name.upper()} | Lag {lag:+d} d")
                axis.set_xlabel("Longitude")
                axis.set_ylabel("Latitude")
                _add_panel_label(axis, chr(65 + row * ncols + col))
                if weak_signal_threshold is not None and _is_weak_signal(olr_panel, threshold=weak_signal_threshold):
                    axis.text(
                        0.5,
                        0.5,
                        "Weak signal\nat daily sampling",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7.8,
                        color="#475569",
                        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "pad": 1.8},
                    )

        if image is not None:
            panel_axes = axes.ravel().tolist()
            _add_shared_colorbar(
                figure,
                image,
                panel_axes,
                label=_field_label("Filtered OLR", olr_lagged_fields[0]),
                orientation=colorbar_orientation,
            )
            if last_quiver is not None:
                _add_quiver_key(axes[-1, -1], last_quiver, x=0.93, y=-0.14, reference=quiver_reference)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes
