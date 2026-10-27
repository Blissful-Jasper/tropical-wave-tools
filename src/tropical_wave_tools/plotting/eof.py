"""EOF pattern, PC and multi-wave EOF summary plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from tropical_wave_tools.plotting._geo import (
    _add_map_guides,
    _case07_focus_half_width,
    _default_map_projection,
    _focus_longitude_center,
    _prepare_projected_field,
    _set_map_extent_from_field,
    _subset_longitude_window,
    ccrs,
)
from tropical_wave_tools.plotting._style import (
    _add_panel_label,
    _add_shared_colorbar,
    _apply_axes_style,
    _filled_levels,
    _integer_levels_from_data,
    _is_weak_signal,
    _symmetric_limit,
    _wave_title_name,
    get_publication_cmap,
    save_figure,
    scientific_plot_style,
)
from tropical_wave_tools.plotting._wind import (
    _add_quiver_key,
    _journal_quiver_kwargs,
    _overlay_anomaly_contours,
    _quiver_reference_value,
)


def plot_eof_spatial_patterns_and_pcs(
    eofs: xr.DataArray,
    pcs: xr.DataArray,
    explained_variance: Sequence[float],
    *,
    modes: Sequence[int] = (1, 2),
    use_cartopy: bool | None = None,
    cmap: str | LinearSegmentedColormap = "olr_diverging",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot EOF spatial patterns alongside their principal components."""
    with scientific_plot_style():
        if not {"mode", "lat", "lon"}.issubset(eofs.dims):
            raise ValueError("`eofs` must contain mode/lat/lon dimensions.")
        time_dim = next((dim for dim in pcs.dims if dim != "mode"), None)
        if time_dim is None:
            raise ValueError("`pcs` must contain one non-mode dimension for time.")

        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        selected_modes = [int(mode) for mode in modes]
        figure = plt.figure(figsize=(13.2, 4.0 * len(selected_modes)), dpi=220, constrained_layout=True)
        grid = figure.add_gridspec(len(selected_modes), 3, width_ratios=(1.05, 0.05, 1.0))
        axes = np.empty((len(selected_modes), 3), dtype=object)
        if use_cartopy:
            projection, data_crs = _default_map_projection()
        else:
            projection = None
            data_crs = None
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap

        for row, mode in enumerate(selected_modes):
            eof_field = eofs.sel(mode=mode)
            pc_series = pcs.sel(mode=mode)
            vmax = _symmetric_limit(eof_field)
            map_ax = figure.add_subplot(grid[row, 0], projection=projection) if use_cartopy else figure.add_subplot(grid[row, 0])
            cbar_ax = figure.add_subplot(grid[row, 1])
            ts_ax = figure.add_subplot(grid[row, 2])
            axes[row, 0] = map_ax
            axes[row, 1] = cbar_ax
            axes[row, 2] = ts_ax

            if use_cartopy:
                plot_data = _prepare_projected_field(eof_field)
                image = plot_data.plot.contourf(
                    ax=map_ax,
                    transform=data_crs,
                    cmap=cmap,
                    levels=_filled_levels(-vmax, vmax),
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
                )
                _set_map_extent_from_field(map_ax, plot_data, data_crs)
                _add_map_guides(map_ax, draw_labels=True)
            else:
                image = eof_field.plot.contourf(
                    ax=map_ax,
                    cmap=cmap,
                    levels=_filled_levels(-vmax, vmax),
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
                )
                _apply_axes_style(map_ax, grid=False)
            explained = float(explained_variance[mode - 1]) if np.isfinite(explained_variance[mode - 1]) else 0.0
            map_ax.set_title(f"EOF {mode} ({explained:.1f}%)")
            map_ax.set_xlabel("Longitude")
            map_ax.set_ylabel("Latitude")
            _add_panel_label(map_ax, chr(65 + 2 * row))
            cbar = figure.colorbar(image, cax=cbar_ax, orientation="vertical")
            cbar.set_label("EOF loading")
            _apply_axes_style(cbar_ax, grid=False)
            if _is_weak_signal(eof_field, threshold=1.0e-6):
                map_ax.text(
                    0.5,
                    0.5,
                    "Weakly resolved\nat daily sampling",
                    transform=map_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="#475569",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "pad": 2.0},
                )

            ts_ax.plot(pc_series[time_dim].values, pc_series.values, color="#0f766e", linewidth=1.5)
            ts_ax.axhline(0.0, color="#64748b", linewidth=0.8, linestyle="--")
            ts_ax.set_title(f"PC {mode}")
            ts_ax.set_xlabel("Time")
            ts_ax.set_ylabel("Amplitude")
            _apply_axes_style(ts_ax, grid=True)
            _add_panel_label(ts_ax, chr(66 + 2 * row))

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_eof_modes_with_wind(
    eofs: xr.DataArray,
    zonal_wind_patterns: xr.DataArray,
    meridional_wind_patterns: xr.DataArray,
    explained_variance: Sequence[float],
    *,
    modes: Sequence[int] = (1, 2),
    wave_name: str | None = None,
    quiver_stride: int = 6,
    quiver_scale: float | None = 55.0,
    use_cartopy: bool | None = None,
    cmap: str | LinearSegmentedColormap = "olr_diverging",
    integer_colorbar: bool = True,
    field_label: str = "PC-regressed OLR (W m$^{-2}$)",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot EOF-derived map patterns with regressed low-level wind vectors."""
    with scientific_plot_style():
        if not {"mode", "lat", "lon"}.issubset(eofs.dims):
            raise ValueError("`eofs` must contain mode/lat/lon dimensions.")
        selected_modes = [int(mode) for mode in modes]
        eof_subset = eofs.sel(mode=selected_modes)
        lon_center = _focus_longitude_center(eof_subset)
        lon_half_width = _case07_focus_half_width(wave_name)
        eof_subset = _subset_longitude_window(eof_subset, center_lon=lon_center, half_width=lon_half_width)
        zonal_wind_patterns = _subset_longitude_window(
            zonal_wind_patterns.sel(mode=selected_modes),
            center_lon=lon_center,
            half_width=lon_half_width,
        )
        meridional_wind_patterns = _subset_longitude_window(
            meridional_wind_patterns.sel(mode=selected_modes),
            center_lon=lon_center,
            half_width=lon_half_width,
        )
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        figure = plt.figure(figsize=(9.25, 2.85 * len(selected_modes) + 0.8), dpi=220, constrained_layout=False)
        grid = figure.add_gridspec(len(selected_modes), 1)
        axes = np.empty(len(selected_modes), dtype=object)
        if use_cartopy:
            projection, data_crs = _default_map_projection()
        else:
            projection = None
            data_crs = None

        if integer_colorbar:
            _, colorbar_ticks = _integer_levels_from_data(eof_subset, symmetric=True, target_steps=8)
            shared_vmax = float(max(abs(colorbar_ticks[0]), abs(colorbar_ticks[-1])))
            contour_levels = _filled_levels(-shared_vmax, shared_vmax, count=25)
        else:
            vmax = _symmetric_limit(eof_subset)
            contour_levels = _filled_levels(-vmax, vmax, count=25)
            colorbar_ticks = None
            shared_vmax = float(vmax)

        quiver_reference = _quiver_reference_value(
            zonal_wind_patterns,
            meridional_wind_patterns,
        )
        last_quiver = None
        for row, mode in enumerate(selected_modes):
            eof_field = eof_subset.sel(mode=mode)
            u_field = zonal_wind_patterns.sel(mode=mode)
            v_field = meridional_wind_patterns.sel(mode=mode)
            map_ax = figure.add_subplot(grid[row, 0], projection=projection) if use_cartopy else figure.add_subplot(grid[row, 0])
            axes[row] = map_ax

            if use_cartopy:
                plot_data = _prepare_projected_field(eof_field)
                image = plot_data.plot.contourf(
                    ax=map_ax,
                    transform=data_crs,
                    cmap=cmap,
                    levels=contour_levels,
                    norm=TwoSlopeNorm(vmin=-shared_vmax, vcenter=0.0, vmax=shared_vmax),
                    add_colorbar=False,
                    extend="neither",
                )
                _overlay_anomaly_contours(map_ax, eof_field, use_cartopy=True, data_crs=data_crs)
                _set_map_extent_from_field(map_ax, plot_data, data_crs)
                _add_map_guides(
                    map_ax,
                    draw_labels=True,
                    label_left=True,
                    label_bottom=row == len(selected_modes) - 1,
                )
            else:
                image = eof_field.plot.contourf(
                    ax=map_ax,
                    cmap=cmap,
                    levels=contour_levels,
                    norm=TwoSlopeNorm(vmin=-shared_vmax, vcenter=0.0, vmax=shared_vmax),
                    add_colorbar=False,
                    extend="neither",
                )
                _overlay_anomaly_contours(map_ax, eof_field)
                _apply_axes_style(map_ax, grid=False)

            u_sub = u_field.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
            v_sub = v_field.isel(lat=slice(None, None, quiver_stride), lon=slice(None, None, quiver_stride))
            lon2d, lat2d = np.meshgrid(u_sub.lon.values, u_sub.lat.values)
            quiver_kwargs = _journal_quiver_kwargs(quiver_scale=quiver_scale, width=0.00195)
            if use_cartopy:
                last_quiver = map_ax.quiver(lon2d, lat2d, u_sub.values, v_sub.values, transform=data_crs, **quiver_kwargs)
            else:
                last_quiver = map_ax.quiver(lon2d, lat2d, u_sub.values, v_sub.values, **quiver_kwargs)

            prefix = _wave_title_name(wave_name).replace(" Wave", "")
            map_ax.set_title(f"{prefix} EOF {mode}", pad=11.5, fontsize=11.5)
            map_ax.set_xlabel("Longitude" if row == len(selected_modes) - 1 else "")
            map_ax.set_ylabel("Latitude")
            _add_panel_label(map_ax, chr(65 + row))
            if _is_weak_signal(eof_field, threshold=max(shared_vmax * 0.05, 1.0e-6)):
                map_ax.text(
                    0.5,
                    0.5,
                    "Weakly resolved\nat daily sampling",
                    transform=map_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#475569",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "pad": 2.0},
                )

        _add_shared_colorbar(
            figure,
            image,
            axes.tolist(),
            label=field_label,
            orientation="horizontal",
            ticks=colorbar_ticks,
            integer=integer_colorbar,
            pad=0.07,
            shrink=0.81,
            aspect=42,
            fraction=0.065,
        )
        if last_quiver is not None:
            _add_quiver_key(axes[-1], last_quiver, x=0.94, y=-0.22, reference=quiver_reference)
        figure.subplots_adjust(left=0.065, right=0.985, top=0.94, bottom=0.22, hspace=0.19)

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes


def plot_multiwave_eof_summary(
    eof_fields: Sequence[xr.DataArray],
    pc_series_list: Sequence[xr.DataArray],
    explained_variances: Sequence[float],
    *,
    wave_names: Sequence[str],
    use_cartopy: bool | None = None,
    cmap: str | LinearSegmentedColormap = "olr_diverging",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot EOF-1 spatial patterns and PC-1 time series for multiple waves."""
    if not (len(eof_fields) == len(pc_series_list) == len(explained_variances) == len(wave_names)):
        raise ValueError("All multi-wave EOF inputs must have the same length.")

    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
        nrows = len(wave_names)
        figure = plt.figure(figsize=(13.2, 2.9 * nrows), dpi=220, constrained_layout=True)
        grid = figure.add_gridspec(nrows, 3, width_ratios=(1.05, 0.05, 1.0))
        axes = np.empty((nrows, 3), dtype=object)
        if use_cartopy:
            projection, data_crs = _default_map_projection()
        else:
            projection = None
            data_crs = None

        for row, (wave_name, eof_field, pc_series, explained) in enumerate(
            zip(wave_names, eof_fields, pc_series_list, explained_variances)
        ):
            map_ax = figure.add_subplot(grid[row, 0], projection=projection) if use_cartopy else figure.add_subplot(grid[row, 0])
            cbar_ax = figure.add_subplot(grid[row, 1])
            ts_ax = figure.add_subplot(grid[row, 2])
            axes[row, 0] = map_ax
            axes[row, 1] = cbar_ax
            axes[row, 2] = ts_ax

            vmax = _symmetric_limit(eof_field)
            if use_cartopy:
                plot_data = _prepare_projected_field(eof_field)
                image = plot_data.plot.contourf(
                    ax=map_ax,
                    transform=data_crs,
                    cmap=cmap,
                    levels=_filled_levels(-vmax, vmax),
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
                )
                _set_map_extent_from_field(map_ax, plot_data, data_crs)
                _add_map_guides(map_ax, draw_labels=True)
            else:
                image = eof_field.plot.contourf(
                    ax=map_ax,
                    cmap=cmap,
                    levels=_filled_levels(-vmax, vmax),
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
                )
                _apply_axes_style(map_ax, grid=False)
            explained_display = float(explained) if np.isfinite(explained) else 0.0
            map_ax.set_title(f"{wave_name.upper()} EOF-1 ({explained_display:.1f}%)")
            map_ax.set_xlabel("Longitude")
            map_ax.set_ylabel("Latitude")
            _add_panel_label(map_ax, chr(65 + 2 * row))
            cbar = figure.colorbar(image, cax=cbar_ax, orientation="vertical")
            cbar.set_label("EOF-1 loading")
            _apply_axes_style(cbar_ax, grid=False)
            if (not np.isfinite(explained)) or _is_weak_signal(eof_field, threshold=max(vmax * 0.05, 1.0e-6)):
                map_ax.text(
                    0.5,
                    0.5,
                    "Weakly resolved\nat daily sampling",
                    transform=map_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#475569",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "pad": 2.0},
                )

            time_dim = next((dim for dim in pc_series.dims if dim != "mode"), pc_series.dims[0])
            ts_ax.plot(pc_series[time_dim].values, pc_series.values, color="#0f766e", linewidth=1.4)
            ts_ax.axhline(0.0, color="#64748b", linewidth=0.8, linestyle="--")
            ts_ax.set_title(f"{wave_name.upper()} PC-1")
            ts_ax.set_xlabel("Time")
            ts_ax.set_ylabel("Amplitude")
            _apply_axes_style(ts_ax, grid=True)
            _add_panel_label(ts_ax, chr(66 + 2 * row))

        if save_path is not None:
            save_figure(figure, save_path)
        return figure, axes
