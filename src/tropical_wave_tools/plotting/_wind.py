"""Wind-vector and contour overlay helpers shared by the horizontal-structure plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tropical_wave_tools.plotting._geo import (
    _prepare_projected_field,
)
from tropical_wave_tools.plotting._style import (
    _finite_abs_max,
    _nice_step,
    _signed_levels_from_data,
)


def _quiver_reference_value(
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
    *,
    quantile: float = 0.85,
) -> float:
    """Return a readable reference-vector magnitude for a wind field."""
    speed = np.hypot(np.asarray(zonal_wind.values, dtype=float), np.asarray(meridional_wind.values, dtype=float))
    finite = speed[np.isfinite(speed)]
    if finite.size == 0:
        return 1.0
    representative = float(np.nanquantile(finite, quantile))
    if not np.isfinite(representative) or representative <= 0.0:
        representative = float(np.nanmax(finite))
    if not np.isfinite(representative) or representative <= 0.0:
        return 1.0
    return max(_nice_step(representative, target_steps=4, integer=False), 0.1)


def _journal_quiver_kwargs(
    *,
    quiver_scale: float | None,
    width: float = 0.0020,
    headwidth: float = 4.2,
    headlength: float = 5.3,
    headaxislength: float = 4.8,
    color: str = "#0f172a",
) -> dict[str, object]:
    """Return a consistent publication-style quiver configuration."""
    kwargs: dict[str, object] = {
        "color": color,
        "pivot": "mid",
        "width": width,
        "headwidth": headwidth,
        "headlength": headlength,
        "headaxislength": headaxislength,
        "linewidths": 0.34,
        "edgecolors": "#f8fafc",
        "units": "width",
        "scale_units": "width",
        "minlength": 0.0,
        "minshaft": 2.0,
        "alpha": 0.97,
        "zorder": 4,
    }
    if quiver_scale is not None:
        kwargs["scale"] = quiver_scale
    return kwargs


def _add_quiver_key(
    axis: plt.Axes,
    quiver: object,
    *,
    x: float,
    y: float,
    reference: float,
    coordinates: str = "axes",
    labelpos: str = "E",
) -> object:
    """Attach a compact, publication-style reference-vector label."""
    key = axis.quiverkey(
        quiver,
        x,
        y,
        reference,
        f"{reference:g} m s$^{{-1}}$",
        labelpos=labelpos,
        coordinates=coordinates,
        fontproperties={"size": 8.4},
    )
    key.text.set_color("#0f172a")
    key.text.set_bbox(
        {
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#cbd5e1",
            "pad": 1.4,
        }
    )
    return key


def _add_reference_vector_legend(
    axis: plt.Axes,
    *,
    reference: float,
    color: str = "#0f172a",
) -> None:
    """Draw a standalone reference-vector legend outside the data panels."""
    axis.set_axis_off()
    axis.annotate(
        "",
        xy=(0.64, 0.54),
        xytext=(0.30, 0.54),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.45,
            "color": color,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "mutation_scale": 10.0,
        },
    )
    axis.text(
        0.47,
        0.68,
        f"{reference:g} m s$^{{-1}}$",
        ha="center",
        va="center",
        fontsize=8.1,
        color=color,
        transform=axis.transAxes,
        bbox={
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cbd5e1",
            "pad": 1.4,
        },
    )


def _prefer_zonal_wind_contours(zonal_wind: xr.DataArray, meridional_wind: xr.DataArray) -> bool:
    """Prefer zonal-wind contours when meridional wind is unresolved."""
    zonal_peak = _finite_abs_max(zonal_wind)
    meridional_peak = _finite_abs_max(meridional_wind)
    if zonal_peak <= 0.0:
        return False
    return meridional_peak < max(0.05, 0.15 * zonal_peak)


def _plot_zonal_wind_contours(
    axis: plt.Axes,
    zonal_wind: xr.DataArray,
    *,
    levels: np.ndarray | None = None,
    use_cartopy: bool = False,
    data_crs: object | None = None,
) -> None:
    """Overlay zonal-wind contours using solid/dashed lines for sign."""
    contour_levels = _signed_levels_from_data(zonal_wind) if levels is None else np.asarray(levels, dtype=float)
    plot_data = _prepare_projected_field(zonal_wind) if use_cartopy else zonal_wind
    contour_kwargs = {"ax": axis, "linewidths": 1.2, "alpha": 0.95}
    if use_cartopy and data_crs is not None:
        contour_kwargs["transform"] = data_crs

    negative_levels = contour_levels[contour_levels < 0.0]
    positive_levels = contour_levels[contour_levels > 0.0]
    if negative_levels.size:
        plot_data.plot.contour(
            levels=negative_levels,
            colors="#1d4ed8",
            linestyles="--",
            add_colorbar=False,
            **contour_kwargs,
        )
    if positive_levels.size:
        plot_data.plot.contour(
            levels=positive_levels,
            colors="#7f1d1d",
            linestyles="-",
            add_colorbar=False,
            **contour_kwargs,
        )

    minimum = float(np.nanmin(plot_data.values))
    maximum = float(np.nanmax(plot_data.values))
    if np.isfinite(minimum) and np.isfinite(maximum) and minimum < 0.0 < maximum:
        plot_data.plot.contour(
            levels=[0.0],
            colors="#111827",
            linewidths=0.8,
            linestyles="-",
            add_colorbar=False,
            **({"ax": axis, "transform": data_crs} if use_cartopy and data_crs is not None else {"ax": axis}),
        )


def _overlay_anomaly_contours(
    axis: plt.Axes,
    field: xr.DataArray,
    *,
    levels: np.ndarray | None = None,
    x: str | None = None,
    y: str | None = None,
    use_cartopy: bool = False,
    data_crs: object | None = None,
) -> None:
    """Overlay sparse anomaly contours to sharpen structure boundaries."""
    contour_levels = _signed_levels_from_data(field, target_steps=6) if levels is None else np.asarray(levels, dtype=float)
    if contour_levels.size > 6:
        step = max(1, contour_levels.size // 6)
        contour_levels = contour_levels[::step]
    plot_data = _prepare_projected_field(field) if use_cartopy else field
    contour_kwargs = {
        "ax": axis,
        "linewidths": 0.68,
        "alpha": 0.62,
        "add_colorbar": False,
        "zorder": 3,
    }
    if x is not None:
        contour_kwargs["x"] = x
    if y is not None:
        contour_kwargs["y"] = y
    if use_cartopy and data_crs is not None:
        contour_kwargs["transform"] = data_crs

    negative_levels = contour_levels[contour_levels < 0.0]
    positive_levels = contour_levels[contour_levels > 0.0]
    if negative_levels.size:
        plot_data.plot.contour(
            levels=negative_levels,
            colors="#334155",
            linestyles="--",
            **contour_kwargs,
        )
    if positive_levels.size:
        plot_data.plot.contour(
            levels=positive_levels,
            colors="#334155",
            linestyles="-",
            **contour_kwargs,
        )

    minimum = float(np.nanmin(plot_data.values))
    maximum = float(np.nanmax(plot_data.values))
    if np.isfinite(minimum) and np.isfinite(maximum) and minimum < 0.0 < maximum:
        plot_data.plot.contour(
            levels=[0.0],
            colors="#0f172a",
            linewidths=0.75,
            linestyles="-",
            alpha=0.75,
            add_colorbar=False,
            zorder=3,
            **({"ax": axis, "transform": data_crs} if use_cartopy and data_crs is not None else {"ax": axis}),
        )
