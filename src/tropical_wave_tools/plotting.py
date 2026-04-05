"""Plotting utilities for spectral and filter outputs."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from tropical_wave_tools.config import DEFAULT_WAVE_SPECS
from tropical_wave_tools.matsuno import matsuno_modes_wk
from tropical_wave_tools.spectral import WKAnalysisResult

try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
except ImportError:  # pragma: no cover - optional dependency
    cartopy = None
    ccrs = None
    cfeature = None
    add_cyclic_point = None


PUBLICATION_CMAPS: dict[str, LinearSegmentedColormap] = {
    "wk_power": LinearSegmentedColormap.from_list(
        "twave_wk_power",
        ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#ef6548", "#b30000", "#4a0c12"],
        N=256,
    ),
    "wave_diverging": LinearSegmentedColormap.from_list(
        "twave_wave_diverging",
        ["#8c510a", "#d8b365", "#f6e8c3", "#f5f5f5", "#c7eae5", "#5ab4ac", "#01665e"],
        N=256,
    ),
    "olr_diverging": LinearSegmentedColormap.from_list(
        "twave_olr_diverging",
        [
            (0.00, "#1e4c8c"),
            (0.18, "#4c85c3"),
            (0.36, "#d7e7f5"),
            (0.46, "#f7fbff"),
            (0.50, "#ffffff"),
            (0.54, "#fffdfc"),
            (0.64, "#fce7db"),
            (0.82, "#ec8460"),
            (1.00, "#b51d30"),
        ],
        N=256,
    ),
    "paper_hovmoller_diverging": LinearSegmentedColormap.from_list(
        "twave_paper_hovmoller_diverging",
        [
            (0.00, "#133b76"),
            (0.14, "#2e67a6"),
            (0.30, "#6f9fce"),
            (0.43, "#dbe7f1"),
            (0.49, "#f7fbff"),
            (0.50, "#ffffff"),
            (0.51, "#fffdfa"),
            (0.58, "#f7e8de"),
            (0.72, "#e7b694"),
            (0.86, "#d26e4e"),
            (1.00, "#9f1d2f"),
        ],
        N=256,
    ),
    "wave_std": LinearSegmentedColormap.from_list(
        "twave_wave_std",
        ["#f8fbff", "#d7eff5", "#94d2bd", "#2a9d8f", "#126782", "#0b3954"],
        N=256,
    ),
    "wave_std_red": LinearSegmentedColormap.from_list(
        "twave_wave_std_red",
        ["#fff8f2", "#ffe5d6", "#ffc2a1", "#ff9575", "#f86752", "#ef4444", "#d7263d"],
        N=256,
    ),
    "wave_activity": LinearSegmentedColormap.from_list(
        "twave_wave_activity",
        [
            "#fbfdff",
            "#e7f0f6",
            "#b7d8d5",
            "#7fb9b2",
            "#e9d07f",
            "#f2a65a",
            "#e1644a",
            "#b11f3a",
        ],
        N=256,
    ),
    "olr_mean": LinearSegmentedColormap.from_list(
        "twave_olr_mean",
        [
            "#2b3a67",
            "#4f6d8c",
            "#87a7b3",
            "#e5eef2",
            "#fff9e8",
            "#f9c77b",
            "#e88b4a",
            "#bf4e36",
            "#7f1d1d",
        ],
        N=256,
    ),
    "variability_std": LinearSegmentedColormap.from_list(
        "twave_variability_std",
        [
            "#fcfbfd",
            "#efe6f5",
            "#dcb7d8",
            "#c57bb4",
            "#dd4f8c",
            "#ef6a62",
            "#f6a15a",
            "#fff1a8",
        ],
        N=256,
    ),
    "precip_clim": LinearSegmentedColormap.from_list(
        "twave_precip_clim",
        [
            "#fbfdff",
            "#dceef5",
            "#a9d6d5",
            "#77bfa3",
            "#4f9d8f",
            "#2d6f8e",
            "#1f3f76",
        ],
        N=256,
    ),
    "wind_std": LinearSegmentedColormap.from_list(
        "twave_wind_std",
        ["#fffdf7", "#f6e7c1", "#e6be7b", "#c98534", "#8f4e1e", "#4d2d18"],
        N=256,
    ),
    "seasonal": LinearSegmentedColormap.from_list(
        "twave_seasonal",
        ["#fffaf0", "#ffe8a3", "#f4b942", "#e07a5f", "#8c5e58", "#355070"],
        N=256,
    ),
}

SCIENTIFIC_MPL_STYLE: dict[str, object] = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfcff",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#0f172a",
    "axes.titlecolor": "#0f172a",
    "axes.titlesize": 12.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10.0,
    "axes.grid": False,
    "grid.color": "#cbd5e1",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.55,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "font.family": ["DejaVu Sans"],
    "mathtext.fontset": "stix",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "legend.frameon": True,
    "legend.framealpha": 0.94,
}


def get_publication_cmap(name: str) -> Union[str, LinearSegmentedColormap]:
    """Return one of the package publication colormaps."""
    return PUBLICATION_CMAPS.get(name, name)


@contextmanager
def scientific_plot_style(overrides: Optional[dict[str, object]] = None):
    """Apply a consistent research-figure style inside a plotting context."""
    style = dict(SCIENTIFIC_MPL_STYLE)
    if overrides:
        style.update(overrides)
    with plt.rc_context(style):
        yield


def use_scientific_style(overrides: Optional[dict[str, object]] = None) -> None:
    """Apply the package plotting style globally for scripts or notebooks."""
    style = dict(SCIENTIFIC_MPL_STYLE)
    if overrides:
        style.update(overrides)
    plt.rcParams.update(style)


def _apply_axes_style(ax: plt.Axes, *, grid: bool = False) -> None:
    """Apply lightweight axis styling shared by all figure templates."""
    ax.set_facecolor("#fbfcff")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#334155", labelsize=9)
    if grid:
        ax.grid(True, color="#cbd5e1", alpha=0.35, linewidth=0.55)


def _symmetric_limit(data: xr.DataArray, *, quantile: float = 0.98) -> float:
    """Return a robust symmetric range for anomaly-style plots."""
    limit = float(np.abs(data).quantile(quantile).item())
    if not np.isfinite(limit) or limit == 0.0:
        limit = float(np.nanmax(np.abs(data.values)))
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0
    return limit


def _std_label(data: xr.DataArray) -> str:
    """Build a compact standard-deviation label."""
    units = str(data.attrs.get("units", "")).strip()
    units = _format_units_for_mathtext(units)
    return f"STD ({units})" if units else "STD"


def _field_label(prefix: str, data: xr.DataArray) -> str:
    """Build a compact field label with optional units."""
    units = str(data.attrs.get("units", "")).strip()
    units = _format_units_for_mathtext(units)
    return f"{prefix} ({units})" if units else prefix


def _format_units_for_mathtext(units: str) -> str:
    """Render simple power syntax like s^-1 using Matplotlib mathtext superscripts."""
    if not units:
        return units
    return re.sub(r"\^(-?\d+)", lambda match: f"$^{{{match.group(1)}}}$", units)


def _filled_levels(vmin: float, vmax: float, *, count: int = 19) -> np.ndarray:
    """Create stable contourf levels for publication-style filled plots."""
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if np.isclose(vmin, vmax):
        span = abs(vmax) if vmax != 0.0 else 1.0
        vmin -= span
        vmax += span
    return np.linspace(vmin, vmax, count)


def _format_longitude_label(value: float) -> str:
    """Format 0-360 longitudes into compact E/W labels for Hovmoller axes."""
    lon = float(value) % 360.0
    if np.isclose(lon, 0.0) or np.isclose(lon, 360.0):
        return "0"
    if np.isclose(lon, 180.0):
        return "180"
    if lon < 180.0:
        return f"{int(round(lon))}E"
    return f"{int(round(360.0 - lon))}W"


def _nice_step(span: float, *, target_steps: int = 8, integer: bool = False) -> float:
    """Return a rounded colorbar step size."""
    if not np.isfinite(span) or span <= 0.0:
        return 1.0

    raw_step = span / max(target_steps, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    normalized = raw_step / magnitude
    candidates = (1.0, 2.0, 3.0, 5.0, 10.0) if integer else (1.0, 2.0, 2.5, 5.0, 10.0)
    step = next(candidate for candidate in candidates if normalized <= candidate) * magnitude
    if integer and step < 1.0:
        step = 1.0
    return float(step)


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
    quiver_scale: Optional[float],
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


def _finite_abs_max(data: object) -> float:
    """Return the finite absolute maximum of a field."""
    values = np.asarray(getattr(data, "values", data), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(finite)))


def _prefer_zonal_wind_contours(zonal_wind: xr.DataArray, meridional_wind: xr.DataArray) -> bool:
    """Prefer zonal-wind contours when meridional wind is unresolved."""
    zonal_peak = _finite_abs_max(zonal_wind)
    meridional_peak = _finite_abs_max(meridional_wind)
    if zonal_peak <= 0.0:
        return False
    return meridional_peak < max(0.05, 0.15 * zonal_peak)


def _signed_levels_from_data(data: object, *, target_steps: int = 6) -> np.ndarray:
    """Build symmetric signed contour levels excluding zero."""
    vmax = _finite_abs_max(data)
    if not np.isfinite(vmax) or vmax <= 0.0:
        return np.array([-1.0, 1.0])
    step = _nice_step(2.0 * vmax, target_steps=target_steps, integer=False)
    vmax = max(step, step * np.ceil(vmax / step))
    levels = np.arange(-vmax, vmax + 0.5 * step, step)
    levels = levels[np.abs(levels) > 0.5 * step]
    if levels.size == 0:
        levels = np.array([-vmax, vmax])
    return levels


def _plot_zonal_wind_contours(
    axis: plt.Axes,
    zonal_wind: xr.DataArray,
    *,
    levels: Optional[np.ndarray] = None,
    use_cartopy: bool = False,
    data_crs: Optional[object] = None,
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
    levels: Optional[np.ndarray] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    use_cartopy: bool = False,
    data_crs: Optional[object] = None,
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


def _normalize_cycle_values(
    values: np.ndarray,
    *,
    method: str = "annual_mean",
) -> tuple[np.ndarray, float]:
    """Normalize a positive seasonal cycle for cross-variable comparison."""
    series = np.asarray(values, dtype=float)
    finite = np.isfinite(series)
    if not finite.any():
        return series.copy(), np.nan

    valid = series[finite]
    if method == "annual_mean":
        denominator = float(np.nanmean(valid))
    elif method == "max":
        denominator = float(np.nanmax(valid))
    else:
        raise ValueError("`method` must be one of {'annual_mean', 'max'}.")

    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return series.copy(), np.nan
    return series / denominator, denominator


def _integer_levels_from_data(
    data: object,
    *,
    symmetric: bool,
    target_steps: int = 8,
    zero_floor: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build evenly spaced integer contour levels and readable ticks."""
    if isinstance(data, (list, tuple)):
        flattened = [np.asarray(getattr(item, "values", item), dtype=float).ravel() for item in data]
        values = np.concatenate(flattened) if flattened else np.array([], dtype=float)
    else:
        values = np.asarray(getattr(data, "values", data), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        default_levels = np.array([-1.0, 0.0, 1.0]) if symmetric else np.array([0.0, 1.0, 2.0])
        return default_levels, default_levels

    if symmetric:
        vmax = float(np.nanmax(np.abs(finite)))
        step = _nice_step(2.0 * vmax, target_steps=target_steps, integer=True)
        vmax = max(step, step * np.ceil(vmax / step))
        levels = np.arange(-vmax, vmax + 0.5 * step, step)
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        step = _nice_step(vmax - vmin, target_steps=target_steps, integer=True)
        lower = 0.0 if zero_floor else step * np.floor(vmin / step)
        upper = step * np.ceil(vmax / step)
        if np.isclose(lower, upper):
            upper = lower + step
        levels = np.arange(lower, upper + 0.5 * step, step)

    tick_stride = max(1, int(np.ceil((len(levels) - 1) / 8)))
    ticks = levels[::tick_stride]
    if ticks.size == 0 or not np.isclose(ticks[-1], levels[-1]):
        ticks = np.append(ticks, levels[-1])
    return levels, ticks


def _continuous_symmetric_levels_from_data(
    data: object,
    *,
    quantile: float = 0.98,
    count: int = 19,
    target_tick_steps: int = 8,
    range_scale: float = 1.0,
    minimum_vmax: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build symmetric continuous contour levels with evenly spaced endpoint ticks."""
    vmax = _symmetric_limit(data, quantile=quantile)
    vmax *= max(float(range_scale), 1.0e-6)
    if minimum_vmax is not None and np.isfinite(minimum_vmax) and minimum_vmax > 0.0:
        vmax = max(vmax, float(minimum_vmax))
    tick_step = _nice_step(2.0 * vmax, target_steps=target_tick_steps, integer=False)
    vmax = max(tick_step, tick_step * np.ceil(vmax / tick_step))
    levels = _filled_levels(-vmax, vmax, count=count)
    ticks = np.arange(-vmax, vmax + 0.5 * tick_step, tick_step)
    return levels, ticks, float(vmax)


def _scientific_integer_levels_from_data(
    data: object,
    *,
    quantile: float = 1.0,
    target_tick_steps: int = 8,
    level_count: int = 19,
    pad_fraction: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build symmetric continuous levels with integer ticks under a scientific-scale factor."""
    if quantile >= 1.0:
        vmax = _finite_abs_max(data)
    else:
        vmax = _symmetric_limit(data, quantile=quantile)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    vmax *= max(1.0 + float(pad_fraction), 1.0)
    exponent = int(np.floor(np.log10(vmax)))
    scale = 10.0**exponent
    scaled_vmax = vmax / scale
    step_scaled = _nice_step(2.0 * scaled_vmax, target_steps=target_tick_steps, integer=True)
    scaled_bound = max(step_scaled, step_scaled * np.ceil(scaled_vmax / step_scaled))
    bound = float(scaled_bound * scale)
    levels = _filled_levels(-bound, bound, count=level_count)
    ticks = np.arange(-scaled_bound, scaled_bound + 0.5 * step_scaled, step_scaled) * scale
    return levels, ticks, bound


def _style_colorbar(
    cbar: object,
    *,
    ticks: Optional[np.ndarray] = None,
    integer: bool = False,
    scientific_integer: bool = False,
) -> None:
    """Apply a compact publication style to a colorbar."""
    if ticks is not None:
        cbar.set_ticks(ticks)
        if integer:
            cbar.set_ticklabels([f"{tick:.0f}" for tick in ticks])
    if scientific_integer:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        formatter.set_useOffset(False)
        cbar.formatter = formatter
        cbar.update_ticks()
    cbar.ax.tick_params(labelsize=8.8, colors="#334155")
    cbar.outline.set_edgecolor("#cbd5e1")
    cbar.outline.set_linewidth(0.75)


def _add_shared_colorbar(
    figure: plt.Figure,
    image: object,
    axes: Sequence[plt.Axes],
    *,
    label: str,
    orientation: str = "vertical",
    ticks: Optional[np.ndarray] = None,
    integer: bool = False,
    pad: Optional[float] = None,
    shrink: Optional[float] = None,
    aspect: Optional[float] = None,
    fraction: Optional[float] = None,
):
    """Attach one publication-style colorbar shared by a set of axes."""
    panel_axes = list(axes)
    if not panel_axes:
        raise ValueError("At least one axis is required for a shared colorbar.")

    if orientation == "vertical":
        cbar = figure.colorbar(
            image,
            ax=panel_axes,
            orientation=orientation,
            pad=0.018 if pad is None else pad,
            shrink=0.96 if shrink is None else shrink,
            aspect=28 if aspect is None else aspect,
            fraction=0.05 if fraction is None else fraction,
        )
    else:
        cbar = figure.colorbar(
            image,
            ax=panel_axes,
            orientation=orientation,
            pad=0.055 if pad is None else pad,
            shrink=0.88 if shrink is None else shrink,
            aspect=40 if aspect is None else aspect,
            fraction=0.065 if fraction is None else fraction,
        )

    cbar.set_label(label)
    _style_colorbar(cbar, ticks=ticks, integer=integer)
    return cbar


def _is_weak_signal(data: object, *, threshold: float = 0.1) -> bool:
    """Return whether a field or series is effectively zero at plotting scale."""
    values = getattr(data, "values", data)
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return True
    peak = float(np.nanmax(np.abs(array[finite])))
    return (not np.isfinite(peak)) or peak < threshold


def _default_map_projection() -> tuple[object, object]:
    """Return the display and data CRS for lat-lon diagnostics."""
    if ccrs is None:  # pragma: no cover - optional dependency
        raise ImportError("cartopy is required for projected plotting.")
    return ccrs.PlateCarree(central_longitude=180.0), ccrs.PlateCarree()


def _has_cartopy_feature(name: str, *, category: str = "physical", scale: str = "110m") -> bool:
    """Return whether a Natural Earth feature is already available locally."""
    if cartopy is None:
        return False
    data_dir = Path(str(cartopy.config.get("data_dir", "")))
    if not data_dir:
        return False
    feature_path = data_dir / "shapefiles" / "natural_earth" / category / f"ne_{scale}_{name}.shp"
    return feature_path.exists()


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    """Annotate one panel with a small figure label."""
    ax.text(
        0.015,
        0.98,
        f"{label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="semibold",
        color="#0f172a",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        zorder=12,
    )


def _add_map_guides(
    ax: plt.Axes,
    *,
    draw_labels: bool = False,
    label_left: bool = True,
    label_bottom: bool = True,
) -> None:
    """Add coastlines, land shading, and optional lat/lon labels to a projected map."""
    if cfeature is not None and _has_cartopy_feature("land"):
        ax.add_feature(cfeature.LAND, facecolor="#eef2f7", edgecolor="none", zorder=0)
    if cfeature is not None and _has_cartopy_feature("ocean"):
        ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    if _has_cartopy_feature("coastline"):
        ax.coastlines(linewidth=0.75, color="#334155")
    gridlines = ax.gridlines(
        draw_labels=draw_labels,
        linewidth=0.45,
        color="#94a3b8",
        alpha=0.28,
        linestyle="--",
    )
    gridlines.xlocator = plt.MaxNLocator(7)
    gridlines.ylocator = plt.MaxNLocator(5)
    if draw_labels:
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.left_labels = label_left
        gridlines.bottom_labels = label_bottom
        gridlines.xlabel_style = {"size": 8.4, "color": "#475569"}
        gridlines.ylabel_style = {"size": 8.4, "color": "#475569"}


def _prepare_projected_field(data: xr.DataArray) -> xr.DataArray:
    """Add a cyclic point when the longitude span is effectively global."""
    if add_cyclic_point is None or "lon" not in data.coords:
        return data

    lon = np.asarray(data["lon"].values, dtype=float)
    if lon.size < 2:
        return data
    if float(lon.max() - lon.min()) < 300.0:
        return data

    cyclic_values, cyclic_lon = add_cyclic_point(data.values, coord=lon, axis=data.get_axis_num("lon"))
    return xr.DataArray(
        cyclic_values,
        dims=data.dims,
        coords={**{dim: data.coords[dim] for dim in data.dims if dim != "lon"}, "lon": cyclic_lon},
        attrs=data.attrs,
        name=data.name,
    )


def _set_map_extent_from_field(ax: plt.Axes, data: xr.DataArray, data_crs: object) -> None:
    """Apply a robust map extent, avoiding Cartopy failures on global pole-to-pole data."""
    if not {"lon", "lat"}.issubset(data.coords):
        return

    lon_values = np.asarray(data["lon"].values, dtype=float)
    lat_values = np.asarray(data["lat"].values, dtype=float)
    if lon_values.size == 0 or lat_values.size == 0:
        return

    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    lat_min = float(np.nanmin(lat_values))
    lat_max = float(np.nanmax(lat_values))
    if not np.all(np.isfinite([lon_min, lon_max, lat_min, lat_max])):
        return

    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lat_clip = 89.999

    if lon_span >= 359.0 and lat_span >= 179.0:
        ax.set_global()
        return

    if lon_span >= 359.0:
        lon_min, lon_max = -179.999, 179.999
    lat_min = max(lat_min, -lat_clip)
    lat_max = min(lat_max, lat_clip)
    if lat_min >= lat_max:
        return

    try:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
    except ValueError:
        if lon_span >= 359.0:
            ax.set_global()


def _wave_title_name(wave_name: Optional[str]) -> str:
    """Return a publication-friendly wave label."""
    if wave_name is None:
        return "Wave"
    mapping = {
        "kelvin": "Kelvin Wave",
        "er": "Equatorial Rossby Wave",
        "mjo": "MJO",
        "mrg": "Mixed Rossby-Gravity Wave",
        "td": "Tropical Disturbance",
    }
    return mapping.get(wave_name.lower(), wave_name.upper())


def _case07_focus_half_width(wave_name: Optional[str]) -> float:
    """Return a longitude half-width tuned for Case 07 focused maps."""
    widths = {
        "kelvin": 75.0,
        "er": 85.0,
        "mjo": 85.0,
        "mrg": 70.0,
        "td": 70.0,
    }
    if wave_name is None:
        return 90.0
    return widths.get(wave_name.lower(), 90.0)


def _focus_longitude_center(data: xr.DataArray) -> float:
    """Estimate the longitude center of the strongest signal."""
    if "lon" not in data.dims:
        return 180.0
    reduced = np.abs(data)
    for dim in list(reduced.dims):
        if dim != "lon":
            reduced = reduced.mean(dim)
    values = np.asarray(reduced.values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return 180.0
    index = int(np.nanargmax(values))
    return float(reduced["lon"].values[index])


def _subset_longitude_window(
    data: xr.DataArray,
    *,
    center_lon: float,
    half_width: float,
) -> xr.DataArray:
    """Subset a longitude window around a chosen signal center."""
    if "lon" not in data.dims:
        return data
    lon_values = np.asarray(data["lon"].values, dtype=float)
    if lon_values.size < 3:
        return data
    shifted = ((lon_values - center_lon + 180.0) % 360.0) - 180.0
    order = np.argsort(shifted)
    shifted_sorted = shifted[order]
    window = np.abs(shifted_sorted) <= float(half_width)
    if int(window.sum()) < 3:
        return data
    subset = data.isel(lon=order[window])
    subset = subset.assign_coords(lon=center_lon + shifted_sorted[window])
    return subset.sortby("lon")


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
    rename_map: Optional[dict[str, str]] = None,
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


def plot_time_series(
    data: xr.DataArray,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
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


def plot_latlon_field(
    data: xr.DataArray,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: Union[str, LinearSegmentedColormap] = "olr_diverging",
    levels: Optional[int] = None,
    use_cartopy: Optional[bool] = None,
    colorbar_label: Optional[str] = None,
    colorbar_orientation: str = "horizontal",
    integer_colorbar: bool = False,
    zero_floor_colorbar: bool = False,
    target_steps: int = 8,
    level_count: int = 19,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a latitude-longitude field with optional cartopy support."""
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        cmap = get_publication_cmap(str(cmap)) if isinstance(cmap, str) else cmap
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
                    if integer_colorbar
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
                    if integer_colorbar
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


def plot_hovmoller_comparison(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    left_title: str = "Raw anomaly",
    right_title: str = "Filtered signal",
    x: str = "lon",
    y: str = "time",
    cmap: Union[str, LinearSegmentedColormap] = "wave_diverging",
    quantile: float = 0.98,
    colorbar_label: str = "Amplitude",
    colorbar_orientation: str = "vertical",
    integer_colorbar: bool = False,
    colorbar_extend: str = "neither",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
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


def plot_wk_spectrum(
    result: WKAnalysisResult,
    *,
    max_wn: int = 15,
    max_freq: float = 0.5,
    add_matsuno_lines: bool = True,
    equivalent_depths: Sequence[float] = (12.0, 25.0, 50.0),
    kelvin_band_depths: Sequence[float] = (8.0, 25.0, 90.0),
    cpd_lines: Sequence[float] = (3.0, 6.0, 30.0),
    cmap: Union[str, LinearSegmentedColormap] = "wk_power",
    levels: Optional[np.ndarray] = None,
    colorbar_orientation: str = "vertical",
    annotate_filter_boxes: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
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


def plot_spatial_std_comparison(
    legacy_std: xr.DataArray,
    cckw_std: xr.DataArray,
    *,
    wave_name: str,
    title_suffix: str = "",
    use_cartopy: Optional[bool] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot legacy-vs-CCKW time-standard-deviation maps."""
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        diff = legacy_std - cckw_std
        vmax = float(np.nanmax([legacy_std.max().item(), cckw_std.max().item()]))
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


def plot_hovmoller_triptych(
    fields: Sequence[xr.DataArray],
    *,
    titles: Sequence[str],
    cmaps: Optional[Sequence[Union[str, LinearSegmentedColormap]]] = None,
    colorbar_labels: Optional[Sequence[str]] = None,
    x: str = "lon",
    y: str = "time",
    quantile: float = 0.98,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
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
    cmap: Union[str, LinearSegmentedColormap] = "paper_hovmoller_diverging",
    figsize: tuple[float, float] = (7.2, 5.0),
    shading_label: str = "Regressed convection proxy (W m$^{-2}$)",
    shading_quantile: float = 0.985,
    shading_level_count: int = 37,
    shading_range_scale: float = 1.0,
    shading_min_vmax: Optional[float] = None,
    contour_target_steps: int = 6,
    contour_quantile: float = 0.98,
    xticks: Optional[Sequence[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
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


def plot_spatial_std_triptych(
    olr_std: xr.DataArray,
    u_std: xr.DataArray,
    v_std: xr.DataArray,
    *,
    titles: Sequence[str] = ("Filtered OLR STD", "Filtered U850 STD", "Filtered V850 STD"),
    use_cartopy: Optional[bool] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot a projected OLR/U850/V850 standard-deviation triptych."""
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
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
    use_cartopy: Optional[bool] = None,
    ncols: int = 2,
    cmap: Union[str, LinearSegmentedColormap] = "wave_std",
    integer_colorbar: bool = False,
    target_steps: int = 8,
    level_count: int = 19,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot a grid comparing filtered spatial distributions across multiple waves."""
    if len(fields) != len(titles):
        raise ValueError("`fields` and `titles` must have the same length.")

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
                "extend": "neither",
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
            if _is_weak_signal(field, threshold=0.1):
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
        auxiliary_axis: Optional[plt.Axes] = None
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
    quiver_scale: Optional[float] = 55.0,
    integer_colorbar: bool = True,
    colorbar_extend: str = "neither",
    use_cartopy: Optional[bool] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot divergence and vorticity maps with wind vectors."""
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
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
                "extend": colorbar_extend,
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


def plot_eof_spatial_patterns_and_pcs(
    eofs: xr.DataArray,
    pcs: xr.DataArray,
    explained_variance: Sequence[float],
    *,
    modes: Sequence[int] = (1, 2),
    use_cartopy: Optional[bool] = None,
    cmap: Union[str, LinearSegmentedColormap] = "olr_diverging",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
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
    wave_name: Optional[str] = None,
    quiver_stride: int = 6,
    quiver_scale: Optional[float] = 55.0,
    use_cartopy: Optional[bool] = None,
    cmap: Union[str, LinearSegmentedColormap] = "olr_diverging",
    integer_colorbar: bool = True,
    field_label: str = "PC-regressed OLR (W m$^{-2}$)",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
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


def plot_lag_longitude_evolution(
    lagged_equatorial_field: xr.DataArray,
    *,
    title: str,
    cmap: Union[str, LinearSegmentedColormap] = "olr_diverging",
    integer_colorbar: bool = False,
    colorbar_extend: str = "neither",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
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


def plot_horizontal_structure(
    olr_field: xr.DataArray,
    u_field: xr.DataArray,
    v_field: xr.DataArray,
    *,
    title: Optional[str] = None,
    quiver_stride: int = 5,
    quiver_scale: Optional[float] = 55.0,
    colorbar_orientation: str = "horizontal",
    integer_colorbar: bool = False,
    colorbar_extend: str = "neither",
    wind_overlay: str = "auto",
    use_cartopy: Optional[bool] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot one horizontal structure map with OLR shading and wind vectors."""
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
        vmax = _symmetric_limit(olr_field)
        cmap = get_publication_cmap("olr_diverging")
        if integer_colorbar:
            contour_levels, colorbar_ticks = _integer_levels_from_data(olr_field, symmetric=True)
            vmax = float(max(abs(contour_levels[0]), abs(contour_levels[-1])))
        else:
            contour_levels = _filled_levels(-vmax, vmax)
            colorbar_ticks = None

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
                extend=colorbar_extend,
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
                extend=colorbar_extend,
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
    quiver_scale: Optional[float] = 55.0,
    use_cartopy: Optional[bool] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Compare lag-0 horizontal structures for multiple waves in one figure."""
    if not (len(olr_fields) == len(zonal_winds) == len(meridional_winds) == len(titles)):
        raise ValueError("All input sequences must have the same length.")

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
                    levels=_filled_levels(-vmax, vmax),
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
                )
                _set_map_extent_from_field(axis, plot_data, data_crs)
                _add_map_guides(axis, draw_labels=True, label_left=col == 0, label_bottom=row == nrows - 1)
            else:
                image = olr_field.plot.contourf(
                    ax=axis,
                    cmap=get_publication_cmap("olr_diverging"),
                    levels=_filled_levels(-vmax, vmax),
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    add_colorbar=False,
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
            if _is_weak_signal(olr_field, threshold=0.1):
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
        auxiliary_axis: Optional[plt.Axes] = None
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
    lags: Optional[Sequence[int]] = None,
    ncols: int = 3,
    quiver_stride: int = 6,
    quiver_scale: Optional[float] = 55.0,
    colorbar_orientation: str = "horizontal",
    integer_colorbar: bool = False,
    colorbar_extend: str = "neither",
    olr_quantile: float = 0.98,
    olr_level_count: int = 19,
    olr_range_scale: float = 1.0,
    olr_min_vmax: Optional[float] = None,
    wind_overlay: str = "auto",
    quiver_width: float = 0.0016,
    quiver_headwidth: float = 2.9,
    quiver_headlength: float = 3.7,
    quiver_headaxislength: float = 3.4,
    focus_longitude: bool = False,
    focus_center_lon: Optional[float] = None,
    focus_half_width: Optional[float] = None,
    suptitle: Optional[str] = None,
    panel_title_template: str = "Lag {lag:+d} d",
    use_cartopy: Optional[bool] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot lagged horizontal composites with OLR shading and low-level wind vectors."""
    with scientific_plot_style():
        use_cartopy = (ccrs is not None) if use_cartopy is None else use_cartopy
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
                    extend=colorbar_extend,
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
                    extend=colorbar_extend,
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
        auxiliary_axis: Optional[plt.Axes] = None
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
    quiver_scale: Optional[float] = 55.0,
    use_cartopy: Optional[bool] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Compare OLR-wind horizontal evolution across multiple waves and lags."""
    if not (
        len(olr_lagged_fields)
        == len(zonal_lagged_fields)
        == len(meridional_lagged_fields)
        == len(wave_names)
    ):
        raise ValueError("All input sequences must have the same length.")

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
                        levels=_filled_levels(-vmax, vmax),
                        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                        add_colorbar=False,
                    )
                    _set_map_extent_from_field(axis, plot_data, data_crs)
                    _add_map_guides(axis, draw_labels=True, label_left=col == 0, label_bottom=row == nrows - 1)
                else:
                    image = olr_panel.plot.contourf(
                        ax=axis,
                        cmap=get_publication_cmap("olr_diverging"),
                        levels=_filled_levels(-vmax, vmax),
                        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                        add_colorbar=False,
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
                if _is_weak_signal(olr_panel, threshold=0.1):
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


def plot_monthly_cycle(
    months: Sequence[int],
    series: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
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
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
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
    region_labels: Optional[dict[str, str]] = None,
    ylabel: str = "Variance contribution [%]",
    note_text: str = "Monthly mean ± 1σ\n20°S–20°N regional mean\nVariance fraction",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
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
    cmap: Union[str, LinearSegmentedColormap] = "seasonal",
    colorbar_label: str = "Projected monthly RMS",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
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


def plot_multiwave_eof_summary(
    eof_fields: Sequence[xr.DataArray],
    pc_series_list: Sequence[xr.DataArray],
    explained_variances: Sequence[float],
    *,
    wave_names: Sequence[str],
    use_cartopy: Optional[bool] = None,
    cmap: Union[str, LinearSegmentedColormap] = "olr_diverging",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
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


def plot_wave_monthly_cycle_comparison(
    monthly_cycles: dict[str, dict[str, np.ndarray]],
    *,
    wave_names: Sequence[str],
    ncols: int = 3,
    monthly_significance: Optional[dict[str, dict[str, xr.DataArray]]] = None,
    normalize_each_series: bool = False,
    normalization: str = "annual_mean",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot monthly RMS annual cycles for multiple waves."""
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
            if all(_is_weak_signal(values, threshold=0.1) for values in series.values()):
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
    cmap: Union[str, LinearSegmentedColormap] = "seasonal",
    colorbar_label: str = "Projected monthly RMS",
    significance_fields: Optional[Sequence[xr.DataArray]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot month-longitude seasonal evolution for multiple waves."""
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
        image = None

        for panel_index, (axis, wave_name, field) in enumerate(zip(axes.flat, wave_names, monthly_longitude_fields)):
            image = field.plot.contourf(
                ax=axis,
                x="lon",
                y="month",
                cmap=cmap,
                levels=_filled_levels(vmin, vmax),
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            if significance_fields is not None:
                significance = significance_fields[panel_index]
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
            if _is_weak_signal(field, threshold=0.1):
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
    trend_pvalues: Optional[dict[str, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot annual RMS time series with fitted linear trends for multiple waves."""
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
            if _is_weak_signal(values[valid] if valid.any() else values, threshold=0.1):
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
