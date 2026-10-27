"""Colormaps, matplotlib style, shared level/label helpers and colorbar utilities."""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

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



def get_publication_cmap(name: str) -> str | LinearSegmentedColormap:
    """Return one of the package publication colormaps."""
    return PUBLICATION_CMAPS.get(name, name)



@contextmanager
def scientific_plot_style(overrides: dict[str, object] | None = None):
    """Apply a consistent research-figure style inside a plotting context."""
    style = dict(SCIENTIFIC_MPL_STYLE)
    if overrides:
        style.update(overrides)
    with plt.rc_context(style):
        yield



def use_scientific_style(overrides: dict[str, object] | None = None) -> None:
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



def _finite_abs_max(data: object) -> float:
    """Return the finite absolute maximum of a field."""
    values = np.asarray(getattr(data, "values", data), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(finite)))



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
    minimum_vmax: float | None = None,
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
    vmax = _finite_abs_max(data) if quantile >= 1.0 else _symmetric_limit(data, quantile=quantile)
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
    ticks: np.ndarray | None = None,
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
    ticks: np.ndarray | None = None,
    integer: bool = False,
    pad: float | None = None,
    shrink: float | None = None,
    aspect: float | None = None,
    fraction: float | None = None,
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



def _wave_title_name(wave_name: str | None) -> str:
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



def save_figure(figure: plt.Figure, path: str | Path, *, dpi: int = 200) -> Path:
    """Save a figure and return its output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path
