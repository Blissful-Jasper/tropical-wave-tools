"""Optional cartopy support and map helpers (projection, guides, extents, longitude windows)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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


def _format_longitude_label(value: float) -> str:
    """Format 0-360 longitudes into compact E/W labels for Hovmoller axes."""
    lon = float(value) % 360.0
    if np.isclose(lon, 0.0) or np.isclose(lon, 360.0):
        return "0"
    if np.isclose(lon, 180.0):
        return "180"
    if lon < 180.0:
        return f"{round(lon)}E"
    return f"{round(360.0 - lon)}W"


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


def _case07_focus_half_width(wave_name: str | None) -> float:
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
