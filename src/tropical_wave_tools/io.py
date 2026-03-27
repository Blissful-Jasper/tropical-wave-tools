"""Input, coordinate normalization, and file I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import xarray as xr

from tropical_wave_tools.exceptions import InvalidDataArrayError


PathLike = Union[str, Path]
XarrayLike = Union[xr.Dataset, xr.DataArray]
STANDARD_DIM_CANDIDATES = {
    "time": ("time", "valid_time", "date", "datetime", "time_counter"),
    "lat": ("lat", "latitude", "nav_lat", "y"),
    "lon": ("lon", "longitude", "nav_lon", "x"),
}


def _find_standard_name(names: Iterable[str], standard_name: str) -> Optional[str]:
    candidates = STANDARD_DIM_CANDIDATES[standard_name]
    lowered = {name.lower(): name for name in names}

    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    for original_name in names:
        lowered_name = original_name.lower()
        for candidate in candidates:
            if candidate in lowered_name:
                return original_name
    return None


def rename_standard_coordinates(data: XarrayLike) -> XarrayLike:
    """Rename common climate-analysis coordinate aliases to ``time/lat/lon``."""
    rename_map: Dict[str, str] = {}
    known_names = list(data.dims) + list(data.coords)

    for standard_name in ("time", "lat", "lon"):
        current_name = _find_standard_name(known_names, standard_name)
        if current_name and current_name != standard_name:
            rename_map[current_name] = standard_name

    if rename_map:
        data = data.rename(rename_map)
    return data


def infer_variable_name(dataset: xr.Dataset) -> str:
    """Infer the main scientific variable from a dataset."""
    candidates = [name for name in dataset.data_vars if name.lower() not in {"info", "nmiss"}]
    if not candidates:
        raise InvalidDataArrayError("No scientific data variable could be inferred from the dataset.")
    return candidates[0]


def to_dataarray(data: XarrayLike, variable: Optional[str] = None) -> xr.DataArray:
    """Convert a Dataset/DataArray input into a DataArray."""
    standardized = rename_standard_coordinates(data)
    if isinstance(standardized, xr.DataArray):
        return standardized
    variable_name = variable or infer_variable_name(standardized)
    return standardized[variable_name]


def ensure_time_lat_lon(data: xr.DataArray) -> xr.DataArray:
    """Validate and reorder a DataArray to ``(time, lat, lon)``."""
    standardized = rename_standard_coordinates(data)
    required = ("time", "lat", "lon")
    missing = [dim for dim in required if dim not in standardized.dims]
    if missing:
        raise InvalidDataArrayError(
            f"Input data must contain dimensions {required}; missing {missing!r}."
        )
    return standardized.transpose("time", "lat", "lon")


def normalize_longitude(
    data: XarrayLike,
    *,
    target: str = "0_360",
) -> XarrayLike:
    """Normalize longitude to either ``0_360`` or ``-180_180`` and sort it."""
    standardized = rename_standard_coordinates(data)
    if "lon" not in standardized.coords:
        return standardized

    lon = standardized["lon"]
    if target == "0_360":
        new_lon = lon % 360.0
    elif target == "-180_180":
        new_lon = ((lon + 180.0) % 360.0) - 180.0
    elif target == "preserve":
        new_lon = lon
    else:
        raise ValueError("`target` must be one of {'0_360', '-180_180', 'preserve'}.")

    standardized = standardized.assign_coords(lon=new_lon)
    return standardized.sortby("lon")


def sort_latitude(data: XarrayLike, *, ascending: bool = True) -> XarrayLike:
    """Sort latitude ascending or descending if it is present."""
    standardized = rename_standard_coordinates(data)
    if "lat" not in standardized.coords:
        return standardized
    if ascending:
        return standardized.sortby("lat")
    return standardized.sortby("lat", ascending=False)


def standardize_data(
    data: XarrayLike,
    *,
    variable: Optional[str] = None,
    lon_target: str = "0_360",
    lat_ascending: bool = True,
) -> xr.DataArray:
    """Standardize Dataset/DataArray inputs to a consistent analysis-ready DataArray."""
    array = to_dataarray(data, variable=variable)
    array = ensure_time_lat_lon(array)
    array = normalize_longitude(array, target=lon_target)
    array = sort_latitude(array, ascending=lat_ascending)
    return ensure_time_lat_lon(array)


def _normalize_slice(bounds: Optional[Tuple[float, float]]) -> Optional[slice]:
    if bounds is None:
        return None
    lower, upper = bounds
    if lower <= upper:
        return slice(lower, upper)
    return slice(upper, lower)


def load_dataset(
    path: PathLike,
    *,
    chunks: Optional[Dict[str, int]] = None,
    lon_target: str = "0_360",
    lat_ascending: bool = True,
) -> xr.Dataset:
    """Load a dataset and normalize its coordinate naming and ordering."""
    dataset = xr.open_dataset(path, chunks=chunks)
    dataset = rename_standard_coordinates(dataset)
    dataset = normalize_longitude(dataset, target=lon_target)
    dataset = sort_latitude(dataset, ascending=lat_ascending)
    return dataset


def load_dataarray(
    path: PathLike,
    variable: Optional[str] = None,
    *,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    time_range: Optional[Tuple[str, str]] = None,
    chunks: Optional[Dict[str, int]] = None,
    sort_lat: bool = True,
    lon_target: str = "0_360",
) -> xr.DataArray:
    """
    Load a NetCDF variable as a standardized ``xarray.DataArray``.

    This helper normalizes dimension names such as ``valid_time`` or
    ``longitude``, sorts latitude to ascending order, and normalizes longitude.
    """
    dataset = load_dataset(path, chunks=chunks, lon_target=lon_target, lat_ascending=sort_lat)
    data = to_dataarray(dataset, variable=variable)

    lat_slice = _normalize_slice(lat_range)
    if lat_slice is not None and "lat" in data.coords:
        data = data.sel(lat=lat_slice)

    if lon_range is not None and "lon" in data.coords:
        lon_min, lon_max = lon_range
        if lon_min <= lon_max:
            data = data.sel(lon=slice(lon_min, lon_max))
        else:
            left = data.sel(lon=slice(lon_min, 360.0))
            right = data.sel(lon=slice(0.0, lon_max))
            data = xr.concat([left, right], dim="lon")

    if time_range is not None:
        data = data.sel(time=slice(*time_range))

    return ensure_time_lat_lon(data)


def describe_dataarray(data: xr.DataArray) -> Dict[str, object]:
    """Return a compact metadata summary for logging or docs."""
    summary: Dict[str, object] = {
        "name": data.name,
        "shape": tuple(int(value) for value in data.shape),
        "dims": tuple(str(dim) for dim in data.dims),
    }
    if "time" in data.coords:
        summary["time_start"] = str(data.time.values[0])
        summary["time_end"] = str(data.time.values[-1])
    if "lat" in data.coords:
        summary["lat_min"] = float(data.lat.min().item())
        summary["lat_max"] = float(data.lat.max().item())
    if "lon" in data.coords:
        summary["lon_min"] = float(data.lon.min().item())
        summary["lon_max"] = float(data.lon.max().item())
    return summary


def save_dataarray(data: xr.DataArray, path: PathLike) -> Path:
    """Save a DataArray and return the resolved output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(output_path)
    return output_path


def save_dataset(dataset: xr.Dataset, path: PathLike) -> Path:
    """Save a Dataset and return the resolved output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    return output_path


def describe_many(dataarrays: Iterable[xr.DataArray]) -> list[Dict[str, object]]:
    """Summarize multiple data arrays."""
    return [describe_dataarray(data) for data in dataarrays]
