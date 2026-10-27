"""Wave-aware EOF analysis and regression of fields onto the principal components."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tropical_wave_tools.atlas._config import (
    CASE07_WAVE_LAT_RANGES,
    CASE07_WAVE_PROJECTIONS,
)
from tropical_wave_tools.atlas.events import (
    _lat_range,
)
from tropical_wave_tools.eof import EOFAnalyzer
from tropical_wave_tools.stats import linear_regression


def _case07_component_field(data: xr.DataArray, *, projection: str) -> xr.DataArray:
    """Return a full-field equatorial component for wave-aware EOF analysis."""
    if projection == "full" or "lat" not in data.dims:
        component = data.copy()
    else:
        mirrored = data.interp(lat=(-data["lat"]).values)
        mirrored = mirrored.assign_coords(lat=data["lat"])
        if projection == "symmetric":
            component = 0.5 * (data + mirrored)
        elif projection == "antisymmetric":
            component = 0.5 * (data - mirrored)
        else:
            raise ValueError("`projection` must be one of {'full', 'symmetric', 'antisymmetric'}.")
    component = component.rename(data.name or "wave_component")
    component.attrs.update(data.attrs)
    component.attrs["case07_projection"] = projection
    return component


def _case07_latitude_weights(data: xr.DataArray) -> xr.DataArray:
    """Return sqrt(cos(lat)) weights for latitude-aware EOF fitting."""
    lat_values = np.asarray(data["lat"].values, dtype=float)
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat_values)), 1.0e-6, None))
    return xr.DataArray(weights, dims=("lat",), coords={"lat": data["lat"]}, name="sqrt_coslat")


def _normalize_eof_signs(eofs: xr.DataArray, pcs: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Flip each mode so that its largest-magnitude loading is positive (EOF and PC flip together).

    The sign of an EOF/PC pair is arbitrary and depends on the SVD routine, so the same data could
    give mirrored figures on different machines. The pair describes the same variability either way.
    """
    signs = []
    for mode in eofs["mode"].values:
        loading = np.asarray(eofs.sel(mode=mode).values, dtype=float).ravel()
        index = int(np.nanargmax(np.abs(loading))) if np.isfinite(loading).any() else 0
        signs.append(-1.0 if loading[index] < 0.0 else 1.0)
    sign = xr.DataArray(signs, dims="mode", coords={"mode": eofs["mode"]})
    return eofs * sign, pcs * sign.sel(mode=pcs["mode"])


def compute_wave_eof(
    data: xr.DataArray,
    *,
    wave_name: str | None = None,
    n_modes: int = 2,
) -> tuple[xr.DataArray, xr.DataArray, np.ndarray, xr.DataArray]:
    """Compute wave-aware EOFs and return the analyzed OLR field used for fitting.

    The field is restricted to the wave's latitude band, projected onto its symmetric or
    antisymmetric component (or kept whole), weighted by ``sqrt(cos(lat))`` for the decomposition,
    and the patterns are divided by the weights again so they are in physical units.
    ``explained_variance`` refers to the weighted, projected data. The sign of each mode is fixed so
    that its largest-magnitude loading is positive.
    """
    wave_key = wave_name.lower() if wave_name is not None else None
    lat_range = CASE07_WAVE_LAT_RANGES.get(wave_key)
    projection = CASE07_WAVE_PROJECTIONS.get(wave_key, "full")
    analysis_field = data
    if lat_range is not None and "lat" in analysis_field.dims:
        analysis_field = _lat_range(analysis_field, *lat_range)
    analysis_field = _case07_component_field(analysis_field, projection=projection)

    eof_input = analysis_field
    if "lat" in eof_input.dims:
        lat_weights = _case07_latitude_weights(eof_input)
        eof_input = eof_input * lat_weights
    else:
        lat_weights = None

    analyzer = EOFAnalyzer(method="svd")
    results = analyzer.fit(eof_input, n_modes=n_modes, n_harmonics=0)
    eofs = results["eofs"]
    if lat_weights is not None and {"mode", "lat", "lon"}.issubset(eofs.dims):
        eofs = eofs / lat_weights
    pcs = results["pc_scores"]
    eofs, pcs = _normalize_eof_signs(eofs, pcs)
    explained = np.asarray(results["explained_variance"], dtype=float)
    analysis_field.attrs["case07_projection"] = projection
    if wave_key is not None:
        analysis_field.attrs["wave_name"] = wave_key
    if lat_range is not None:
        analysis_field.attrs["case07_lat_range"] = lat_range
    return eofs, pcs, explained, analysis_field


def regress_field_onto_pcs(
    field: xr.DataArray,
    pcs: xr.DataArray,
    *,
    standardize_pc: bool = False,
) -> xr.DataArray:
    """Regress one field onto each EOF principal component.

    The slope is computed over the times where both the field (at each grid point) and the PC are
    finite. With ``standardize_pc=True`` the units are field units per standard deviation of the PC.
    """
    time_dim = next((dim for dim in pcs.dims if dim != "mode"), None)
    if time_dim is None:
        raise ValueError("`pcs` must contain one non-mode dimension.")

    patterns: list[xr.DataArray] = []
    for mode in pcs["mode"].values:
        pc = pcs.sel(mode=mode)
        if standardize_pc:
            pc_std = float(pc.std(time_dim).item())
            if np.isfinite(pc_std) and pc_std > 0.0:
                pc = (pc - pc.mean(time_dim)) / pc_std
        slope = linear_regression(pc, field, dim=time_dim)["slope"]
        patterns.append(slope.expand_dims(mode=[mode]))

    return xr.concat(patterns, dim="mode")
