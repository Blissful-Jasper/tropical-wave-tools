"""EOF analysis utilities ported from the original project."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tropical_wave_tools.io import normalize_longitude, rename_standard_coordinates
from tropical_wave_tools.preprocessing import extract_low_harmonics

try:  # pragma: no cover - optional dependency
    from global_land_mask import globe

    _LAND_MASK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _LAND_MASK_AVAILABLE = False
    globe = None


class EOFAnalyzer:
    """EOF analyzer with SVD and optional xeofs backends."""

    def __init__(
        self,
        *,
        method: str = "svd",
        apply_land_mask: bool = False,
        ocean_only: bool = True,
        mask_resolution: str = "c",
    ) -> None:
        if method not in {"svd", "xeofs"}:
            raise ValueError("method must be 'svd' or 'xeofs'")
        self.method = method
        self.apply_land_mask = apply_land_mask
        self.ocean_only = ocean_only
        self.mask_resolution = mask_resolution
        self.land_mask: Optional[xr.DataArray] = None
        self.eof_results: Dict[str, object] = {}

    def create_land_mask(self, data: xr.DataArray, resolution: str = "c") -> xr.DataArray:
        """Create a land/ocean mask using ``global-land-mask`` if available."""
        if not _LAND_MASK_AVAILABLE or globe is None:
            raise ImportError("global_land_mask is required for land/ocean masking.")

        dims = self._detect_dims(data)
        lat_name = dims["lat"]
        lon_name = dims["lon"]
        if lat_name is None or lon_name is None:
            raise ValueError("Could not identify latitude and longitude coordinates.")

        lats = data[lat_name].values
        lons = data[lon_name].values
        lons_converted = np.where(lons > 180, lons - 360, lons)
        lon_grid, lat_grid = np.meshgrid(lons_converted, lats)
        land_mask_2d = globe.is_land(lat_grid, lon_grid)
        mask = xr.DataArray(land_mask_2d, coords={lat_name: lats, lon_name: lons}, dims=[lat_name, lon_name])
        self.land_mask = ~mask if self.ocean_only else mask
        return self.land_mask

    def _detect_dims(self, data: xr.DataArray) -> Dict[str, Optional[str]]:
        dims = list(data.dims)
        dims_lc = [dim.lower() for dim in dims]

        def find_dim(candidates: List[str]) -> Optional[str]:
            for candidate in candidates:
                if candidate in dims_lc:
                    return dims[dims_lc.index(candidate)]
            for index, dim in enumerate(dims_lc):
                for candidate in candidates:
                    if candidate in dim:
                        return dims[index]
            return None

        return {
            "time": find_dim(["time", "valid_time", "date", "datetime", "time_counter"]),
            "level": find_dim(["plev", "lev", "level", "pressure", "pfull"]),
            "lat": find_dim(["lat", "latitude", "y"]),
            "lon": find_dim(["lon", "longitude", "x"]),
            "ensemble": find_dim(["member", "ensemble", "ens"]),
        }

    def _compute_eof_svd(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute EOFs from a feature x sample matrix."""
        valid_mask = ~np.any(np.isnan(matrix), axis=0)
        matrix_valid = matrix[:, valid_mask]
        left, singular_values, _ = np.linalg.svd(matrix_valid, full_matrices=False)
        eof_patterns = left.T
        pc_series = np.dot(eof_patterns, matrix_valid)
        nt = matrix_valid.shape[1]
        eigenvalues = singular_values**2 / nt
        explained_variance = eigenvalues / np.sum(eigenvalues) * 100.0
        phi_L, phi_0, dof = self._estimate_dof(matrix_valid, L=1)
        eigenvalue_errors = explained_variance * np.sqrt(2.0 / dof)
        return {
            "eof_patterns": eof_patterns,
            "pc_series": pc_series,
            "eigenvalues": eigenvalues,
            "explained_variance": explained_variance,
            "eigenvalue_errors": eigenvalue_errors,
            "degrees_of_freedom": dof,
            "valid_mask": valid_mask,
            "phi_0": phi_0,
            "phi_L": phi_L,
        }

    def _compute_eof_xeofs(self, data: xr.DataArray, n_modes: int = 10) -> Dict[str, object]:
        """Compute EOFs using xeofs if available."""
        try:
            import xeofs as xe
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("xeofs is required for method='xeofs'.") from exc

        dims = self._detect_dims(data)
        time_dim = dims["time"]
        level_dim = dims["level"]
        lat_dim = dims["lat"]
        lon_dim = dims["lon"]

        if time_dim is None or lat_dim is None or lon_dim is None:
            raise ValueError("time/lat/lon dimensions are required.")

        if level_dim is not None:
            stacked = data.stack(sample=(time_dim, lat_dim, lon_dim)).rename({level_dim: "feature"})
            model = xe.single.EOF(n_modes=n_modes, check_nans=False)
            model.fit(stacked, dim="sample")
            components = model.components().rename({"feature": level_dim})
            scores = model.scores().unstack("sample")
        else:
            model = xe.single.EOF(n_modes=n_modes, check_nans=False)
            model.fit(data, dim=time_dim)
            components = model.components()
            scores = model.scores()

        explained_variance = model.explained_variance_ratio() * 100.0
        return {
            "eofs": components,
            "pc_scores": scores,
            "explained_variance": explained_variance.values,
            "eof_patterns": components.values,
            "pc_series": getattr(scores, "values", None),
            "eigenvalues": None,
            "eigenvalue_errors": None,
            "degrees_of_freedom": None,
            "valid_mask": None,
            "phi_0": None,
            "phi_L": None,
        }

    def _estimate_dof(self, matrix: np.ndarray, L: int = 1) -> Tuple[float, float, float]:
        """Estimate degrees of freedom for the North test."""
        nt = matrix.shape[1]
        covariance = 0.0
        for index in range(L, nt - L):
            covariance += np.sum(matrix[:, index] * matrix[:, index + L])
        phi_L = covariance / (nt - 2 * L)
        phi_0 = np.sum(matrix**2) / nt
        r_L = phi_L / phi_0
        dof = (1 - r_L**2) / (1 + r_L**2) * nt
        return phi_L, phi_0, dof

    def fit(
        self,
        data: xr.DataArray,
        *,
        time_slice: Optional[slice] = None,
        lat_slice: Optional[slice] = None,
        level_slice: Optional[slice] = None,
        n_harmonics: int = 3,
        n_modes: int = 10,
    ) -> Dict[str, object]:
        """Run the EOF workflow."""
        data = rename_standard_coordinates(data)
        data = normalize_longitude(data, target="0_360")

        dims = self._detect_dims(data)
        time_dim = dims["time"]
        lat_dim = dims["lat"]
        lon_dim = dims["lon"]
        level_dim = dims["level"]

        if time_dim is None or lat_dim is None or lon_dim is None:
            raise ValueError("Data must contain time/lat/lon dimensions.")

        if time_slice is not None:
            data = data.sel({time_dim: time_slice})
        if lat_slice is not None:
            data = data.sel({lat_dim: lat_slice})
        if level_slice is not None and level_dim is not None:
            data = data.sel({level_dim: level_slice})

        expected_order = [time_dim, lat_dim, lon_dim]
        if level_dim is not None:
            expected_order = [time_dim, level_dim, lat_dim, lon_dim]
        if list(data.dims) != expected_order:
            data = data.transpose(*expected_order)

        if self.apply_land_mask:
            mask_indexer = {time_dim: 0}
            if level_dim is not None:
                mask_indexer[level_dim] = 0
            mask = self.create_land_mask(data.isel(mask_indexer))
            data = data.where(mask.broadcast_like(data))

        try:
            climatology = data.groupby(f"{time_dim}.dayofyear").mean(dim=time_dim)
            climatology_smoothed = extract_low_harmonics(climatology, n_harm=n_harmonics, dim="dayofyear")
            anomalies = data.groupby(f"{time_dim}.dayofyear") - climatology_smoothed
        except Exception:
            climatology = data.mean(dim=time_dim)
            climatology_smoothed = climatology
            anomalies = data - climatology

        if self.method == "xeofs":
            eof_results = self._compute_eof_xeofs(anomalies, n_modes=n_modes)
            eofs = eof_results["eofs"]
            pcs = eof_results["pc_scores"]
        else:
            if level_dim is not None:
                stacked = anomalies.stack(sample=(time_dim, lat_dim, lon_dim))
                matrix = stacked.transpose(level_dim, "sample").values
                eof_results = self._compute_eof_svd(matrix)
                n_use = min(n_modes, eof_results["eof_patterns"].shape[0])
                eofs = xr.DataArray(
                    eof_results["eof_patterns"][:n_use, :],
                    dims=("mode", level_dim),
                    coords={"mode": np.arange(1, n_use + 1), level_dim: anomalies[level_dim]},
                    name="eofs",
                )
                valid_mask = eof_results["valid_mask"]
                pc_valid = eof_results["pc_series"][:n_use, :]
                full_scores = xr.DataArray(
                    np.full((n_use, stacked.sample.size), np.nan),
                    dims=("mode", "sample"),
                    coords={"mode": np.arange(1, n_use + 1), "sample": stacked.sample},
                )
                full_scores.loc[dict(sample=stacked.sample[valid_mask])] = pc_valid
                pcs = full_scores.unstack("sample").rename("pcs")
            else:
                stacked = anomalies.stack(feature=(lat_dim, lon_dim)).transpose("feature", time_dim)
                matrix = stacked.values
                eof_results = self._compute_eof_svd(matrix)
                n_use = min(n_modes, eof_results["eof_patterns"].shape[0])
                eof_flat = xr.DataArray(
                    eof_results["eof_patterns"][:n_use, :],
                    dims=("mode", "feature"),
                    coords={"mode": np.arange(1, n_use + 1), "feature": stacked.feature},
                    name="eofs",
                )
                eofs = eof_flat.unstack("feature")
                valid_mask = eof_results["valid_mask"]
                full_pc = np.full((n_use, stacked[time_dim].size), np.nan)
                full_pc[:, valid_mask] = eof_results["pc_series"][:n_use, :]
                pcs = xr.DataArray(
                    full_pc,
                    dims=("mode", time_dim),
                    coords={"mode": np.arange(1, n_use + 1), time_dim: anomalies[time_dim]},
                    name="pcs",
                )

        self.eof_results = {
            **eof_results,
            "eofs": eofs,
            "pc_scores": pcs,
            "pressure_levels": data[level_dim].values if level_dim is not None else None,
            "original_data": data,
            "anomalies": anomalies,
            "climatology": climatology_smoothed,
            "n_harmonics": n_harmonics,
            "mask_applied": self.apply_land_mask,
            "ocean_only": self.ocean_only if self.apply_land_mask else None,
            "method": self.method,
        }
        return self.eof_results

    def plot_vertical_profiles(
        self,
        *,
        n_modes: int = 4,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        normalize: bool = True,
    ) -> plt.Figure:
        """Plot EOF vertical profiles when the decomposition uses a level dimension."""
        if not self.eof_results:
            raise ValueError("Run fit() first.")

        eofs = self.eof_results["eofs"]
        if not isinstance(eofs, xr.DataArray) or len(eofs.dims) != 2:
            raise ValueError("Vertical-profile plotting requires EOFs with a level dimension.")

        _, level_dim = eofs.dims
        pressure_levels = eofs[level_dim].values
        patterns = eofs.values.copy()
        explained = np.asarray(self.eof_results["explained_variance"])

        if normalize:
            for index in range(patterns.shape[0]):
                max_value = np.abs(patterns[index, :]).max()
                if max_value > 0:
                    patterns[index, :] /= max_value

        figure, axis = plt.subplots(figsize=figsize)
        colors = ["#2E8B57", "#FF8C00", "#4169E1", "#DC143C"]
        for index in range(min(n_modes, patterns.shape[0])):
            axis.plot(
                patterns[index, :],
                pressure_levels,
                label=f"EOF{index+1} ({explained[index]:.1f}%)",
                color=colors[index % len(colors)],
                linewidth=3,
                marker="o",
                markersize=4,
            )

        axis.axvline(0, color="black", linestyle=":", alpha=0.7, linewidth=1)
        axis.set_ylim(pressure_levels.max(), pressure_levels.min())
        axis.set_yscale("log")
        axis.set_xlabel("Normalized EOF Amplitude" if normalize else "EOF Amplitude")
        axis.set_ylabel(level_dim)
        axis.legend(loc="best")
        axis.grid(True, alpha=0.3)
        axis.set_title(f"EOF Vertical Profiles - {self.method.upper()}")
        plt.tight_layout()

        if save_path:
            figure.savefig(save_path, dpi=300, bbox_inches="tight")
        return figure

    def save_results(self, save_path: str) -> None:
        """Serialize EOF results to disk."""
        if not self.eof_results:
            raise ValueError("No EOF results available.")
        with open(save_path, "wb") as file_handle:
            pickle.dump(self.eof_results, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_results(self, load_path: str) -> Dict[str, object]:
        """Load EOF results from disk."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)
        with open(load_path, "rb") as file_handle:
            self.eof_results = pickle.load(file_handle)
        self.method = str(self.eof_results.get("method", "svd"))
        return self.eof_results


def quick_eof_analysis(
    data: xr.DataArray,
    *,
    method: str = "svd",
    n_modes: int = 4,
    time_slice: Optional[slice] = None,
    lat_slice: Optional[slice] = None,
    level_slice: Optional[slice] = None,
    n_harmonics: int = 3,
    plot: bool = True,
    save_fig: Optional[str] = None,
) -> Tuple[EOFAnalyzer, Dict[str, object], Optional[plt.Figure]]:
    """Run EOF analysis in one call."""
    analyzer = EOFAnalyzer(method=method)
    results = analyzer.fit(
        data=data,
        time_slice=time_slice,
        lat_slice=lat_slice,
        level_slice=level_slice,
        n_harmonics=n_harmonics,
        n_modes=n_modes,
    )
    figure = analyzer.plot_vertical_profiles(n_modes=n_modes, save_path=save_fig) if plot and results["pressure_levels"] is not None else None
    return analyzer, results, figure


def eof_svd(data: xr.DataArray, **kwargs: object) -> Tuple[EOFAnalyzer, Dict[str, object], Optional[plt.Figure]]:
    """Convenience wrapper for the SVD backend."""
    return quick_eof_analysis(data, method="svd", **kwargs)


def eof_xeofs(data: xr.DataArray, **kwargs: object) -> Tuple[EOFAnalyzer, Dict[str, object], Optional[plt.Figure]]:
    """Convenience wrapper for the xeofs backend."""
    return quick_eof_analysis(data, method="xeofs", **kwargs)


def _array_to_features(vert_vel: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray, Dict[str, xr.DataArray]]:
    """Convert a 4D field to a NaN-filtered feature matrix."""
    stacked = vert_vel.stack(sample=("time", "lat", "lon"))
    original_coords = {"sample": stacked.sample, "level": stacked.level}
    valid_mask = ~stacked.isnull().all(dim="level")
    features = stacked.isel(sample=valid_mask.values).persist() if hasattr(stacked, "persist") else stacked.isel(sample=valid_mask.values)
    return features, valid_mask, original_coords


def _get_eof_model(features: xr.DataArray, n_modes: int = 2):
    """Fit an xeofs EOF model."""
    import xeofs as xe  # pragma: no cover - optional dependency

    model = xe.single.EOF(n_modes=n_modes, check_nans=False)
    model.fit(features, dim="sample")
    return model


def _reconstruct_scores(
    ds: xr.DataArray,
    scores: xr.DataArray,
    valid_mask: xr.DataArray,
    original_coords: Dict[str, xr.DataArray],
) -> xr.DataArray:
    """Reconstruct score fields to the original sample space."""
    full_scores = xr.DataArray(
        np.full((scores.mode.size, len(original_coords["sample"])), np.nan),
        coords={"mode": scores.mode, "sample": original_coords["sample"]},
        dims=["mode", "sample"],
    )
    full_scores.loc[dict(sample=original_coords["sample"][valid_mask])] = scores.values
    return full_scores.unstack("sample")


def vertical_eof_with_nan_handling(
    vert_vel: xr.DataArray,
    *,
    n_modes: int = 2,
    zg: Optional[xr.DataArray] = None,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Perform EOF analysis on vertical-velocity data with NaN handling."""
    features, valid_mask, original_coords = _array_to_features(vert_vel)
    model = _get_eof_model(features, n_modes=n_modes)
    components = model.components()
    scores = model.scores()
    scores = _reconstruct_scores(vert_vel, scores, valid_mask, original_coords)
    explained_variance = model.explained_variance_ratio()
    eofs = components
    if zg is not None:
        eofs = eofs.assign_coords(level=zg)
    return eofs, scores, explained_variance


def align_eof_signs(eof_ref: xr.DataArray, eof_target: xr.DataArray) -> int:
    """Return ``+1`` or ``-1`` to align EOF signs by correlation."""
    correlation = np.corrcoef(eof_ref.values, eof_target.values)[0, 1]
    return 1 if correlation > 0 else -1


def compare_vertical_eofs(
    eofs_dict: Dict[str, Tuple[xr.DataArray, xr.DataArray, xr.DataArray]],
    *,
    reference_key: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    colors: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare vertical EOF profiles from multiple experiments."""
    if colors is None:
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#006D77", "#E63946"]

    markers = ["o", "s", "^", "D", "v"]
    figure, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    aligned = {}
    if reference_key and reference_key in eofs_dict:
        reference_eofs = eofs_dict[reference_key][0]
        for name, (eofs, pcs, variance) in eofs_dict.items():
            sign_mode1 = align_eof_signs(reference_eofs.isel(mode=0), eofs.isel(mode=0))
            sign_mode2 = align_eof_signs(reference_eofs.isel(mode=1), eofs.isel(mode=1))
            signs = xr.DataArray([sign_mode1, sign_mode2], dims=["mode"], coords={"mode": eofs.mode[:2]})
            aligned[name] = (eofs * signs, pcs, variance)
    else:
        aligned = eofs_dict

    for index, (name, (eofs, _, variance)) in enumerate(aligned.items()):
        axes[0].plot(
            eofs.isel(mode=0),
            eofs.level,
            label=f"{name} ({float(variance.isel(mode=0))*100:.1f}%)",
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
        )
        axes[1].plot(
            eofs.isel(mode=1),
            eofs.level,
            label=f"{name} ({float(variance.isel(mode=1))*100:.1f}%)",
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
        )

    for axis, title in zip(axes, ["EOF1", "EOF2"]):
        axis.axvline(0, color="black", linestyle=":")
        axis.set_yscale("log")
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()

    if save_path:
        figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return figure


__all__ = [
    "EOFAnalyzer",
    "quick_eof_analysis",
    "eof_svd",
    "eof_xeofs",
    "vertical_eof_with_nan_handling",
    "align_eof_signs",
    "compare_vertical_eofs",
]

