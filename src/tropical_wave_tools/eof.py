"""EOF analysis utilities ported from the original project."""

from __future__ import annotations

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import xarray as xr

from tropical_wave_tools.io import normalize_longitude, rename_standard_coordinates
from tropical_wave_tools.preprocessing import extract_low_harmonics

try:  # pragma: no cover - optional dependency
    from global_land_mask import globe

    _LAND_MASK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _LAND_MASK_AVAILABLE = False
    globe = None


def _thin_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left singular vectors and singular values of ``matrix``, robust to LAPACK failing to converge.

    ``numpy.linalg.svd`` uses the divide-and-conquer driver (``gesdd``), which can raise
    "SVD did not converge" on perfectly finite but degenerate matrices. A propagating wave
    projected onto its symmetric component is exactly that: rank-deficient (mirrored latitudes
    repeat) with exactly paired singular values. The slower QR-iteration driver (``gesvd``)
    handles them.
    """
    try:
        left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        left, singular_values, _ = scipy.linalg.svd(matrix, full_matrices=False, lapack_driver="gesvd")
    return left, singular_values


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
        self.land_mask: xr.DataArray | None = None
        self.eof_results: dict[str, object] = {}

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

    def _detect_dims(self, data: xr.DataArray) -> dict[str, str | None]:
        dims = list(data.dims)
        dims_lc = [dim.lower() for dim in dims]

        def find_dim(candidates: list[str]) -> str | None:
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

    @staticmethod
    def _select_valid(matrix: np.ndarray, nan_axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Drop the missing data of a ``feature x sample`` matrix; return it with the kept masks.

        ``nan_axis`` says what a missing value invalidates:

        * ``"sample"`` (vertical EOF; samples are time x lat x lon points): drop the samples
          that contain any missing value.
        * ``"feature"`` (horizontal EOF; features are grid points, samples are times): drop
          all-missing samples, then the grid points that still contain a missing value, for
          example a land mask. Dropping *samples* here would discard every time step as soon
          as one grid point is masked.
        """
        if nan_axis == "feature":
            valid_mask = ~np.all(np.isnan(matrix), axis=0)
            feature_valid = ~np.any(np.isnan(matrix[:, valid_mask]), axis=1)
            matrix_valid = matrix[np.ix_(feature_valid, valid_mask)]
        elif nan_axis == "sample":
            valid_mask = ~np.any(np.isnan(matrix), axis=0)
            feature_valid = np.ones(matrix.shape[0], dtype=bool)
            matrix_valid = matrix[:, valid_mask]
        else:
            raise ValueError("`nan_axis` must be 'sample' or 'feature'.")
        if matrix_valid.size == 0:
            raise ValueError("No valid data remain after removing missing values.")
        return matrix_valid, valid_mask, feature_valid

    def _assemble_eof_results(
        self,
        matrix: np.ndarray,
        matrix_valid: np.ndarray,
        patterns_valid: np.ndarray,
        singular_values: np.ndarray,
        valid_mask: np.ndarray,
        feature_valid: np.ndarray,
        temporal_shape: tuple[int, int] | None,
    ) -> dict[str, np.ndarray]:
        """Turn a decomposition of the valid data into the result dictionary shared by both backends.

        ``patterns_valid`` is ``mode x valid feature`` with unit-norm rows and ``singular_values``
        the matching singular values (at least as many modes as requested). The explained variance
        is relative to the *total* variance of the valid data, so truncated decompositions report
        the same percentages as the full one. The data are not re-centred: they are anomalies
        already.
        """
        nt = matrix_valid.shape[1]
        pc_series = np.dot(patterns_valid, matrix_valid)
        eof_patterns = np.full((patterns_valid.shape[0], matrix.shape[0]), np.nan)
        eof_patterns[:, feature_valid] = patterns_valid
        eigenvalues = singular_values**2 / nt
        total_variance = float(np.sum(matrix_valid**2)) / nt
        explained_variance = eigenvalues / total_variance * 100.0
        phi_L, phi_0, dof = self._estimate_dof(matrix, L=1, temporal_shape=temporal_shape)
        eigenvalue_errors = explained_variance * np.sqrt(2.0 / dof)
        return {
            "eof_patterns": eof_patterns,
            "pc_series": pc_series,
            "eigenvalues": eigenvalues,
            "explained_variance": explained_variance,
            "eigenvalue_errors": eigenvalue_errors,
            "degrees_of_freedom": dof,
            "valid_mask": valid_mask,
            "valid_feature_mask": feature_valid,
            "phi_0": phi_0,
            "phi_L": phi_L,
        }

    def _compute_eof_svd(
        self,
        matrix: np.ndarray,
        *,
        nan_axis: str = "sample",
        temporal_shape: tuple[int, int] | None = None,
        n_modes: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute EOFs from a ``feature x sample`` matrix with a full SVD.

        ``nan_axis`` is described in :meth:`_select_valid`. ``temporal_shape=(n_time, n_space)``
        tells the degrees-of-freedom estimate that the samples are ordered time-major, so the
        lag-1 autocorrelation is taken in time. ``n_modes`` is accepted for symmetry with the
        xeofs backend; the full decomposition is always computed.
        """
        matrix_valid, valid_mask, feature_valid = self._select_valid(matrix, nan_axis)
        left, singular_values = _thin_svd(matrix_valid)
        return self._assemble_eof_results(
            matrix, matrix_valid, left.T, singular_values, valid_mask, feature_valid, temporal_shape
        )

    def _compute_eof_xeofs(
        self,
        matrix: np.ndarray,
        *,
        nan_axis: str = "sample",
        temporal_shape: tuple[int, int] | None = None,
        n_modes: int = 10,
    ) -> dict[str, np.ndarray]:
        """Compute EOFs of a ``feature x sample`` matrix with xeofs.

        The missing-data handling, scaling and result dictionary are the same as for the SVD
        backend (xeofs itself refuses any NaN); only the decomposition is delegated to
        ``xeofs.single.EOF``, without re-centring or weighting. The modes are truncated to
        ``n_modes``. xeofs may use a randomized SVD for large problems, which is accurate for the
        leading, well-separated modes but approximate for weak ones.
        """
        try:
            import xeofs as xe
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("xeofs is required for method='xeofs'.") from exc

        matrix_valid, valid_mask, feature_valid = self._select_valid(matrix, nan_axis)
        n_use = int(min(n_modes, *matrix_valid.shape))
        model = xe.single.EOF(
            n_modes=n_use, center=False, standardize=False, use_coslat=False, check_nans=False
        )
        model.fit(
            xr.DataArray(
                matrix_valid.T,
                dims=("sample", "feature"),
                coords={"sample": np.arange(matrix_valid.shape[1]), "feature": np.arange(matrix_valid.shape[0])},
            ),
            dim="sample",
        )
        patterns_valid = model.components().transpose("mode", "feature").values
        singular_values = model.singular_values().values
        return self._assemble_eof_results(
            matrix, matrix_valid, patterns_valid, singular_values, valid_mask, feature_valid, temporal_shape
        )

    def _estimate_dof(
        self,
        matrix: np.ndarray,
        L: int = 1,
        *,
        temporal_shape: tuple[int, int] | None = None,
    ) -> tuple[float, float, float]:
        """Effective number of independent time samples for the North et al. (1982) test.

        ``matrix`` is ``feature x sample``. Without ``temporal_shape`` the samples are
        consecutive times. With ``temporal_shape=(n_time, n_space)`` they are ordered
        time-major (time, then space) and the lag is taken along time at each point; a lag along
        the stacked axis would compare neighbouring grid points instead and overstate the
        degrees of freedom by orders of magnitude. Spatial dependence is not modelled, so the
        result is the temporal effective sample size.
        """
        if temporal_shape is None:
            cube = matrix[:, :, np.newaxis]
        else:
            n_time, n_space = temporal_shape
            cube = matrix.reshape(matrix.shape[0], n_time, n_space)
        n_time = cube.shape[1]
        leading, lagged = cube[:, L:, :], cube[:, :-L, :]
        both = np.isfinite(leading) & np.isfinite(lagged)
        phi_L = float(np.sum(leading[both] * lagged[both]) / max(both.sum(), 1))
        finite = np.isfinite(cube)
        phi_0 = float(np.sum(cube[finite] ** 2) / max(finite.sum(), 1))
        r_L = phi_L / phi_0
        dof = (1 - r_L**2) / (1 + r_L**2) * n_time
        return phi_L, phi_0, dof

    def fit(
        self,
        data: xr.DataArray,
        *,
        time_slice: slice | None = None,
        lat_slice: slice | None = None,
        level_slice: slice | None = None,
        n_harmonics: int = 3,
        n_modes: int = 10,
    ) -> dict[str, object]:
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
            if n_harmonics <= 0:
                climatology_smoothed = data.mean(dim=time_dim)
                climatology = climatology_smoothed
                anomalies = data - climatology_smoothed
            else:
                climatology = data.groupby(f"{time_dim}.dayofyear").mean(dim=time_dim)
                climatology_smoothed = extract_low_harmonics(climatology, n_harm=n_harmonics, dim="dayofyear")
                anomalies = data.groupby(f"{time_dim}.dayofyear") - climatology_smoothed
        except (KeyError, AttributeError, ValueError) as exc:
            warnings.warn(
                f"Seasonal-cycle removal failed ({exc}); falling back to the time mean.",
                RuntimeWarning,
                stacklevel=2,
            )
            climatology = data.mean(dim=time_dim)
            climatology_smoothed = climatology
            anomalies = data - climatology

        compute = self._compute_eof_xeofs if self.method == "xeofs" else self._compute_eof_svd
        if level_dim is not None:
            stacked = anomalies.stack(sample=(time_dim, lat_dim, lon_dim))
            matrix = stacked.transpose(level_dim, "sample").values
            eof_results = compute(
                matrix,
                nan_axis="sample",
                temporal_shape=(anomalies[time_dim].size, anomalies[lat_dim].size * anomalies[lon_dim].size),
                n_modes=n_modes,
            )
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
            full_scores.loc[{"sample": stacked.sample[valid_mask]}] = pc_valid
            pcs = full_scores.unstack("sample").rename("pcs")
        else:
            stacked = anomalies.stack(feature=(lat_dim, lon_dim)).transpose("feature", time_dim)
            matrix = stacked.values
            eof_results = compute(matrix, nan_axis="feature", n_modes=n_modes)
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
        figsize: tuple[int, int] = (12, 10),
        save_path: str | None = None,
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

    def load_results(self, load_path: str) -> dict[str, object]:
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
    time_slice: slice | None = None,
    lat_slice: slice | None = None,
    level_slice: slice | None = None,
    n_harmonics: int = 3,
    plot: bool = True,
    save_fig: str | None = None,
) -> tuple[EOFAnalyzer, dict[str, object], plt.Figure | None]:
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


def eof_svd(data: xr.DataArray, **kwargs: object) -> tuple[EOFAnalyzer, dict[str, object], plt.Figure | None]:
    """Convenience wrapper for the SVD backend."""
    return quick_eof_analysis(data, method="svd", **kwargs)


def eof_xeofs(data: xr.DataArray, **kwargs: object) -> tuple[EOFAnalyzer, dict[str, object], plt.Figure | None]:
    """Convenience wrapper for the xeofs backend."""
    return quick_eof_analysis(data, method="xeofs", **kwargs)


def _array_to_features(vert_vel: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, dict[str, xr.DataArray]]:
    """Convert a 4D field to a feature matrix of the samples that are complete at every level.

    A sample (one time, latitude and longitude) with a missing level, for example a level below the
    ground, is dropped as a whole, which is also what ``EOFAnalyzer`` does for vertical EOFs.
    """
    stacked = vert_vel.stack(sample=("time", "lat", "lon"))
    original_coords = {"sample": stacked.sample, "level": stacked.level}
    valid_mask = ~stacked.isnull().any(dim="level")
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
    original_coords: dict[str, xr.DataArray],
) -> xr.DataArray:
    """Reconstruct score fields to the original sample space."""
    full_scores = xr.DataArray(
        np.full((scores.mode.size, len(original_coords["sample"])), np.nan),
        coords={"mode": scores.mode, "sample": original_coords["sample"]},
        dims=["mode", "sample"],
    )
    full_scores.loc[{"sample": original_coords["sample"][valid_mask]}] = scores.transpose("mode", "sample").values
    return full_scores.unstack("sample")


def vertical_eof_with_nan_handling(
    vert_vel: xr.DataArray,
    *,
    n_modes: int = 2,
    zg: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
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
    eofs_dict: dict[str, tuple[xr.DataArray, xr.DataArray, xr.DataArray]],
    *,
    reference_key: str | None = None,
    figsize: tuple[int, int] = (14, 6),
    colors: list[str] | None = None,
    save_path: str | None = None,
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
    "align_eof_signs",
    "compare_vertical_eofs",
    "eof_svd",
    "eof_xeofs",
    "quick_eof_analysis",
    "vertical_eof_with_nan_handling",
]
