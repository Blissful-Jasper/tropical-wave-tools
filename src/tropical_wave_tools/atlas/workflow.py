"""Figure titles, label formatting and the end-to-end local wave atlas workflow."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from tropical_wave_tools.atlas._config import (
    CASE03_HOV_LON_REFS,
    CASE04_LAT_RANGE,
    CASE04_WAVES,
    CASE05_LAT_RANGE,
    CASE05_PRECIP_VARIABLE,
    CASE05_REGION_LABELS,
    CASE05_REGION_RANGES,
    CASE05_WAVES,
    CASE06_WAVE_QUIVER_SCALES,
    CASE06_WAVE_QUIVER_STRIDES,
    CASE07_WAVE_QUIVER_SCALES,
    CASE07_WAVE_QUIVER_STRIDES,
    CASE07_WAVES,
    CASE08_EVENT_BASE_POINTS,
    CASE08_EVENT_LAT_BANDS,
    CASE08_EVENT_LON_REFS,
    CASE08_EVENT_MIN_SPACING,
    CASE08_WAVE_FOCUS_CENTERS,
    CASE08_WAVE_FOCUS_HALF_WIDTH,
    CASE08_WAVE_LAGS,
    CASE08_WAVE_MAP_LAGS,
    CASE08_WAVE_OLR_LEVEL_COUNT,
    CASE08_WAVE_OLR_MIN_LIMIT,
    CASE08_WAVE_OLR_RANGE_SCALE,
    CASE08_WAVE_QUIVER_SCALES,
    CASE08_WAVE_QUIVER_STRIDES,
    CASE08_WAVE_QUIVER_WIDTHS,
    CASE09_WAVES,
    CASE10_BASE_POINTS,
    CASE10_CONTOUR_TARGET_STEPS,
    CASE10_FIGURE_SIZES,
    CASE10_SHADING_MIN_LIMIT,
    CASE10_SHADING_RANGE_SCALE,
    CASE10_WAVES,
    CASE10_XTICKS,
    COMPARISON_WAVE_GROUPS,
    WAVE_HOVMOLLER_WINDOWS,
    WAVE_PROPAGATION_DIRECTIONS,
    WAVE_QUIVER_SCALES,
)
from tropical_wave_tools.atlas._paths import (
    DEFAULT_LOCAL_PATHS,
    PathLike,
    fill_missing_with_time_mean,
    load_local_wave_fields,
)
from tropical_wave_tools.atlas.case10 import (
    compute_case10_regression_hovmoller,
)
from tropical_wave_tools.atlas.events import (
    _hovmoller_projection,
    _lat_range,
    detect_point_events,
    detect_wave_events,
    lagged_composite,
    select_hovmoller_window,
    wave_longitude_projection,
)
from tropical_wave_tools.atlas.monthly import (
    compute_longitude_mean_monthly_rms_climatology_and_significance,
    compute_monthly_climatology_and_significance,
    compute_monthly_rms,
    compute_monthly_variance_fraction_samples,
    compute_yearly_rms,
    summarize_variance_fraction_cycle,
)
from tropical_wave_tools.atlas.wave_eof import (
    compute_wave_eof,
    regress_field_onto_pcs,
)
from tropical_wave_tools.config import DEFAULT_WAVE_SPECS, SpectralConfig
from tropical_wave_tools.diagnostics import (
    horizontal_divergence,
    relative_vorticity,
)
from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.io import load_dataarray, save_dataset
from tropical_wave_tools.plotting import (
    plot_case05_regional_variance_cycles,
    plot_case05_seasonal_variance_cycles,
    plot_eof_modes_with_wind,
    plot_horizontal_structure,
    plot_hovmoller_triptych,
    plot_lag_longitude_evolution,
    plot_lagged_horizontal_structure,
    plot_monthly_cycle,
    plot_monthly_longitude_heatmap,
    plot_paper_style_hovmoller,
    plot_spatial_std_triptych,
    plot_wave_annual_trend_comparison,
    plot_wave_monthly_cycle_comparison,
    plot_wave_monthly_longitude_comparison,
    plot_wave_spatial_comparison,
    plot_wind_diagnostics_panel,
    plot_wk_spectrum,
)
from tropical_wave_tools.preprocessing import compute_anomaly
from tropical_wave_tools.spectral import analyze_wk_spectrum
from tropical_wave_tools.stats import linear_trend


def _case08_wave_figure_title(wave_name: str) -> str:
    """Return a publication-style figure title for Case 08."""
    titles = {
        "kelvin": "Kelvin wave composite phase evolution",
        "er": "Equatorial Rossby composite phase evolution",
        "mjo": "MJO composite phase evolution",
        "mrg": "Mixed Rossby-Gravity composite phase evolution",
        "td": "Tropical disturbance composite phase evolution",
    }
    return titles.get(wave_name.lower(), f"{wave_name.upper()} regional phase evolution")


def _case10_wave_figure_title(wave_name: str) -> str:
    """Return a publication-style Case 10 panel title."""
    titles = {
        "kelvin": "Kelvin wave",
        "er": "Equatorial Rossby wave",
        "mjo": "Madden-Julian Oscillation",
        "mrg": "Mixed Rossby-Gravity wave",
        "td": "Tropical disturbance",
    }
    return titles.get(wave_name.lower(), wave_name.upper())


def _format_lat_label(value: float) -> str:
    """Format a latitude coordinate into a compact N/S string."""
    hemi = "N" if value >= 0.0 else "S"
    magnitude = abs(float(value))
    if np.isclose(magnitude, round(magnitude)):
        return f"{round(magnitude)}{hemi}"
    return f"{magnitude:.1f}{hemi}"


def _format_lon_label(value: float) -> str:
    """Format a 0-360 longitude into a compact E/W string."""
    lon = float(value) % 360.0
    if np.isclose(lon, 0.0) or np.isclose(lon, 360.0):
        return "0"
    if np.isclose(lon, 180.0):
        return "180"
    if lon < 180.0:
        return f"{round(lon)}E"
    return f"{round(360.0 - lon)}W"


def _group_comparison_waves(waves: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Return ordered wave subsets for less crowded multi-panel comparisons."""
    available = [wave.lower() for wave in waves]
    grouped: list[tuple[str, list[str]]] = []
    assigned: set[str] = set()

    for group_name, members in COMPARISON_WAVE_GROUPS:
        selected = [wave for wave in members if wave in available and wave not in assigned]
        if selected:
            grouped.append((group_name, selected))
            assigned.update(selected)

    leftovers = [wave for wave in available if wave not in assigned]
    if leftovers:
        grouped.append(("other", leftovers))

    return grouped


def _comparison_ncols(n_panels: int) -> int:
    """Choose a readable panel count for grouped atlas figures."""
    return max(1, min(3, int(n_panels)))


def _save_wk_triplet(
    dataset: xr.Dataset,
    output_root: Path,
    *,
    spectral_lat_range: tuple[float, float] = (-15.0, 15.0),
    spectral_config: SpectralConfig | None = None,
) -> None:
    """Generate WK spectra for OLR, U850, and V850."""
    spectra_dir = output_root / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    config = spectral_config or SpectralConfig()

    for var_name in ("olr", "u850", "v850"):
        result = analyze_wk_spectrum(_lat_range(dataset[var_name], *spectral_lat_range), config=config)
        figure, _ = plot_wk_spectrum(result, save_path=spectra_dir / f"wk_spectrum_{var_name}.png")
        plt.close(figure)
        save_dataset(result.to_dataset(), spectra_dir / f"wk_spectrum_{var_name}.nc")


def generate_local_wave_atlas(
    *,
    output_dir: PathLike,
    olr_path: PathLike = DEFAULT_LOCAL_PATHS["olr"],
    u850_path: PathLike = DEFAULT_LOCAL_PATHS["u850"],
    v850_path: PathLike = DEFAULT_LOCAL_PATHS["v850"],
    waves: Iterable[str] | None = None,
    time_range: tuple[str, str] | None = ("1979-01-01", "2014-12-31"),
    lat_range: tuple[float, float] = (-20.0, 20.0),
    lat_band: tuple[float, float] = (-5.0, 5.0),
    lon_ref: float = 180.0,
    hovmoller_days: int = 180,
    event_threshold_std: float = 1.0,
    event_min_spacing_days: int = 7,
    lags: Sequence[int] = (-8, -6, -4, -2, 0, 2, 4, 6, 8),
    n_harm: int = 3,
    n_workers: int = 4,
) -> pd.DataFrame:
    """Generate a publication-oriented atlas from local OLR and low-level wind data."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_local_wave_fields(
        olr_path=olr_path,
        u850_path=u850_path,
        v850_path=v850_path,
        time_range=time_range,
        lat_range=lat_range,
    )
    raw_olr_anomaly = compute_anomaly(dataset["olr"], group="month")
    wave_list = [wave.lower() for wave in (waves or DEFAULT_WAVE_SPECS.keys())]
    multiwave_std_fields: dict[str, xr.DataArray] = {}
    multiwave_olr_lagged: dict[str, xr.DataArray] = {}
    multiwave_u_lagged: dict[str, xr.DataArray] = {}
    multiwave_v_lagged: dict[str, xr.DataArray] = {}
    multiwave_monthly_cycles: dict[str, dict[str, np.ndarray]] = {}
    multiwave_monthly_significance: dict[str, dict[str, xr.DataArray]] = {}
    multiwave_monthly_olr: dict[str, xr.DataArray] = {}
    multiwave_monthly_olr_significance: dict[str, xr.DataArray] = {}
    multiwave_annual_rms: dict[str, xr.DataArray] = {}
    multiwave_annual_trend_pvalue: dict[str, float] = {}
    multiwave_eofs: dict[str, xr.DataArray] = {}
    multiwave_eof_u: dict[str, xr.DataArray] = {}
    multiwave_eof_v: dict[str, xr.DataArray] = {}
    multiwave_eof_explained: dict[str, np.ndarray] = {}
    multiwave_event_indices: dict[str, np.ndarray] = {}
    multiwave_filtered_olr: dict[str, xr.DataArray] = {}

    _save_wk_triplet(dataset, output_root)

    rows: list[dict[str, object]] = []
    for wave_name in wave_list:
        wave_root = output_root / wave_name
        wave_root.mkdir(parents=True, exist_ok=True)

        filtered = {
            name: filter_wave_signal(
                dataset[name],
                wave_name=wave_name,
                method="cckw",
                obs_per_day=1,
                n_harm=n_harm,
                n_workers=n_workers,
            )
            for name in ("olr", "u850", "v850")
        }
        multiwave_filtered_olr[wave_name] = filtered["olr"]

        std_dataset = xr.Dataset({f"{name}_std": field.std("time") for name, field in filtered.items()})
        multiwave_std_fields[wave_name] = std_dataset["olr_std"]
        save_dataset(std_dataset, wave_root / f"{wave_name}_std_maps.nc")
        std_figure, _ = plot_spatial_std_triptych(
            std_dataset["olr_std"],
            std_dataset["u850_std"],
            std_dataset["v850_std"],
            titles=(
                f"{wave_name.upper()} OLR STD",
                f"{wave_name.upper()} U850 STD",
                f"{wave_name.upper()} V850 STD",
            ),
            save_path=wave_root / f"{wave_name}_spatial_std_triptych.png",
        )
        plt.close(std_figure)

        hov_lon_ref = CASE03_HOV_LON_REFS.get(wave_name, lon_ref)
        hov_projection_raw = _hovmoller_projection(raw_olr_anomaly, wave_name=wave_name, lat_band=lat_band)
        hov_projection_olr = _hovmoller_projection(filtered["olr"], wave_name=wave_name, lat_band=lat_band)
        hov_projection_u = _hovmoller_projection(filtered["u850"], wave_name=wave_name, lat_band=lat_band)
        hov_slice = select_hovmoller_window(
            hov_projection_olr,
            wave_name=None,
            lat_band=lat_band,
            lon_ref=hov_lon_ref,
            window_days=WAVE_HOVMOLLER_WINDOWS.get(wave_name, hovmoller_days),
        )
        hov_fields = [
            hov_projection_raw.sel(time=hov_slice),
            hov_projection_olr.sel(time=hov_slice),
            hov_projection_u.sel(time=hov_slice),
        ]
        hov_figure, _ = plot_hovmoller_triptych(
            hov_fields,
            titles=(
                f"{wave_name.upper()} raw OLR anomaly",
                f"{wave_name.upper()} filtered OLR",
                f"{wave_name.upper()} filtered U850",
            ),
            colorbar_labels=("Raw OLR anomaly", "Filtered OLR", "Filtered U850"),
            cmaps=("olr_diverging", "olr_diverging", "wave_diverging"),
            save_path=wave_root / f"{wave_name}_hovmoller_triptych.png",
        )
        plt.close(hov_figure)

        if wave_name in CASE10_WAVES:
            case10_ds = compute_case10_regression_hovmoller(
                raw_olr_anomaly,
                filtered["olr"],
                wave_name=wave_name,
            )
            base_lat, base_lon = CASE10_BASE_POINTS[wave_name]
            base_point_label = f"Base point: {_format_lat_label(base_lat)}, {_format_lon_label(base_lon)}"
            case10_figure, _ = plot_paper_style_hovmoller(
                case10_ds["shading"],
                case10_ds["contours"],
                title=_case10_wave_figure_title(wave_name),
                base_point_label=base_point_label,
                cmap="paper_hovmoller_diverging",
                figsize=CASE10_FIGURE_SIZES.get(wave_name, (7.4, 5.1)),
                shading_label="Regressed convection proxy from -OLR' (W m$^{-2}$)",
                shading_range_scale=CASE10_SHADING_RANGE_SCALE.get(wave_name, 1.25),
                shading_min_vmax=CASE10_SHADING_MIN_LIMIT.get(wave_name),
                contour_target_steps=CASE10_CONTOUR_TARGET_STEPS.get(wave_name, 6),
                xticks=CASE10_XTICKS.get(wave_name),
                save_path=wave_root / f"{wave_name}_paper_hovmoller.png",
            )
            plt.close(case10_figure)
            save_dataset(case10_ds, wave_root / f"{wave_name}_paper_hovmoller.nc")

        event_indices, reference = detect_wave_events(
            filtered["olr"],
            wave_name=wave_name,
            lat_band=lat_band,
            lon_ref=lon_ref,
            threshold_std=event_threshold_std,
            min_distance_days=event_min_spacing_days,
        )
        event_note: str | None = None
        if event_indices.size == 0:
            reference_values = np.asarray(reference.values, dtype=float)
            if np.any(np.isfinite(reference_values)):
                event_indices = np.array([int(np.nanargmax(np.abs(reference_values)))], dtype=int)
                event_note = "Used strongest absolute event as fallback."
            else:
                rows.append({"wave": wave_name, "event_count": 0, "note": "No finite reference values."})
                continue

        event_times = pd.to_datetime(reference.time.isel(time=event_indices).values)
        pd.DataFrame({"event_time": event_times}).to_csv(wave_root / f"{wave_name}_event_times.csv", index=False)
        multiwave_event_indices[wave_name] = np.asarray(event_indices, dtype=int)

        lagged_olr = lagged_composite(filtered["olr"], event_indices, lags=lags)
        lagged_u = lagged_composite(filtered["u850"], event_indices, lags=lags)
        lagged_v = lagged_composite(filtered["v850"], event_indices, lags=lags)
        composite_ds = xr.Dataset({"olr": lagged_olr, "u850": lagged_u, "v850": lagged_v})
        save_dataset(composite_ds, wave_root / f"{wave_name}_lagged_composites.nc")

        structure_figure, _ = plot_horizontal_structure(
            lagged_olr.sel(lag=0),
            lagged_u.sel(lag=0),
            lagged_v.sel(lag=0),
            title=f"{wave_name.upper()} composite horizontal structure (lag 0 d)",
            quiver_scale=WAVE_QUIVER_SCALES.get(wave_name, 55.0),
            integer_colorbar=True,
            save_path=wave_root / f"{wave_name}_horizontal_structure_lag0.png",
        )
        plt.close(structure_figure)
        multiwave_olr_lagged[wave_name] = lagged_olr
        multiwave_u_lagged[wave_name] = lagged_u
        multiwave_v_lagged[wave_name] = lagged_v

        lagged_figure, _ = plot_lagged_horizontal_structure(
            lagged_olr,
            lagged_u,
            lagged_v,
            lags=lags,
            quiver_scale=WAVE_QUIVER_SCALES.get(wave_name, 55.0),
            integer_colorbar=True,
            save_path=wave_root / f"{wave_name}_horizontal_structure_lags.png",
        )
        plt.close(lagged_figure)

        lead_lag_lags = tuple(int(value) for value in CASE08_WAVE_LAGS.get(wave_name, tuple(int(value) for value in lags)))
        lead_lag_event_lat_band = CASE08_EVENT_LAT_BANDS.get(wave_name, lat_band)
        lead_lag_lon_ref = CASE08_EVENT_LON_REFS.get(wave_name, lon_ref)
        lead_lag_event_spacing = CASE08_EVENT_MIN_SPACING.get(wave_name, event_min_spacing_days)
        lead_lag_base_point = CASE08_EVENT_BASE_POINTS.get(wave_name)
        if lead_lag_base_point is not None:
            lead_lag_event_indices, lead_lag_reference = detect_point_events(
                filtered["olr"],
                base_lat=lead_lag_base_point[0],
                base_lon=lead_lag_base_point[1],
                threshold_std=event_threshold_std,
                min_distance_days=lead_lag_event_spacing,
            )
        else:
            lead_lag_event_indices, lead_lag_reference = detect_wave_events(
                filtered["olr"],
                wave_name=wave_name,
                lat_band=lead_lag_event_lat_band,
                lon_ref=lead_lag_lon_ref,
                threshold_std=event_threshold_std,
                min_distance_days=lead_lag_event_spacing,
            )
        if lead_lag_event_indices.size == 0:
            lead_lag_reference_values = np.asarray(lead_lag_reference.values, dtype=float)
            if np.any(np.isfinite(lead_lag_reference_values)):
                lead_lag_event_indices = np.array([int(np.nanargmax(np.abs(lead_lag_reference_values)))], dtype=int)
            else:
                lead_lag_event_indices = event_indices
        lead_lag_olr = lagged_olr
        lead_lag_u = lagged_u
        lead_lag_v = lagged_v
        if tuple(int(value) for value in lagged_olr["lag"].values) != lead_lag_lags:
            lead_lag_olr = lagged_composite(filtered["olr"], lead_lag_event_indices, lags=lead_lag_lags)
            lead_lag_u = lagged_composite(filtered["u850"], lead_lag_event_indices, lags=lead_lag_lags)
            lead_lag_v = lagged_composite(filtered["v850"], lead_lag_event_indices, lags=lead_lag_lags)
        else:
            lead_lag_olr = lagged_composite(filtered["olr"], lead_lag_event_indices, lags=CASE08_WAVE_MAP_LAGS.get(wave_name, lags))
            lead_lag_u = lagged_composite(filtered["u850"], lead_lag_event_indices, lags=CASE08_WAVE_MAP_LAGS.get(wave_name, lags))
            lead_lag_v = lagged_composite(filtered["v850"], lead_lag_event_indices, lags=CASE08_WAVE_MAP_LAGS.get(wave_name, lags))
        if wave_name in CASE08_WAVE_MAP_LAGS:
            lead_lag_figure, _ = plot_lagged_horizontal_structure(
                lead_lag_olr,
                lead_lag_u,
                lead_lag_v,
                lags=CASE08_WAVE_MAP_LAGS[wave_name],
                ncols=3,
                quiver_stride=CASE08_WAVE_QUIVER_STRIDES.get(wave_name, 3),
                quiver_scale=CASE08_WAVE_QUIVER_SCALES.get(
                    wave_name,
                    WAVE_QUIVER_SCALES.get(wave_name, 55.0),
                ),
                integer_colorbar=False,
                colorbar_extend="both",
                olr_quantile=0.95 if wave_name == "mjo" else 0.94,
                olr_level_count=CASE08_WAVE_OLR_LEVEL_COUNT.get(wave_name, 33),
                olr_range_scale=CASE08_WAVE_OLR_RANGE_SCALE.get(wave_name, 1.25),
                olr_min_vmax=CASE08_WAVE_OLR_MIN_LIMIT.get(wave_name),
                wind_overlay="vectors",
                quiver_width=CASE08_WAVE_QUIVER_WIDTHS.get(wave_name, 0.0019),
                quiver_headwidth=3.4,
                quiver_headlength=4.2,
                quiver_headaxislength=3.9,
                focus_longitude=True,
                focus_center_lon=CASE08_WAVE_FOCUS_CENTERS.get(wave_name),
                focus_half_width=CASE08_WAVE_FOCUS_HALF_WIDTH.get(wave_name, 85.0),
                suptitle=_case08_wave_figure_title(wave_name),
                panel_title_template="Day {lag:+d}",
                save_path=wave_root / f"{wave_name}_lead_lag_evolution.png",
            )
        else:
            lead_lag_figure, _ = plot_lag_longitude_evolution(
                wave_longitude_projection(lead_lag_olr, wave_name=wave_name, lat_band=lat_band),
                title=f"{wave_name.upper()} lead-lag evolution",
                integer_colorbar=False,
                colorbar_extend="both",
                save_path=wave_root / f"{wave_name}_lead_lag_evolution.png",
            )
        plt.close(lead_lag_figure)

        divergence = horizontal_divergence(lagged_u.sel(lag=0), lagged_v.sel(lag=0))
        vorticity = relative_vorticity(lagged_u.sel(lag=0), lagged_v.sel(lag=0))
        wind_diag_figure, _ = plot_wind_diagnostics_panel(
            divergence,
            vorticity,
            lagged_u.sel(lag=0),
            lagged_v.sel(lag=0),
            titles=(
                f"{wave_name.upper()} divergence",
                f"{wave_name.upper()} vorticity",
            ),
            quiver_stride=CASE06_WAVE_QUIVER_STRIDES.get(wave_name, 4),
            quiver_scale=CASE06_WAVE_QUIVER_SCALES.get(wave_name, WAVE_QUIVER_SCALES.get(wave_name, 55.0)),
            integer_colorbar=True,
            colorbar_extend="neither",
            save_path=wave_root / f"{wave_name}_wind_diagnostics_lag0.png",
        )
        plt.close(wind_diag_figure)
        save_dataset(
            xr.Dataset({"divergence": divergence, "vorticity": vorticity}),
            wave_root / f"{wave_name}_wind_diagnostics_lag0.nc",
        )

        eofs, pcs, explained, eof_input = compute_wave_eof(filtered["olr"], wave_name=wave_name, n_modes=2)
        eof_olr = regress_field_onto_pcs(eof_input, pcs, standardize_pc=True)
        eof_u = regress_field_onto_pcs(filtered["u850"].sel(lat=eof_input["lat"]), pcs, standardize_pc=True)
        eof_v = regress_field_onto_pcs(filtered["v850"].sel(lat=eof_input["lat"]), pcs, standardize_pc=True)
        multiwave_eofs[wave_name] = eof_olr
        multiwave_eof_u[wave_name] = eof_u
        multiwave_eof_v[wave_name] = eof_v
        multiwave_eof_explained[wave_name] = explained
        if wave_name in CASE07_WAVES:
            eof_figure, _ = plot_eof_modes_with_wind(
                eof_olr,
                eof_u,
                eof_v,
                explained,
                modes=(1, 2),
                wave_name=wave_name,
                quiver_stride=CASE07_WAVE_QUIVER_STRIDES.get(wave_name, 4),
                quiver_scale=CASE07_WAVE_QUIVER_SCALES.get(
                    wave_name,
                    WAVE_QUIVER_SCALES.get(wave_name, 55.0),
                ),
                integer_colorbar=True,
                field_label="PC-regressed OLR (W m$^{-2}$)",
                save_path=wave_root / f"{wave_name}_eof_modes.png",
            )
            plt.close(eof_figure)
        save_dataset(
            xr.Dataset(
                {
                    "eofs": eofs,
                    "pcs": pcs,
                    "olr_regressed": eof_olr,
                    "u850_regressed": eof_u,
                    "v850_regressed": eof_v,
                    "eof_input_olr": eof_input,
                    "explained_variance": xr.DataArray(
                        explained[: eofs.sizes["mode"]],
                        dims=("mode",),
                        coords={"mode": eofs["mode"]},
                    ),
                }
            ),
            wave_root / f"{wave_name}_eof_modes.nc",
        )

        monthly_projection = {
            name: wave_longitude_projection(field, wave_name=wave_name, lat_band=lat_band)
            for name, field in filtered.items()
        }
        olr_line_clim, olr_line_pvalue = compute_longitude_mean_monthly_rms_climatology_and_significance(
            monthly_projection["olr"]
        )
        u_line_clim, u_line_pvalue = compute_longitude_mean_monthly_rms_climatology_and_significance(
            monthly_projection["u850"]
        )
        v_line_clim, v_line_pvalue = compute_longitude_mean_monthly_rms_climatology_and_significance(
            monthly_projection["v850"]
        )
        olr_monthly_rms, olr_monthly_pvalue = compute_monthly_climatology_and_significance(monthly_projection["olr"])
        u_monthly_rms = compute_monthly_rms(monthly_projection["u850"])
        v_monthly_rms = compute_monthly_rms(monthly_projection["v850"])
        monthly_cycle_data = {
            "OLR": olr_line_clim.values,
            "U850": u_line_clim.values,
            "V850": v_line_clim.values,
        }
        multiwave_monthly_cycles[wave_name] = monthly_cycle_data
        multiwave_monthly_significance[wave_name] = {
            "OLR": olr_line_pvalue,
            "U850": u_line_pvalue,
            "V850": v_line_pvalue,
        }
        multiwave_monthly_olr[wave_name] = olr_monthly_rms
        multiwave_monthly_olr_significance[wave_name] = olr_monthly_pvalue
        cycle_figure, _ = plot_monthly_cycle(
            np.arange(1, 13),
            monthly_cycle_data,
            title=f"{wave_name.upper()} projected monthly RMS cycle",
            ylabel="Projected RMS amplitude",
            save_path=wave_root / f"{wave_name}_monthly_cycle.png",
        )
        plt.close(cycle_figure)

        month_lon_figure, _ = plot_monthly_longitude_heatmap(
            olr_monthly_rms,
            title=f"{wave_name.upper()} OLR seasonal evolution",
            colorbar_label="Projected monthly RMS",
            save_path=wave_root / f"{wave_name}_seasonal_longitude_olr.png",
        )
        plt.close(month_lon_figure)

        annual_rms_olr = compute_yearly_rms(monthly_projection["olr"]).mean("lon").rename("annual_rms")
        multiwave_annual_rms[wave_name] = annual_rms_olr
        trend_ds = linear_trend(annual_rms_olr, dim="year")
        multiwave_annual_trend_pvalue[wave_name] = float(trend_ds["pvalue"].item())

        save_dataset(
            xr.Dataset(
                {
                    "olr_monthly_rms": olr_monthly_rms,
                    "u850_monthly_rms": u_monthly_rms,
                    "v850_monthly_rms": v_monthly_rms,
                    "olr_monthly_pvalue": olr_monthly_pvalue,
                    "olr_line_monthly_pvalue": olr_line_pvalue,
                    "u850_line_monthly_pvalue": u_line_pvalue,
                    "v850_line_monthly_pvalue": v_line_pvalue,
                    "olr_annual_rms": annual_rms_olr,
                    "olr_annual_rms_trend_slope": trend_ds["slope"],
                    "olr_annual_rms_trend_intercept": trend_ds["intercept"],
                    "olr_annual_rms_trend_pvalue": trend_ds["pvalue"],
                    "olr_annual_rms_trend_stderr": trend_ds["stderr"],
                }
            ),
            wave_root / f"{wave_name}_monthly_rms.nc",
        )

        rows.append(
            {
                "wave": wave_name,
                "event_count": int(event_indices.size),
                "time_start": str(dataset.time.values[0]),
                "time_end": str(dataset.time.values[-1]),
                "reference_projection": reference.attrs.get("wave_projection", "equatorial_mean"),
                "expected_direction": WAVE_PROPAGATION_DIRECTIONS.get(wave_name, "unknown"),
                "note": event_note,
            }
        )

    summary = pd.DataFrame(rows)
    compare_spatial_waves = [wave for wave in CASE04_WAVES if wave in multiwave_std_fields]
    if compare_spatial_waves:
        case04_olr = fill_missing_with_time_mean(
            load_dataarray(
                olr_path,
                variable="olr",
                time_range=time_range,
                lat_range=CASE04_LAT_RANGE,
            )
        )
        case04_std_fields = {
            wave: filter_wave_signal(
                case04_olr,
                wave_name=wave,
                method="cckw",
                obs_per_day=1,
                n_harm=n_harm,
                n_workers=n_workers,
            ).std("time")
            for wave in compare_spatial_waves
        }
        save_dataset(
            xr.Dataset({f"{wave}_olr_std": case04_std_fields[wave] for wave in compare_spatial_waves}),
            output_root / "multiwave_filtered_olr_std.nc",
        )
        for group_name, group_waves in _group_comparison_waves(compare_spatial_waves):
            compare_figure, _ = plot_wave_spatial_comparison(
                [case04_std_fields[wave] for wave in group_waves],
                titles=[wave.upper() for wave in group_waves],
                colorbar_label="Filtered OLR STD",
                ncols=1,
                cmap="wave_std_red",
                colorbar_orientation="vertical",
                integer_colorbar=True,
                target_steps=16,
                save_path=output_root / f"multiwave_{group_name}_filtered_olr_std.png",
            )
            plt.close(compare_figure)

    case05_waves = [wave for wave in CASE05_WAVES if wave in wave_list]
    if case05_waves:
        case05_source_name = "OLR"
        case05_note_text = "Monthly mean ± 1σ\n20°S–20°N regional mean\nOLR variance fraction"
        case05_description = "Monthly OLR variance fraction summaries for Case 05."
        case05_raw_field = _lat_range(raw_olr_anomaly, *CASE05_LAT_RANGE)
        case05_filtered_fields: dict[str, xr.DataArray] = {
            wave: _lat_range(multiwave_filtered_olr[wave], *CASE05_LAT_RANGE)
            for wave in case05_waves
            if wave in multiwave_filtered_olr
        }

        case05_gpcp_path = Path(DEFAULT_LOCAL_PATHS["gpcp"])
        if case05_gpcp_path.exists():
            case05_precip = fill_missing_with_time_mean(
                load_dataarray(
                    case05_gpcp_path,
                    variable=CASE05_PRECIP_VARIABLE,
                    time_range=time_range,
                    lat_range=CASE05_LAT_RANGE,
                )
            )
            case05_raw_field = compute_anomaly(case05_precip, group="month")
            case05_filtered_fields = {
                wave: filter_wave_signal(
                    case05_raw_field,
                    wave_name=wave,
                    method="cckw",
                    obs_per_day=1,
                    n_harm=n_harm,
                    n_workers=n_workers,
                ).sel(lat=slice(*CASE05_LAT_RANGE))
                for wave in case05_waves
            }
            case05_source_name = "GPCP precipitation"
            case05_note_text = "Monthly mean ± 1σ\n20°S–20°N regional mean\nGPCP precipitation variance fraction"
            case05_description = "Monthly GPCP precipitation variance fraction summaries for Case 05."

        case05_waves = [wave for wave in case05_waves if wave in case05_filtered_fields]
        tropical_means: list[xr.DataArray] = []
        tropical_stds: list[xr.DataArray] = []
        regional_means: list[xr.DataArray] = []
        regional_stds: list[xr.DataArray] = []

        for wave in case05_waves:
            variance_fraction = compute_monthly_variance_fraction_samples(
                case05_raw_field,
                case05_filtered_fields[wave],
                lat_range=CASE05_LAT_RANGE,
            )
            tropical_mean, tropical_std = summarize_variance_fraction_cycle(
                variance_fraction,
            )
            tropical_means.append(tropical_mean.expand_dims(wave=[wave]))
            tropical_stds.append(tropical_std.expand_dims(wave=[wave]))

            wave_regional_means: list[xr.DataArray] = []
            wave_regional_stds: list[xr.DataArray] = []
            for region_name, lon_range in CASE05_REGION_RANGES:
                region_mean, region_std = summarize_variance_fraction_cycle(
                    compute_monthly_variance_fraction_samples(
                        case05_raw_field,
                        case05_filtered_fields[wave],
                        lat_range=CASE05_LAT_RANGE,
                        lon_range=lon_range,
                    ),
                )
                wave_regional_means.append(region_mean.expand_dims(region=[region_name]))
                wave_regional_stds.append(region_std.expand_dims(region=[region_name]))

            regional_means.append(xr.concat(wave_regional_means, dim="region").expand_dims(wave=[wave]))
            regional_stds.append(xr.concat(wave_regional_stds, dim="region").expand_dims(wave=[wave]))

        tropical_mean_da = xr.concat(tropical_means, dim="wave")
        tropical_std_da = xr.concat(tropical_stds, dim="wave")
        regional_mean_da = xr.concat(regional_means, dim="wave").transpose("region", "wave", "month")
        regional_std_da = xr.concat(regional_stds, dim="wave").transpose("region", "wave", "month")
        case05_summary = xr.Dataset(
            {
                "tropical_mean": tropical_mean_da,
                "tropical_std": tropical_std_da,
                "regional_mean": regional_mean_da,
                "regional_std": regional_std_da,
            }
        )
        case05_summary.attrs["description"] = case05_description
        case05_summary.attrs["source"] = case05_source_name
        save_dataset(case05_summary, output_root / "case05_variance_fraction_summary.nc")

        case05_cycle_figure, _ = plot_case05_seasonal_variance_cycles(
            tropical_mean_da,
            tropical_std_da,
            wave_order=case05_waves,
            ylabel="Variance contribution [%]",
            save_path=output_root / "case05_seasonal_variance_cycle.png",
        )
        plt.close(case05_cycle_figure)

        case05_regional_figure, _ = plot_case05_regional_variance_cycles(
            regional_mean_da,
            regional_std_da,
            region_order=[name for name, _ in CASE05_REGION_RANGES],
            wave_order=case05_waves,
            region_labels=CASE05_REGION_LABELS,
            ylabel="Variance contribution [%]",
            note_text=case05_note_text,
            save_path=output_root / "case05_regional_variance_cycle.png",
        )
        plt.close(case05_regional_figure)

    compare_summary_waves = [
        wave
        for wave in CASE09_WAVES
        if wave in multiwave_monthly_cycles and wave in multiwave_monthly_olr and wave in multiwave_annual_rms
    ]
    if compare_summary_waves:
        for group_name, group_waves in _group_comparison_waves(compare_summary_waves):
            monthly_compare_figure, _ = plot_wave_monthly_cycle_comparison(
                {wave: multiwave_monthly_cycles[wave] for wave in group_waves},
                wave_names=group_waves,
                ncols=_comparison_ncols(len(group_waves)),
                monthly_significance={wave: multiwave_monthly_significance[wave] for wave in group_waves},
                normalize_each_series=(group_name == "westward"),
                normalization="annual_mean",
                save_path=output_root / f"multiwave_{group_name}_monthly_cycle.png",
            )
            plt.close(monthly_compare_figure)

            seasonal_compare_figure, _ = plot_wave_monthly_longitude_comparison(
                [multiwave_monthly_olr[wave] for wave in group_waves],
                wave_names=group_waves,
                ncols=_comparison_ncols(len(group_waves)),
                significance_fields=[multiwave_monthly_olr_significance[wave] for wave in group_waves],
                save_path=output_root / f"multiwave_{group_name}_seasonal_longitude_olr.png",
            )
            plt.close(seasonal_compare_figure)

            annual_compare_figure, _ = plot_wave_annual_trend_comparison(
                {wave: multiwave_annual_rms[wave] for wave in group_waves},
                wave_names=group_waves,
                ncols=_comparison_ncols(len(group_waves)),
                trend_pvalues={wave: multiwave_annual_trend_pvalue[wave] for wave in group_waves},
                save_path=output_root / f"multiwave_{group_name}_annual_rms_trend.png",
            )
            plt.close(annual_compare_figure)
    summary.to_csv(output_root / "wave_atlas_summary.csv", index=False)
    return summary
