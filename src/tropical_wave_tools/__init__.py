"""Modern tropical atmospheric wave analysis toolkit.

The top-level package keeps imports intentionally lightweight so that
submodules with optional or heavy dependencies are only imported on demand.
This avoids surprising side effects during packaging, testing, and app startup.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CCKWFilter": ("tropical_wave_tools.filters", "CCKWFilter"),
    "CrossSpectrumConfig": ("tropical_wave_tools.cross_spectrum", "CrossSpectrumConfig"),
    "DEFAULT_WAVE_SPECS": ("tropical_wave_tools.config", "DEFAULT_WAVE_SPECS"),
    "EOFAnalyzer": ("tropical_wave_tools.eof", "EOFAnalyzer"),
    "MemoryMonitor": ("tropical_wave_tools.cross_spectrum_analysis", "MemoryMonitor"),
    "SpectralConfig": ("tropical_wave_tools.config", "SpectralConfig"),
    "WKAnalysisResult": ("tropical_wave_tools.spectral", "WKAnalysisResult"),
    "WKSpectralAnalysis": ("tropical_wave_tools.spectral", "WKSpectralAnalysis"),
    "WaveFilter": ("tropical_wave_tools.filters", "WaveFilter"),
    "WaveSpec": ("tropical_wave_tools.config", "WaveSpec"),
    "align_eof_signs": ("tropical_wave_tools.eof", "align_eof_signs"),
    "analyze_cross_spectrum": (
        "tropical_wave_tools.cross_spectrum_analysis",
        "analyze_cross_spectrum",
    ),
    "analyze_wk_spectrum": ("tropical_wave_tools.spectral", "analyze_wk_spectrum"),
    "analyze_wk_spectrum_from_file": (
        "tropical_wave_tools.workflows",
        "analyze_wk_spectrum_from_file",
    ),
    "area_weighted_mean": ("tropical_wave_tools.diagnostics", "area_weighted_mean"),
    "butter_lowpass_filter": ("tropical_wave_tools.phase", "butter_lowpass_filter"),
    "calc_dse": ("tropical_wave_tools.diagnostics", "calc_dse"),
    "calc_horizontal_gms": ("tropical_wave_tools.diagnostics", "calc_horizontal_gms"),
    "calc_vertical_gms": ("tropical_wave_tools.diagnostics", "calc_vertical_gms"),
    "calculate_cross_spectrum": ("tropical_wave_tools.cross_spectrum", "calculate_cross_spectrum"),
    "calculate_kelvin_phase": ("tropical_wave_tools.phase", "calculate_kelvin_phase"),
    "compare_vertical_eofs": ("tropical_wave_tools.eof", "compare_vertical_eofs"),
    "composite_kw_phase": ("tropical_wave_tools.phase", "composite_kw_phase"),
    "compute_anomaly": ("tropical_wave_tools.preprocess", "compute_anomaly"),
    "compute_climatology": ("tropical_wave_tools.preprocess", "compute_climatology"),
    "compute_monthly_rms": ("tropical_wave_tools.atlas", "compute_monthly_rms"),
    "compute_case10_regression_hovmoller": ("tropical_wave_tools.atlas", "compute_case10_regression_hovmoller"),
    "compute_monthly_variance_fraction_samples": ("tropical_wave_tools.atlas", "compute_monthly_variance_fraction_samples"),
    "compute_yearly_rms": ("tropical_wave_tools.atlas", "compute_yearly_rms"),
    "compute_wave_eof": ("tropical_wave_tools.atlas", "compute_wave_eof"),
    "compute_cross_spectrum_for_experiments": (
        "tropical_wave_tools.cross_spectrum_analysis",
        "compute_cross_spectrum_for_experiments",
    ),
    "compute_dx_dy": ("tropical_wave_tools.diagnostics", "compute_dx_dy"),
    "create_demo_subset": ("tropical_wave_tools.workflows", "create_demo_subset"),
    "eof_svd": ("tropical_wave_tools.eof", "eof_svd"),
    "eof_xeofs": ("tropical_wave_tools.eof", "eof_xeofs"),
    "filter_wave_signal": ("tropical_wave_tools.filters", "filter_wave_signal"),
    "generate_local_wave_atlas": ("tropical_wave_tools.atlas", "generate_local_wave_atlas"),
    "get_cckw_envelope_curve": ("tropical_wave_tools.plotting", "get_cckw_envelope_curve"),
    "get_publication_cmap": ("tropical_wave_tools.plotting", "get_publication_cmap"),
    "get_sample_path": ("tropical_wave_tools.sample_data", "get_sample_path"),
    "horizontal_divergence": ("tropical_wave_tools.diagnostics", "horizontal_divergence"),
    "lag_composite": ("tropical_wave_tools.phase", "lag_composite"),
    "legacy_calculate_wk_spectrum": ("tropical_wave_tools.spectral", "calculate_wk_spectrum"),
    "linear_regression": ("tropical_wave_tools.stats", "linear_regression"),
    "linear_trend": ("tropical_wave_tools.stats", "linear_trend"),
    "load_dataarray": ("tropical_wave_tools.io", "load_dataarray"),
    "load_local_wave_fields": ("tropical_wave_tools.atlas", "load_local_wave_fields"),
    "load_dataset": ("tropical_wave_tools.io", "load_dataset"),
    "load_multiple_experiments": (
        "tropical_wave_tools.cross_spectrum_analysis",
        "load_multiple_experiments",
    ),
    "load_netcdf_data": ("tropical_wave_tools.cross_spectrum_analysis", "load_netcdf_data"),
    "meridional_mean": ("tropical_wave_tools.diagnostics", "meridional_mean"),
    "meridional_projection": ("tropical_wave_tools.phase", "meridional_projection"),
    "monthly_mean": ("tropical_wave_tools.preprocess", "monthly_mean"),
    "nan_to_value_by_interp_3D": ("tropical_wave_tools.cross_spectrum", "nan_to_value_by_interp_3D"),
    "normalize_longitude": ("tropical_wave_tools.io", "normalize_longitude"),
    "open_example_olr": ("tropical_wave_tools.sample_data", "open_example_olr"),
    "optimize_peak_detection": ("tropical_wave_tools.phase", "optimize_peak_detection"),
    "one_sample_ttest": ("tropical_wave_tools.stats", "one_sample_ttest"),
    "pearson_correlation": ("tropical_wave_tools.stats", "pearson_correlation"),
    "phase_composite": ("tropical_wave_tools.phase", "phase_composite"),
    "plot_cross_spectrum_panel": (
        "tropical_wave_tools.cross_spectrum_analysis",
        "plot_cross_spectrum_panel",
    ),
    "plot_horizontal_structure": ("tropical_wave_tools.plotting", "plot_horizontal_structure"),
    "plot_hovmoller_comparison": ("tropical_wave_tools.plotting", "plot_hovmoller_comparison"),
    "plot_paper_style_hovmoller": ("tropical_wave_tools.plotting", "plot_paper_style_hovmoller"),
    "plot_hovmoller_triptych": ("tropical_wave_tools.plotting", "plot_hovmoller_triptych"),
    "plot_latlon_field": ("tropical_wave_tools.plotting", "plot_latlon_field"),
    "plot_lag_longitude_evolution": ("tropical_wave_tools.plotting", "plot_lag_longitude_evolution"),
    "plot_lagged_horizontal_structure": ("tropical_wave_tools.plotting", "plot_lagged_horizontal_structure"),
    "plot_case05_regional_variance_cycles": ("tropical_wave_tools.plotting", "plot_case05_regional_variance_cycles"),
    "plot_case05_seasonal_variance_cycles": ("tropical_wave_tools.plotting", "plot_case05_seasonal_variance_cycles"),
    "plot_monthly_cycle": ("tropical_wave_tools.plotting", "plot_monthly_cycle"),
    "plot_monthly_longitude_heatmap": ("tropical_wave_tools.plotting", "plot_monthly_longitude_heatmap"),
    "plot_multiwave_eof_summary": ("tropical_wave_tools.plotting", "plot_multiwave_eof_summary"),
    "plot_eof_spatial_patterns_and_pcs": (
        "tropical_wave_tools.plotting",
        "plot_eof_spatial_patterns_and_pcs",
    ),
    "plot_eof_modes_with_wind": ("tropical_wave_tools.plotting", "plot_eof_modes_with_wind"),
    "plot_spatial_std_triptych": ("tropical_wave_tools.plotting", "plot_spatial_std_triptych"),
    "plot_spatial_std_comparison": ("tropical_wave_tools.plotting", "plot_spatial_std_comparison"),
    "plot_time_series": ("tropical_wave_tools.plotting", "plot_time_series"),
    "plot_wave_horizontal_structure_comparison": (
        "tropical_wave_tools.plotting",
        "plot_wave_horizontal_structure_comparison",
    ),
    "plot_wave_evolution_comparison": ("tropical_wave_tools.plotting", "plot_wave_evolution_comparison"),
    "plot_wave_annual_trend_comparison": ("tropical_wave_tools.plotting", "plot_wave_annual_trend_comparison"),
    "plot_wave_monthly_cycle_comparison": ("tropical_wave_tools.plotting", "plot_wave_monthly_cycle_comparison"),
    "plot_wave_monthly_longitude_comparison": ("tropical_wave_tools.plotting", "plot_wave_monthly_longitude_comparison"),
    "plot_wave_spatial_comparison": ("tropical_wave_tools.plotting", "plot_wave_spatial_comparison"),
    "plot_wind_diagnostics_panel": ("tropical_wave_tools.plotting", "plot_wind_diagnostics_panel"),
    "plot_wk_spectrum": ("tropical_wave_tools.plotting", "plot_wk_spectrum"),
    "scientific_plot_style": ("tropical_wave_tools.plotting", "scientific_plot_style"),
    "preprocess_data_with_mask": (
        "tropical_wave_tools.cross_spectrum_analysis",
        "preprocess_data_with_mask",
    ),
    "quick_cross_spectrum": ("tropical_wave_tools.cross_spectrum", "quick_cross_spectrum"),
    "quick_eof_analysis": ("tropical_wave_tools.eof", "quick_eof_analysis"),
    "remove_clm": ("tropical_wave_tools.phase", "remove_clm"),
    "save_composite_to_netcdf": ("tropical_wave_tools.phase", "save_composite_to_netcdf"),
    "save_figure": ("tropical_wave_tools.plotting", "save_figure"),
    "seasonal_mean": ("tropical_wave_tools.preprocess", "seasonal_mean"),
    "select_region": ("tropical_wave_tools.preprocess", "select_region"),
    "select_time": ("tropical_wave_tools.preprocess", "select_time"),
    "standard_deviation": ("tropical_wave_tools.stats", "standard_deviation"),
    "standardize_data": ("tropical_wave_tools.io", "standardize_data"),
    "to_dataarray": ("tropical_wave_tools.io", "to_dataarray"),
    "use_scientific_style": ("tropical_wave_tools.plotting", "use_scientific_style"),
    "variance": ("tropical_wave_tools.stats", "variance"),
    "vertical_eof_with_nan_handling": ("tropical_wave_tools.eof", "vertical_eof_with_nan_handling"),
    "relative_vorticity": ("tropical_wave_tools.diagnostics", "relative_vorticity"),
    "zonal_mean": ("tropical_wave_tools.diagnostics", "zonal_mean"),
}

__all__ = ["__version__", *_LAZY_IMPORTS.keys()]


def __getattr__(name: str) -> Any:
    """Load public attributes lazily to keep package import side-effect free."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the merged static and lazily exposed attributes."""
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
