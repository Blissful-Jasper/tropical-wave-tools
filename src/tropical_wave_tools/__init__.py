"""Modern tropical atmospheric wave analysis toolkit."""

from tropical_wave_tools.config import DEFAULT_WAVE_SPECS, SpectralConfig, WaveSpec
from tropical_wave_tools.cross_spectrum import (
    CrossSpectrumConfig,
    calculate_cross_spectrum,
    nan_to_value_by_interp_3D,
    quick_cross_spectrum,
)
from tropical_wave_tools.cross_spectrum_analysis import (
    MemoryMonitor,
    analyze_cross_spectrum,
    compute_cross_spectrum_for_experiments,
    load_multiple_experiments,
    load_netcdf_data,
    plot_cross_spectrum_panel,
    preprocess_data_with_mask,
)
from tropical_wave_tools.diagnostics import (
    area_weighted_mean,
    calc_dse,
    calc_horizontal_gms,
    calc_vertical_gms,
    compute_dx_dy,
    meridional_mean,
    zonal_mean,
)
from tropical_wave_tools.filters import CCKWFilter, WaveFilter, filter_wave_signal
from tropical_wave_tools.io import (
    load_dataarray,
    load_dataset,
    normalize_longitude,
    standardize_data,
    to_dataarray,
)
from tropical_wave_tools.eof import (
    EOFAnalyzer,
    align_eof_signs,
    compare_vertical_eofs,
    eof_svd,
    eof_xeofs,
    quick_eof_analysis,
    vertical_eof_with_nan_handling,
)
from tropical_wave_tools.phase import (
    butter_lowpass_filter,
    calculate_kelvin_phase,
    composite_kw_phase,
    lag_composite,
    meridional_projection,
    optimize_peak_detection,
    phase_composite,
    remove_clm,
    save_composite_to_netcdf,
)
from tropical_wave_tools.plotting import (
    get_cckw_envelope_curve,
    plot_latlon_field,
    plot_spatial_std_comparison,
    plot_time_series,
    plot_wk_spectrum,
    save_figure,
)
from tropical_wave_tools.preprocess import (
    compute_anomaly,
    compute_climatology,
    monthly_mean,
    seasonal_mean,
    select_region,
    select_time,
)
from tropical_wave_tools.sample_data import get_sample_path, open_example_olr
from tropical_wave_tools.spectral import WKAnalysisResult, WKSpectralAnalysis, analyze_wk_spectrum
from tropical_wave_tools.spectral import calculate_wk_spectrum as legacy_calculate_wk_spectrum
from tropical_wave_tools.stats import (
    linear_regression,
    linear_trend,
    pearson_correlation,
    standard_deviation,
    variance,
)
from tropical_wave_tools.workflows import analyze_wk_spectrum_from_file, create_demo_subset

__all__ = [
    "CCKWFilter",
    "CrossSpectrumConfig",
    "DEFAULT_WAVE_SPECS",
    "EOFAnalyzer",
    "MemoryMonitor",
    "SpectralConfig",
    "WKAnalysisResult",
    "WKSpectralAnalysis",
    "WaveFilter",
    "WaveSpec",
    "align_eof_signs",
    "analyze_cross_spectrum",
    "analyze_wk_spectrum",
    "analyze_wk_spectrum_from_file",
    "area_weighted_mean",
    "butter_lowpass_filter",
    "calc_dse",
    "calc_horizontal_gms",
    "calc_vertical_gms",
    "calculate_cross_spectrum",
    "calculate_kelvin_phase",
    "compare_vertical_eofs",
    "composite_kw_phase",
    "compute_anomaly",
    "compute_climatology",
    "compute_cross_spectrum_for_experiments",
    "compute_dx_dy",
    "create_demo_subset",
    "eof_svd",
    "eof_xeofs",
    "filter_wave_signal",
    "get_cckw_envelope_curve",
    "get_sample_path",
    "lag_composite",
    "legacy_calculate_wk_spectrum",
    "linear_regression",
    "linear_trend",
    "load_dataarray",
    "load_dataset",
    "load_multiple_experiments",
    "load_netcdf_data",
    "meridional_mean",
    "meridional_projection",
    "monthly_mean",
    "nan_to_value_by_interp_3D",
    "normalize_longitude",
    "open_example_olr",
    "optimize_peak_detection",
    "pearson_correlation",
    "phase_composite",
    "plot_latlon_field",
    "plot_cross_spectrum_panel",
    "plot_spatial_std_comparison",
    "plot_time_series",
    "plot_wk_spectrum",
    "preprocess_data_with_mask",
    "quick_cross_spectrum",
    "quick_eof_analysis",
    "remove_clm",
    "save_figure",
    "save_composite_to_netcdf",
    "seasonal_mean",
    "select_region",
    "select_time",
    "standard_deviation",
    "standardize_data",
    "to_dataarray",
    "variance",
    "vertical_eof_with_nan_handling",
    "zonal_mean",
]

__version__ = "0.1.0"
