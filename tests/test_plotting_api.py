"""The plotting package keeps the public names of the former single-file ``plotting`` module."""

from __future__ import annotations

import importlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

PUBLIC_PLOT_API = [
    "PUBLICATION_CMAPS",
    "get_cckw_envelope_curve",
    "get_publication_cmap",
    "plot_case05_regional_variance_cycles",
    "plot_case05_seasonal_variance_cycles",
    "plot_eof_modes_with_wind",
    "plot_eof_spatial_patterns_and_pcs",
    "plot_horizontal_structure",
    "plot_hovmoller_comparison",
    "plot_hovmoller_triptych",
    "plot_lag_longitude_evolution",
    "plot_lagged_horizontal_structure",
    "plot_latlon_field",
    "plot_monthly_cycle",
    "plot_monthly_longitude_heatmap",
    "plot_multiwave_eof_summary",
    "plot_paper_style_hovmoller",
    "plot_spatial_std_comparison",
    "plot_spatial_std_triptych",
    "plot_time_series",
    "plot_wave_annual_trend_comparison",
    "plot_wave_evolution_comparison",
    "plot_wave_horizontal_structure_comparison",
    "plot_wave_monthly_cycle_comparison",
    "plot_wave_monthly_longitude_comparison",
    "plot_wave_spatial_comparison",
    "plot_wind_diagnostics_panel",
    "plot_wk_spectrum",
    "save_figure",
    "scientific_plot_style",
    "use_scientific_style",
    "SCIENTIFIC_MPL_STYLE",
]


@pytest.mark.parametrize("name", PUBLIC_PLOT_API)
def test_public_plot_names_are_importable_from_the_package(name: str) -> None:
    package = importlib.import_module("tropical_wave_tools.plotting")
    assert hasattr(package, name)
    assert name in package.__all__


def test_lazy_top_level_exports_resolve() -> None:
    import tropical_wave_tools as twt

    for name in ("plot_wk_spectrum", "get_cckw_envelope_curve", "save_figure", "use_scientific_style"):
        assert callable(getattr(twt, name))


def test_submodules_group_the_functions_by_theme() -> None:
    expected = {
        "spectrum": "plot_wk_spectrum",
        "hovmoller": "plot_hovmoller_comparison",
        "maps": "plot_latlon_field",
        "eof": "plot_eof_modes_with_wind",
        "climatology": "plot_monthly_cycle",
    }
    for module, function in expected.items():
        assert hasattr(importlib.import_module(f"tropical_wave_tools.plotting.{module}"), function)


def test_representative_plots_render() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr

    from tropical_wave_tools import plotting
    from tropical_wave_tools.sample_data import open_example_olr
    from tropical_wave_tools.spectral import analyze_wk_spectrum

    olr = open_example_olr().load().sortby("lat")
    figure, _ = plotting.plot_wk_spectrum(analyze_wk_spectrum(olr))
    plt.close(figure)
    figure, _ = plotting.plot_latlon_field(olr.mean("time"), title="mean", use_cartopy=False)
    plt.close(figure)
    months = np.arange(1, 13)
    figure, _ = plotting.plot_monthly_cycle(months, {"a": np.sin(months / 2.0) + 2}, title="t", ylabel="y")
    plt.close(figure)
    heat = xr.DataArray(
        np.random.default_rng(0).random((12, 36)),
        dims=("month", "lon"),
        coords={"month": months, "lon": np.arange(36) * 10.0},
    )
    figure, _ = plotting.plot_monthly_longitude_heatmap(heat, title="h", colorbar_label="c")
    plt.close(figure)
    assert pd.Timestamp("2000-01-01") < pd.Timestamp("2001-01-01")  # keep pandas import used
