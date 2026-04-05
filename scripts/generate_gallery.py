from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
LOCAL_OLR_PATH = PROJECT_ROOT / "data" / "local" / "olr.day.mean.nc"
LOCAL_U850_PATH = PROJECT_ROOT / "data" / "local" / "uwnd_850hPa_1979-2024.nc"
LOCAL_V850_PATH = PROJECT_ROOT / "data" / "local" / "vwnd_850hPa_1979-2024.nc"
GALLERY_RANGE_MODE = os.environ.get("TWAVE_GALLERY_RANGE_MODE", "demo").strip().lower()
FULL_GALLERY_TIME_RANGE = ("1979-01-01", "2014-12-31")
FAST_GALLERY_TIME_RANGE = ("2005-01-01", "2014-12-31")
STABLE_GALLERY_TIME_RANGE = ("1995-01-01", "2014-12-31")
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from tropical_wave_tools.atlas import generate_local_wave_atlas
from tropical_wave_tools.config import SpectralConfig
from tropical_wave_tools.io import load_dataarray
from tropical_wave_tools.plotting import (
    plot_latlon_field,
    plot_wk_spectrum,
    use_scientific_style,
)
from tropical_wave_tools.preprocess import compute_anomaly
from tropical_wave_tools.sample_data import open_example_olr
from tropical_wave_tools.spectral import analyze_wk_spectrum
from tropical_wave_tools.stats import standard_deviation


def _load_gallery_data(*, time_range: tuple[str, str], lat_range: tuple[float, float] = (-15.0, 15.0)):
    """Load the preferred gallery dataset, falling back to the packaged sample."""
    if LOCAL_OLR_PATH.exists():
        return load_dataarray(
            LOCAL_OLR_PATH,
            variable="olr",
            lat_range=lat_range,
            time_range=time_range,
        )

    data = open_example_olr()
    return data.sel(time=slice(*time_range), lat=slice(*lat_range))


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)


ACTIVE_DOC_ASSETS = {
    "sample_mean_field.png",
    "monthly_anomaly_std.png",
    "wk_spectrum.png",
    "kelvin_hovmoller_triptych.png",
    "er_hovmoller_triptych.png",
    "mjo_hovmoller_triptych.png",
    "mrg_hovmoller_triptych.png",
    "wave_spatial_compare_large_scale.png",
    "wave_spatial_compare_westward.png",
    "case05_seasonal_variance_cycle.png",
    "case05_regional_variance_cycle.png",
    "kelvin_wind_diagnostics_lag0.png",
    "er_wind_diagnostics_lag0.png",
    "mjo_wind_diagnostics_lag0.png",
    "mrg_wind_diagnostics_lag0.png",
    "td_wind_diagnostics_lag0.png",
    "kelvin_eof_modes.png",
    "er_eof_modes.png",
    "mjo_eof_modes.png",
    "mrg_eof_modes.png",
    "td_eof_modes.png",
    "kelvin_lead_lag_evolution.png",
    "er_lead_lag_evolution.png",
    "mjo_lead_lag_evolution.png",
    "mrg_lead_lag_evolution.png",
    "td_lead_lag_evolution.png",
    "multiwave_seasonal_longitude_olr_large_scale.png",
    "multiwave_seasonal_longitude_olr_westward.png",
    "kelvin_paper_hovmoller.png",
    "er_paper_hovmoller.png",
    "mjo_paper_hovmoller.png",
    "mrg_paper_hovmoller.png",
    "td_paper_hovmoller.png",
}

FAST_ATLAS_ASSET_SPECS = (
    ("kelvin/kelvin_hovmoller_triptych.png", "kelvin_hovmoller_triptych.png"),
    ("er/er_hovmoller_triptych.png", "er_hovmoller_triptych.png"),
    ("mjo/mjo_hovmoller_triptych.png", "mjo_hovmoller_triptych.png"),
    ("mrg/mrg_hovmoller_triptych.png", "mrg_hovmoller_triptych.png"),
)

STABLE_ATLAS_ASSET_SPECS = (
    ("multiwave_large_scale_filtered_olr_std.png", "wave_spatial_compare_large_scale.png"),
    ("multiwave_westward_filtered_olr_std.png", "wave_spatial_compare_westward.png"),
    ("case05_seasonal_variance_cycle.png", "case05_seasonal_variance_cycle.png"),
    ("case05_regional_variance_cycle.png", "case05_regional_variance_cycle.png"),
    ("kelvin/kelvin_wind_diagnostics_lag0.png", "kelvin_wind_diagnostics_lag0.png"),
    ("er/er_wind_diagnostics_lag0.png", "er_wind_diagnostics_lag0.png"),
    ("mjo/mjo_wind_diagnostics_lag0.png", "mjo_wind_diagnostics_lag0.png"),
    ("mrg/mrg_wind_diagnostics_lag0.png", "mrg_wind_diagnostics_lag0.png"),
    ("td/td_wind_diagnostics_lag0.png", "td_wind_diagnostics_lag0.png"),
    ("kelvin/kelvin_eof_modes.png", "kelvin_eof_modes.png"),
    ("er/er_eof_modes.png", "er_eof_modes.png"),
    ("mjo/mjo_eof_modes.png", "mjo_eof_modes.png"),
    ("mrg/mrg_eof_modes.png", "mrg_eof_modes.png"),
    ("td/td_eof_modes.png", "td_eof_modes.png"),
    ("kelvin/kelvin_lead_lag_evolution.png", "kelvin_lead_lag_evolution.png"),
    ("er/er_lead_lag_evolution.png", "er_lead_lag_evolution.png"),
    ("mjo/mjo_lead_lag_evolution.png", "mjo_lead_lag_evolution.png"),
    ("mrg/mrg_lead_lag_evolution.png", "mrg_lead_lag_evolution.png"),
    ("td/td_lead_lag_evolution.png", "td_lead_lag_evolution.png"),
    ("multiwave_large_scale_seasonal_longitude_olr.png", "multiwave_seasonal_longitude_olr_large_scale.png"),
    ("multiwave_westward_seasonal_longitude_olr.png", "multiwave_seasonal_longitude_olr_westward.png"),
    ("kelvin/kelvin_paper_hovmoller.png", "kelvin_paper_hovmoller.png"),
    ("er/er_paper_hovmoller.png", "er_paper_hovmoller.png"),
    ("mjo/mjo_paper_hovmoller.png", "mjo_paper_hovmoller.png"),
    ("mrg/mrg_paper_hovmoller.png", "mrg_paper_hovmoller.png"),
    ("td/td_paper_hovmoller.png", "td_paper_hovmoller.png"),
)


def _cleanup_stale_assets(docs_assets: Path) -> None:
    for asset_path in docs_assets.glob("*.png"):
        if asset_path.name not in ACTIVE_DOC_ASSETS:
            asset_path.unlink(missing_ok=True)


def _time_range_label(time_range: tuple[str, str]) -> str:
    return f"{time_range[0][:4]}-{time_range[1][:4]}"


def _copy_asset_specs(source_root: Path, docs_assets: Path, asset_specs: tuple[tuple[str, str], ...]) -> None:
    for source_rel, target_name in asset_specs:
        _copy_if_exists(source_root / source_rel, docs_assets / target_name)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    use_scientific_style()
    range_mode = "full" if GALLERY_RANGE_MODE == "full" else "demo"
    overview_range = FULL_GALLERY_TIME_RANGE if range_mode == "full" else STABLE_GALLERY_TIME_RANGE
    fast_range = FULL_GALLERY_TIME_RANGE if range_mode == "full" else FAST_GALLERY_TIME_RANGE
    stable_range = FULL_GALLERY_TIME_RANGE if range_mode == "full" else STABLE_GALLERY_TIME_RANGE
    docs_assets = PROJECT_ROOT / "docs" / "assets"
    docs_assets.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_assets(docs_assets)

    data = _load_gallery_data(time_range=overview_range, lat_range=(-90.0, 90.0))
    plot_latlon_field(
        data.mean("time"),
        title=f"Mean OLR over {_time_range_label(overview_range)}",
        cmap="Spectral_r",
        integer_colorbar=True,
        save_path=docs_assets / "sample_mean_field.png",
    )

    anomaly = compute_anomaly(data, group="month")
    plot_latlon_field(
        standard_deviation(anomaly, dim="time"),
        title=f"Monthly-anomaly standard deviation ({_time_range_label(overview_range)})",
        cmap="magma",
        integer_colorbar=True,
        zero_floor_colorbar=True,
        save_path=docs_assets / "monthly_anomaly_std.png",
    )

    spectral_source = _load_gallery_data(time_range=overview_range)
    result = analyze_wk_spectrum(
        spectral_source,
        config=SpectralConfig(window_size_days=128, window_skip_days=32),
    )
    figure, _ = plot_wk_spectrum(result, save_path=docs_assets / "wk_spectrum.png")
    plt.close(figure)

    if LOCAL_U850_PATH.exists() and LOCAL_V850_PATH.exists():
        if range_mode == "full":
            atlas_root = PROJECT_ROOT / "outputs" / "docs_gallery_atlas"
            summary = generate_local_wave_atlas(
                output_dir=atlas_root,
                waves=("kelvin", "er", "mjo", "mrg", "td"),
                time_range=FULL_GALLERY_TIME_RANGE,
                lat_range=(-25.0, 30.0),
                hovmoller_days=240,
                n_workers=1,
            )
            print(summary.to_string(index=False))
            _copy_asset_specs(atlas_root, docs_assets, FAST_ATLAS_ASSET_SPECS + STABLE_ATLAS_ASSET_SPECS)
        else:
            atlas_fast_root = PROJECT_ROOT / "outputs" / "docs_gallery_atlas_fast10"
            fast_summary = generate_local_wave_atlas(
                output_dir=atlas_fast_root,
                waves=("kelvin", "er", "mjo", "mrg"),
                time_range=fast_range,
                lat_range=(-25.0, 30.0),
                hovmoller_days=240,
                n_workers=1,
            )
            print(f"[fast demo {fast_range[0]} to {fast_range[1]}]")
            print(fast_summary.to_string(index=False))
            _copy_asset_specs(atlas_fast_root, docs_assets, FAST_ATLAS_ASSET_SPECS)

            atlas_stable_root = PROJECT_ROOT / "outputs" / "docs_gallery_atlas_stable20"
            stable_summary = generate_local_wave_atlas(
                output_dir=atlas_stable_root,
                waves=("kelvin", "er", "mjo", "mrg", "td"),
                time_range=stable_range,
                lat_range=(-25.0, 30.0),
                hovmoller_days=240,
                n_workers=1,
            )
            print(f"[stable demo {stable_range[0]} to {stable_range[1]}]")
            print(stable_summary.to_string(index=False))
            _copy_asset_specs(atlas_stable_root, docs_assets, STABLE_ATLAS_ASSET_SPECS)
        _cleanup_stale_assets(docs_assets)


if __name__ == "__main__":
    main()
