"""High-level workflows and reusable command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from tropical_wave_tools.filters import CCKWFilter, WaveFilter
from tropical_wave_tools.io import describe_dataarray, load_dataarray, save_dataarray, save_dataset
from tropical_wave_tools.plotting import plot_spatial_std_comparison, plot_wk_spectrum
from tropical_wave_tools.sample_data import copy_full_example_data
from tropical_wave_tools.spectral import SpectralConfig, WKAnalysisResult, analyze_wk_spectrum


def create_demo_subset(
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    variable: str = "olr",
    time_range: Tuple[str, str] = ("1979-01-01", "1979-12-31"),
    lat_range: Tuple[float, float] = (-15.0, 15.0),
) -> Path:
    """Create a lightweight equatorial OLR sample subset for tests and docs."""
    data = load_dataarray(
        source_path,
        variable=variable,
        time_range=time_range,
        lat_range=lat_range,
    )
    dataset = data.to_dataset(name=variable)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)
    return output


def install_local_data_copy(source_path: Union[str, Path], target_dir: Union[str, Path]) -> Path:
    """Copy the full local source file into the project-local data folder."""
    return copy_full_example_data(source_path, target_dir)


def analyze_wk_spectrum_from_file(
    input_path: Union[str, Path],
    *,
    variable: str = "olr",
    lat_range: Tuple[float, float] = (-15.0, 15.0),
    time_range: Optional[Tuple[str, str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[SpectralConfig] = None,
) -> Tuple[WKAnalysisResult, Dict[str, object]]:
    """Load data, compute a WK spectrum, and optionally save outputs."""
    data = load_dataarray(
        input_path,
        variable=variable,
        lat_range=lat_range,
        time_range=time_range,
    )
    result = analyze_wk_spectrum(data, config=config)
    summary = describe_dataarray(data)

    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        save_dataset(result.to_dataset(), output_root / "wk_spectrum.nc")
        figure, _ = plot_wk_spectrum(result, save_path=output_root / "wk_spectrum.png")
        plt.close(figure)

    return result, summary


def compute_spatial_metrics(
    legacy_std: xr.DataArray,
    cckw_std: xr.DataArray,
) -> Dict[str, float]:
    """Compute summary statistics for legacy-vs-CCKW comparisons."""
    diff = legacy_std - cckw_std
    legacy_values = legacy_std.values.ravel()
    cckw_values = cckw_std.values.ravel()
    diff_values = diff.values.ravel()
    valid = pd.notna(legacy_values) & pd.notna(cckw_values)

    if not valid.any():
        return {
            "legacy_mean_std": float("nan"),
            "cckw_mean_std": float("nan"),
            "mean_bias": float("nan"),
            "mean_abs_diff": float("nan"),
            "rmse": float("nan"),
            "pattern_corr": float("nan"),
        }

    return {
        "legacy_mean_std": float(legacy_std.mean().item()),
        "cckw_mean_std": float(cckw_std.mean().item()),
        "mean_bias": float(diff_values[valid].mean()),
        "mean_abs_diff": float(abs(diff_values[valid]).mean()),
        "rmse": float((diff_values[valid] ** 2).mean() ** 0.5),
        "pattern_corr": float(pd.Series(legacy_values[valid]).corr(pd.Series(cckw_values[valid]))),
    }


def compare_filter_spatial_fields(
    input_path: Union[str, Path],
    *,
    variable: str = "olr",
    waves: Sequence[str] = ("kelvin", "mjo"),
    time_range: Tuple[str, str] = ("1979-01-01", "1981-12-31"),
    lat_range: Tuple[float, float] = (-25.0, 25.0),
    spd: int = 1,
    n_harm: int = 3,
    n_jobs: int = -1,
    n_workers: int = 4,
    use_parallel: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Run legacy and CCKW filters on the same dataset and compare their spatial STD."""
    data = load_dataarray(
        input_path,
        variable=variable,
        time_range=time_range,
        lat_range=lat_range,
    )
    output_root = Path(output_dir) if output_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Union[float, str]]] = []
    for wave_name in waves:
        legacy = WaveFilter().extract_wave_signal(
            data,
            wave_name=wave_name,
            obs_per_day=spd,
            use_parallel=use_parallel,
            n_jobs=n_jobs,
            n_harm=n_harm,
        )
        cckw = CCKWFilter(
            ds=data,
            wave_name=wave_name,
            units=data.attrs.get("units"),
            spd=spd,
            n_workers=n_workers,
            n_harm=n_harm,
            verbose=False,
        ).process()

        legacy_std = legacy.std("time").rename(f"{wave_name}_legacy_std")
        cckw_std = cckw.std("time").rename(f"{wave_name}_cckw_std")

        metrics = compute_spatial_metrics(legacy_std, cckw_std)
        metrics["wave"] = wave_name
        rows.append(metrics)

        if output_root is not None:
            plot_spatial_std_comparison(
                legacy_std,
                cckw_std,
                wave_name=wave_name,
                title_suffix=f"({time_range[0]} to {time_range[1]})",
                save_path=output_root / f"{wave_name}_spatial_std_compare.png",
            )
            save_dataset(
                xr.Dataset(
                    {
                        f"{wave_name}_legacy_std": legacy_std,
                        f"{wave_name}_cckw_std": cckw_std,
                        f"{wave_name}_legacy_minus_cckw": (legacy_std - cckw_std).rename(
                            f"{wave_name}_legacy_minus_cckw"
                        ),
                    }
                ),
                output_root / f"{wave_name}_spatial_std_comparison.nc",
            )

    summary = pd.DataFrame(rows)
    if output_root is not None:
        summary.to_csv(output_root / "wave_filter_spatial_comparison_summary.csv", index=False)
    return summary
