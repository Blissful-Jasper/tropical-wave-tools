"""Command-line interface for the project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.io import load_dataarray, save_dataarray
from tropical_wave_tools.sample_data import get_sample_path
from tropical_wave_tools.spectral import SpectralConfig
from tropical_wave_tools.atlas import generate_local_wave_atlas
from tropical_wave_tools.workflows import (
    analyze_wk_spectrum_from_file,
    compare_filter_spatial_fields,
    create_demo_subset,
    install_local_data_copy,
)


def _resolve_time_range(
    time_start: Optional[str],
    time_end: Optional[str],
    demo_years: Optional[int] = None,
) -> Optional[tuple[str, str]]:
    """Resolve one explicit or demo-style time range."""
    if time_start is None or time_end is None:
        if demo_years is not None:
            raise ValueError("`demo_years` requires both `time_start` and `time_end`.")
        return None

    if demo_years is None:
        return (time_start, time_end)

    end_year = int(str(time_end)[:4])
    demo_start = f"{end_year - int(demo_years) + 1:04d}-01-01"
    if demo_start < time_start:
        demo_start = time_start
    return (demo_start, time_end)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="tropical-wave-tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("prepare-sample-data", help="Create packaged and local demo data.")
    sample_parser.add_argument("--source", required=True, help="Source NetCDF file.")
    sample_parser.add_argument("--variable", default="olr", help="Variable name.")
    sample_parser.add_argument("--time-start", default="1979-01-01", help="Subset start date.")
    sample_parser.add_argument("--time-end", default="1979-12-31", help="Subset end date.")
    sample_parser.add_argument("--lat-min", type=float, default=-15.0, help="Minimum latitude.")
    sample_parser.add_argument("--lat-max", type=float, default=15.0, help="Maximum latitude.")
    sample_parser.add_argument(
        "--copy-full-data",
        action="store_true",
        help="Also copy the full source file into data/local/.",
    )

    wk_parser = subparsers.add_parser("wk-spectrum", help="Compute a Wheeler-Kiladis spectrum.")
    wk_parser.add_argument("--input", required=True, help="Input NetCDF file.")
    wk_parser.add_argument("--var", default="olr", help="Variable name.")
    wk_parser.add_argument("--time-start", default=None, help="Start date.")
    wk_parser.add_argument("--time-end", default=None, help="End date.")
    wk_parser.add_argument("--demo-years", type=int, choices=(10, 20), default=None, help="Trim to the last 10 or 20 years within the requested time window.")
    wk_parser.add_argument("--lat-min", type=float, default=-15.0, help="Minimum latitude.")
    wk_parser.add_argument("--lat-max", type=float, default=15.0, help="Maximum latitude.")
    wk_parser.add_argument("--window-days", type=int, default=96, help="Window size in days.")
    wk_parser.add_argument("--skip-days", type=int, default=30, help="Window skip in days.")
    wk_parser.add_argument("--output-dir", required=True, help="Output directory.")

    filter_parser = subparsers.add_parser("filter-wave", help="Extract one wave signal from a dataset.")
    filter_parser.add_argument("--input", required=True, help="Input NetCDF file.")
    filter_parser.add_argument("--var", default="olr", help="Variable name.")
    filter_parser.add_argument("--wave", required=True, help="Wave name.")
    filter_parser.add_argument("--method", choices=("legacy", "cckw"), default="cckw")
    filter_parser.add_argument("--time-start", default=None, help="Start date.")
    filter_parser.add_argument("--time-end", default=None, help="End date.")
    filter_parser.add_argument("--demo-years", type=int, choices=(10, 20), default=None, help="Trim to the last 10 or 20 years within the requested time window.")
    filter_parser.add_argument("--lat-min", type=float, default=-15.0, help="Minimum latitude.")
    filter_parser.add_argument("--lat-max", type=float, default=15.0, help="Maximum latitude.")
    filter_parser.add_argument("--spd", type=int, default=1, help="Samples per day.")
    filter_parser.add_argument("--n-harm", type=int, default=3, help="Annual-cycle harmonics.")
    filter_parser.add_argument("--n-workers", type=int, default=4, help="CCKW workers.")
    filter_parser.add_argument("--n-jobs", type=int, default=-1, help="Legacy parallel jobs.")
    filter_parser.add_argument("--output", required=True, help="Output NetCDF path.")

    compare_parser = subparsers.add_parser("compare-filters", help="Compare legacy and CCKW filters.")
    compare_parser.add_argument("--input", required=True, help="Input NetCDF file.")
    compare_parser.add_argument("--var", default="olr", help="Variable name.")
    compare_parser.add_argument("--waves", nargs="+", default=["kelvin", "mjo"], help="Wave names.")
    compare_parser.add_argument("--time-start", default="1979-01-01", help="Start date.")
    compare_parser.add_argument("--time-end", default="1981-12-31", help="End date.")
    compare_parser.add_argument("--demo-years", type=int, choices=(10, 20), default=None, help="Trim to the last 10 or 20 years within the requested time window.")
    compare_parser.add_argument("--lat-min", type=float, default=-25.0, help="Minimum latitude.")
    compare_parser.add_argument("--lat-max", type=float, default=25.0, help="Maximum latitude.")
    compare_parser.add_argument("--spd", type=int, default=1, help="Samples per day.")
    compare_parser.add_argument("--n-harm", type=int, default=3, help="Annual-cycle harmonics.")
    compare_parser.add_argument("--n-jobs", type=int, default=-1, help="Legacy filter jobs.")
    compare_parser.add_argument("--n-workers", type=int, default=4, help="CCKW workers.")
    compare_parser.add_argument("--output-dir", required=True, help="Output directory.")

    atlas_parser = subparsers.add_parser(
        "local-wave-atlas",
        help="Generate a publication-style OLR/U850/V850 equatorial wave atlas from data/local.",
    )
    atlas_parser.add_argument("--output-dir", required=True, help="Atlas output directory.")
    atlas_parser.add_argument("--olr", default="data/local/olr.day.mean.nc", help="Local OLR NetCDF file.")
    atlas_parser.add_argument(
        "--u850",
        default="data/local/uwnd_850hPa_1979-2024.nc",
        help="Local 850 hPa zonal-wind NetCDF file.",
    )
    atlas_parser.add_argument(
        "--v850",
        default="data/local/vwnd_850hPa_1979-2024.nc",
        help="Local 850 hPa meridional-wind NetCDF file.",
    )
    atlas_parser.add_argument("--waves", nargs="+", default=None, help="Wave names. Defaults to all supported filters.")
    atlas_parser.add_argument("--time-start", default="1979-01-01", help="Start date.")
    atlas_parser.add_argument("--time-end", default="2014-12-31", help="End date.")
    atlas_parser.add_argument("--demo-years", type=int, choices=(10, 20), default=None, help="Trim to the last 10 or 20 years within the requested time window.")
    atlas_parser.add_argument("--lat-min", type=float, default=-25.0, help="Minimum latitude.")
    atlas_parser.add_argument("--lat-max", type=float, default=25.0, help="Maximum latitude.")
    atlas_parser.add_argument("--eq-lat-min", type=float, default=-5.0, help="Equatorial-band minimum latitude.")
    atlas_parser.add_argument("--eq-lat-max", type=float, default=5.0, help="Equatorial-band maximum latitude.")
    atlas_parser.add_argument("--lon-ref", type=float, default=180.0, help="Reference longitude for event composites.")
    atlas_parser.add_argument("--hov-days", type=int, default=180, help="Days shown in the representative Hovmoller window.")
    atlas_parser.add_argument("--event-threshold-std", type=float, default=1.0, help="Event threshold in units of standard deviation.")
    atlas_parser.add_argument("--event-min-spacing", type=int, default=7, help="Minimum spacing between events in days.")
    atlas_parser.add_argument("--n-harm", type=int, default=3, help="Annual-cycle harmonics removed before filtering.")
    atlas_parser.add_argument("--n-workers", type=int, default=4, help="CCKW worker count.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    if getattr(args, "demo_years", None) is not None and (
        getattr(args, "time_start", None) is None or getattr(args, "time_end", None) is None
    ):
        parser.error("--demo-years requires both --time-start and --time-end.")

    if args.command == "prepare-sample-data":
        sample_target = get_sample_path()
        create_demo_subset(
            args.source,
            sample_target,
            variable=args.variable,
            time_range=(args.time_start, args.time_end),
            lat_range=(args.lat_min, args.lat_max),
        )
        print(f"Created packaged sample: {sample_target}")
        if args.copy_full_data:
            copied_path = install_local_data_copy(args.source, project_root / "data" / "local")
            print(f"Copied full local file: {copied_path}")
        return 0

    if args.command == "wk-spectrum":
        time_range = _resolve_time_range(args.time_start, args.time_end, getattr(args, "demo_years", None))
        config = SpectralConfig(
            window_size_days=args.window_days,
            window_skip_days=args.skip_days,
        )
        _, summary = analyze_wk_spectrum_from_file(
            args.input,
            variable=args.var,
            lat_range=(args.lat_min, args.lat_max),
            time_range=time_range,
            output_dir=args.output_dir,
            config=config,
        )
        print(f"WK spectrum written to: {args.output_dir}")
        print(f"Input summary: {summary}")
        return 0

    if args.command == "filter-wave":
        time_range = _resolve_time_range(args.time_start, args.time_end, getattr(args, "demo_years", None))
        data = load_dataarray(
            args.input,
            variable=args.var,
            lat_range=(args.lat_min, args.lat_max),
            time_range=time_range,
        )
        filtered = filter_wave_signal(
            data,
            wave_name=args.wave,
            method=args.method,
            obs_per_day=args.spd,
            n_harm=args.n_harm,
            n_workers=args.n_workers,
            n_jobs=args.n_jobs,
        )
        output_path = save_dataarray(filtered, args.output)
        print(f"Saved filtered data: {output_path}")
        return 0

    if args.command == "compare-filters":
        time_range = _resolve_time_range(args.time_start, args.time_end, getattr(args, "demo_years", None))
        summary = compare_filter_spatial_fields(
            args.input,
            variable=args.var,
            waves=args.waves,
            time_range=time_range,
            lat_range=(args.lat_min, args.lat_max),
            spd=args.spd,
            n_harm=args.n_harm,
            n_jobs=args.n_jobs,
            n_workers=args.n_workers,
            output_dir=args.output_dir,
        )
        print(summary.to_string(index=False))
        return 0

    if args.command == "local-wave-atlas":
        time_range = _resolve_time_range(args.time_start, args.time_end, getattr(args, "demo_years", None))
        summary = generate_local_wave_atlas(
            output_dir=args.output_dir,
            olr_path=args.olr,
            u850_path=args.u850,
            v850_path=args.v850,
            waves=args.waves,
            time_range=time_range,
            lat_range=(args.lat_min, args.lat_max),
            lat_band=(args.eq_lat_min, args.eq_lat_max),
            lon_ref=args.lon_ref,
            hovmoller_days=args.hov_days,
            event_threshold_std=args.event_threshold_std,
            event_min_spacing_days=args.event_min_spacing,
            n_harm=args.n_harm,
            n_workers=args.n_workers,
        )
        print(summary.to_string(index=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
