"""Command-line interface for the project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.io import load_dataarray, save_dataarray
from tropical_wave_tools.sample_data import get_sample_path
from tropical_wave_tools.spectral import SpectralConfig
from tropical_wave_tools.workflows import (
    analyze_wk_spectrum_from_file,
    compare_filter_spatial_fields,
    create_demo_subset,
    install_local_data_copy,
)


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
    compare_parser.add_argument("--lat-min", type=float, default=-25.0, help="Minimum latitude.")
    compare_parser.add_argument("--lat-max", type=float, default=25.0, help="Maximum latitude.")
    compare_parser.add_argument("--spd", type=int, default=1, help="Samples per day.")
    compare_parser.add_argument("--n-harm", type=int, default=3, help="Annual-cycle harmonics.")
    compare_parser.add_argument("--n-jobs", type=int, default=-1, help="Legacy filter jobs.")
    compare_parser.add_argument("--n-workers", type=int, default=4, help="CCKW workers.")
    compare_parser.add_argument("--output-dir", required=True, help="Output directory.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]

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
        time_range = None
        if args.time_start and args.time_end:
            time_range = (args.time_start, args.time_end)
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
        time_range = None
        if args.time_start and args.time_end:
            time_range = (args.time_start, args.time_end)
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
        summary = compare_filter_spatial_fields(
            args.input,
            variable=args.var,
            waves=args.waves,
            time_range=(args.time_start, args.time_end),
            lat_range=(args.lat_min, args.lat_max),
            spd=args.spd,
            n_harm=args.n_harm,
            n_jobs=args.n_jobs,
            n_workers=args.n_workers,
            output_dir=args.output_dir,
        )
        print(summary.to_string(index=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
