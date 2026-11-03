"""Command-line interface for the project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from tropical_wave_tools.atlas import DEFAULT_LOCAL_PATHS, generate_local_wave_atlas
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
    wk_parser.add_argument(
        "--skip-days",
        type=int,
        default=30,
        help="Window advance in days between segment starts (30 days gives about two months overlap for a 96-day window).",
    )
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
    atlas_parser.add_argument("--olr", default=str(DEFAULT_LOCAL_PATHS["olr"]), help="Local OLR NetCDF file.")
    atlas_parser.add_argument(
        "--u850",
        default=str(DEFAULT_LOCAL_PATHS["u850"]),
        help="Local 850 hPa zonal-wind NetCDF file.",
    )
    atlas_parser.add_argument(
        "--v850",
        default=str(DEFAULT_LOCAL_PATHS["v850"]),
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


    kelvin_parser = subparsers.add_parser(
        "kelvin-events",
        help="Identify Kelvin-wave events from wind (dynamical) and OLR (convective) channels.",
    )
    kelvin_parser.add_argument("--olr", default=None, help="OLR NetCDF file (convective channel).")
    kelvin_parser.add_argument("--olr-var", default="olr", help="OLR variable name.")
    kelvin_parser.add_argument("--u", default=None, help="Zonal-wind NetCDF file (dynamical channel).")
    kelvin_parser.add_argument("--u-var", default="u", help="Zonal-wind variable name.")
    kelvin_parser.add_argument("--z", default=None, help="Optional geopotential-height file (exact Matsuno projection).")
    kelvin_parser.add_argument("--z-var", default="z", help="Geopotential-height variable name (metres).")
    kelvin_parser.add_argument("--v", default=None, help="Optional meridional-wind file (v/u diagnostic only).")
    kelvin_parser.add_argument("--v-var", default="v", help="Meridional-wind variable name.")
    kelvin_parser.add_argument("--time-start", default=None, help="Start of the analysis window.")
    kelvin_parser.add_argument("--time-end", default=None, help="End of the analysis window.")
    kelvin_parser.add_argument("--buffer-days", type=int, default=30, help="Extra days read on each side to keep filter edge effects out of the window.")
    kelvin_parser.add_argument("--lat-min", type=float, default=-20.0, help="Minimum latitude (keep symmetric about 0).")
    kelvin_parser.add_argument("--lat-max", type=float, default=20.0, help="Maximum latitude.")
    kelvin_parser.add_argument("--level", type=float, default=None, help="Pressure level (hPa) to select if files have a level dimension, e.g. 850.")
    kelvin_parser.add_argument("--regrid-deg", type=float, default=None, help="Regrid to a global symmetric grid of this resolution (2.5 recommended for 0.25-degree ERA5).")
    kelvin_parser.add_argument("--z-geopotential", action="store_true", help="--z is geopotential (m2 s-2, as ERA5 'z'); divide by g to get height.")
    kelvin_parser.add_argument("--olr-factor", type=float, default=1.0, help="Multiply the OLR variable by this factor (e.g. -1/3600 for hourly-accumulated ERA5 ttr; verify accumulation convention).")
    kelvin_parser.add_argument("--region-lon", type=float, nargs=2, metavar=("MIN", "MAX"), default=None, help="Report only events passing through this longitude window (deg E; wrap-around allowed), e.g. 120 280 for the Pacific. Detection still uses the global grid.")
    kelvin_parser.add_argument("--min-region-points", type=int, default=3, help="Minimum tracked days inside --region-lon.")
    kelvin_parser.add_argument("--keep-seasonal-cycle", action="store_true", help="Do not remove the seasonal cycle (only for records shorter than a year).")
    kelvin_parser.add_argument("--no-figures", action="store_true", help="Skip the report figure.")
    kelvin_parser.add_argument("--spd", type=int, default=1, help="Samples per day.")
    kelvin_parser.add_argument("--n-harm", type=int, default=3, help="Annual-cycle harmonics removed before filtering.")
    kelvin_parser.add_argument("--depth", type=float, default=25.0, help="Projection equivalent depth (m).")
    kelvin_parser.add_argument("--peak-sigma", type=float, default=1.5, help="Peak threshold in standard deviations.")
    kelvin_parser.add_argument("--continue-sigma", type=float, default=0.75, help="Track-continuation threshold in standard deviations.")
    kelvin_parser.add_argument("--dyn-min-amplitude", type=float, default=None, help="Absolute minimum dynamical amplitude (m/s).")
    kelvin_parser.add_argument("--conv-min-amplitude", type=float, default=None, help="Absolute minimum convective amplitude (W m-2).")
    kelvin_parser.add_argument("--axis-lat-min", type=float, default=-15.0, help="Southern bound of the convective-axis search (sampling only; widen it, do not assume a hemisphere).")
    kelvin_parser.add_argument("--axis-lat-max", type=float, default=20.0, help="Northern bound of the convective-axis search.")
    kelvin_parser.add_argument("--output-dir", required=True, help="Output directory (events.csv, tracks.csv, indices.nc).")

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

    if args.command == "kelvin-events":
        from tropical_wave_tools.kelvin_events import KelvinEventConfig
        from tropical_wave_tools.kelvin_workflow import load_reanalysis_field, run_regional_workflow

        if args.olr is None and args.u is None:
            parser.error("kelvin-events needs --olr and/or --u.")
        time_range = _resolve_time_range(args.time_start, args.time_end)
        lat_range = (args.lat_min, args.lat_max)

        def _load(path: Optional[str], variable: str, **extra):
            if path is None:
                return None
            return load_reanalysis_field(
                path,
                variable,
                level=args.level,
                lat_range=lat_range,
                time_range=time_range,
                buffer_days=args.buffer_days,
                regrid_deg=args.regrid_deg,
                **extra,
            )

        config = KelvinEventConfig(
            obs_per_day=args.spd,
            n_harm=args.n_harm,
            remove_seasonal_cycle=not args.keep_seasonal_cycle,
            projection_depth=args.depth,
            peak_sigma=args.peak_sigma,
            continue_sigma=args.continue_sigma,
            dyn_min_amplitude=args.dyn_min_amplitude,
            conv_min_amplitude=args.conv_min_amplitude,
            axis_lat_range=(args.axis_lat_min, args.axis_lat_max),
        )
        result = run_regional_workflow(
            u=_load(args.u, args.u_var),
            z=_load(args.z, args.z_var, geopotential=args.z_geopotential),
            v=_load(args.v, args.v_var),
            olr=_load(args.olr, args.olr_var, factor=args.olr_factor),
            config=config,
            region_lon=tuple(args.region_lon) if args.region_lon else None,
            time_range=time_range,
            min_points=args.min_region_points,
            output_dir=args.output_dir,
            make_figures=not args.no_figures,
        )
        print(f"Kelvin events written to: {args.output_dir}")
        print(result.catalog.summary().to_string())
        if not result.summary.empty:
            print(result.summary.round(2).to_string(index=False))
        if result.structure is not None and not result.structure.empty:
            print("\nConvective meridional-structure classes:")
            print(result.structure.round(2).to_string(index=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
