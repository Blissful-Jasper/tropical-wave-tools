from __future__ import annotations

import argparse
from pathlib import Path

from tropical_wave_tools.sample_data import get_sample_path
from tropical_wave_tools.workflows import create_demo_subset, install_local_data_copy


def main() -> None:
    parser = argparse.ArgumentParser(description="Create packaged and local sample data files.")
    parser.add_argument("--source", required=True, help="Source NetCDF file.")
    parser.add_argument("--variable", default="olr", help="Variable name.")
    parser.add_argument("--time-start", default="1979-01-01", help="Subset start date.")
    parser.add_argument("--time-end", default="1979-12-31", help="Subset end date.")
    parser.add_argument("--lat-min", type=float, default=-15.0, help="Minimum latitude.")
    parser.add_argument("--lat-max", type=float, default=15.0, help="Maximum latitude.")
    parser.add_argument("--copy-full-data", action="store_true", help="Copy the full file to data/local.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sample_path = get_sample_path()
    create_demo_subset(
        args.source,
        sample_path,
        variable=args.variable,
        time_range=(args.time_start, args.time_end),
        lat_range=(args.lat_min, args.lat_max),
    )
    print(f"Created packaged sample: {sample_path}")

    if args.copy_full_data:
        copied = install_local_data_copy(args.source, project_root / "data" / "local")
        print(f"Copied full data file: {copied}")


if __name__ == "__main__":
    main()
