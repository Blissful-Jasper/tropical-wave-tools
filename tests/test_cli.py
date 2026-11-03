from __future__ import annotations

from pathlib import Path

from tropical_wave_tools.cli import build_parser


def test_cli_wk_parser() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "wk-spectrum",
            "--input",
            "sample.nc",
            "--output-dir",
            "outputs/wk",
        ]
    )
    assert args.command == "wk-spectrum"
    assert args.var == "olr"


def test_cli_local_wave_atlas_parser() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "local-wave-atlas",
            "--output-dir",
            "outputs/local_atlas",
        ]
    )
    assert args.command == "local-wave-atlas"
    assert Path(args.olr).is_absolute()
    assert Path(args.olr).name == "olr.day.mean.nc"


def test_cli_kelvin_events_parser_and_run(tmp_path) -> None:
    from tropical_wave_tools.cli import main
    from tropical_wave_tools.sample_data import get_sample_path

    parser = build_parser()
    args = parser.parse_args(["kelvin-events", "--olr", "x.nc", "--output-dir", "out"])
    assert args.command == "kelvin-events"
    assert args.peak_sigma == 1.5 and args.depth == 25.0

    output = tmp_path / "kelvin"
    status = main(
        [
            "kelvin-events",
            "--olr",
            str(get_sample_path()),
            "--lat-min",
            "-15",
            "--lat-max",
            "15",
            "--output-dir",
            str(output),
        ]
    )
    assert status == 0
    assert (output / "events.csv").exists()
