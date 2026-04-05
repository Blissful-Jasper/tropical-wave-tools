from __future__ import annotations

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
    assert args.olr.endswith("olr.day.mean.nc")
