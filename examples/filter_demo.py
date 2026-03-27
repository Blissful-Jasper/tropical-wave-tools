from __future__ import annotations

from pathlib import Path

from tropical_wave_tools import open_example_olr
from tropical_wave_tools.filters import filter_wave_signal


def main() -> None:
    data = open_example_olr()
    filtered = filter_wave_signal(data, wave_name="kelvin", method="cckw")
    output_dir = Path("outputs/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_netcdf(output_dir / "kelvin_filtered_demo.nc")


if __name__ == "__main__":
    main()

