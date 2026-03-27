from __future__ import annotations

from pathlib import Path

from tropical_wave_tools import open_example_olr
from tropical_wave_tools.plotting import plot_wk_spectrum
from tropical_wave_tools.spectral import analyze_wk_spectrum


def main() -> None:
    data = open_example_olr()
    result = analyze_wk_spectrum(data)
    output_dir = Path("outputs/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_wk_spectrum(result, save_path=output_dir / "wk_spectrum_demo.png")
    result.to_dataset().to_netcdf(output_dir / "wk_spectrum_demo.nc")


if __name__ == "__main__":
    main()

