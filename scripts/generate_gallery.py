from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tropical_wave_tools.filters import CCKWFilter, WaveFilter
from tropical_wave_tools.plotting import plot_spatial_std_comparison, plot_wk_spectrum
from tropical_wave_tools.sample_data import open_example_olr
from tropical_wave_tools.spectral import analyze_wk_spectrum


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    docs_assets = PROJECT_ROOT / "docs" / "assets"
    docs_assets.mkdir(parents=True, exist_ok=True)

    data = open_example_olr()
    result = analyze_wk_spectrum(data)
    figure, _ = plot_wk_spectrum(result, save_path=docs_assets / "wk_spectrum.png")
    plt.close(figure)

    legacy = WaveFilter().extract_wave_signal(data, wave_name="kelvin", use_parallel=False)
    cckw = CCKWFilter(ds=data, wave_name="kelvin", n_workers=1, verbose=False).process()
    figure, _ = plot_spatial_std_comparison(
        legacy.std("time"),
        cckw.std("time"),
        wave_name="kelvin",
        title_suffix="(sample data)",
        save_path=docs_assets / "kelvin_std_compare.png",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
