from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from tropical_wave_tools import open_example_olr
from tropical_wave_tools.filters import filter_wave_signal
from tropical_wave_tools.plotting import (
    plot_hovmoller_comparison,
    plot_latlon_field,
    plot_wk_spectrum,
)
from tropical_wave_tools.preprocess import compute_anomaly
from tropical_wave_tools.spectral import analyze_wk_spectrum
from tropical_wave_tools.stats import standard_deviation


def main() -> None:
    output_dir = Path("outputs/examples/olr_showcase")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = open_example_olr()

    mean_figure, _ = plot_latlon_field(
        data.mean("time"),
        title="Mean OLR over the sample period",
        cmap="Spectral_r",
        save_path=output_dir / "sample_mean_field.png",
    )
    plt.close(mean_figure)

    std_figure, _ = plot_latlon_field(
        standard_deviation(compute_anomaly(data, group="month"), dim="time"),
        title="Monthly-anomaly standard deviation",
        cmap="magma",
        save_path=output_dir / "monthly_anomaly_std.png",
    )
    plt.close(std_figure)

    wk_figure, _ = plot_wk_spectrum(
        analyze_wk_spectrum(data),
        save_path=output_dir / "wk_spectrum.png",
    )
    plt.close(wk_figure)

    kelvin = filter_wave_signal(data, wave_name="kelvin", method="cckw", n_workers=1)
    kelvin.to_netcdf(output_dir / "kelvin_filtered.nc")

    hovmoller_figure, _ = plot_hovmoller_comparison(
        compute_anomaly(data, group="month").sel(lat=slice(-5, 5)).mean("lat"),
        kelvin.sel(lat=slice(-5, 5)).mean("lat"),
        left_title="Equatorial OLR anomaly",
        right_title="Kelvin-filtered signal",
        colorbar_label="OLR anomaly",
        save_path=output_dir / "kelvin_hovmoller.png",
    )
    plt.close(hovmoller_figure)


if __name__ == "__main__":
    main()
