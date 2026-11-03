"""Dual-channel Kelvin-wave event identification on a synthetic case.

The case reproduces the Pacific situation: Kelvin-wave circulation symmetric
about the equator, convection displaced to 8N.  No real data are needed.

Run from the repository root::

    python examples/kelvin_event_demo.py [output.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tropical_wave_tools.kelvin_events import (
    KelvinEventConfig,
    event_composite,
    identify_kelvin_events,
)
from tropical_wave_tools.synthetic import make_synthetic_kelvin_case


def main(output: str = "kelvin_events_demo.png") -> None:
    case = make_synthetic_kelvin_case()
    config = KelvinEventConfig(
        remove_seasonal_cycle=False,
        axis_window_days=61,
        peak_sigma=3.0,
        continue_sigma=1.5,
        dyn_min_amplitude=1.5,
        conv_min_amplitude=3.0,
    )
    catalog = identify_kelvin_events(u=case.u, z=case.z, v=case.v, olr=case.olr, config=config)
    show = [
        "event_id", "kind", "start_time", "lifetime_days", "speed_ms", "peak_amp",
        "coupling_corr", "coupling_pvalue", "conv_offset_deg", "conv_axis_lat",
        "conv_peak", "conv_peak_fixed_band", "u_symmetry_index", "v_to_u_ratio",
    ]
    print(catalog.events[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    time = catalog.indices["kelvin_dynamical"].time.values
    lon = catalog.indices["kelvin_dynamical"].lon.values
    dyn_tracks = catalog.tracks[catalog.tracks["channel"] == "dynamical"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    panels = [
        ("kelvin_dynamical", "(a) Dynamical: Kelvin projection of u,z (m/s)", 4.0),
        ("kelvin_convective_axis", "(b) Convective: -OLR_K along convective axis", 8.0),
    ]
    for ax, (name, title, vmax) in zip(axes[0], panels):
        field = catalog.indices[name]
        mesh = ax.pcolormesh(lon, time, field.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        for _, group in dyn_tracks.groupby("event_id"):
            ax.plot(group["lon"], group["time"], ".", color="k", ms=3)
        ax.set(title=title, xlabel="Longitude (deg E)")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    axes[0, 0].set_ylabel("Time")

    ax = axes[1, 0]
    fixed = catalog.indices["kelvin_convective_fixed_band"]
    mesh = ax.pcolormesh(lon, time, fixed.values, cmap="RdBu_r", vmin=-8.0, vmax=8.0, shading="auto")
    for _, group in dyn_tracks.groupby("event_id"):
        ax.plot(group["lon"], group["time"], ".", color="k", ms=3)
    ax.set(title="(c) Same, but fixed 5S-5N band (same colour scale as b)", xlabel="Longitude (deg E)", ylabel="Time")
    fig.colorbar(mesh, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    u_comp = event_composite(case.u, catalog.events, kinds=("coupled",), lags_days=(0.0,), half_width_deg=40.0)
    o_comp = event_composite(case.olr - 230.0, catalog.events, kinds=("coupled",), lags_days=(0.0,), half_width_deg=40.0)
    u_amp = np.abs(u_comp.isel(lag=0)).max("rel_lon")
    o_ext = (-o_comp.isel(lag=0)).max("rel_lon")
    ax.plot(u_amp.lat, u_amp / u_amp.max(), color="#2a6fb0", lw=2, label="|u| (circulation)")
    ax.plot(o_ext.lat, o_ext / o_ext.max(), color="#c4561d", lw=2, label="-OLR (convection)")
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set(title="(d) Event composite, latitude profile of peak anomaly", xlabel="Latitude", ylabel="Normalised peak")
    ax.legend(frameon=False)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
