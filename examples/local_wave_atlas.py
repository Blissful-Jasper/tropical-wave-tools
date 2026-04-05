"""Generate an OLR + low-level wind equatorial wave atlas from local data."""

from __future__ import annotations

from pathlib import Path

from tropical_wave_tools.atlas import generate_local_wave_atlas


if __name__ == "__main__":
    summary = generate_local_wave_atlas(
        output_dir=Path("outputs/local_wave_atlas"),
        waves=("kelvin", "er", "mrg", "eig", "wig", "td", "mjo"),
        time_range=("1979-01-01", "1984-12-31"),
        n_workers=1,
    )
    print(summary.to_string(index=False))
