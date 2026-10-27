"""Write the single-latitude input shared by the NCL and Python filter runs.

Usage: python validation/make_input.py
"""

from __future__ import annotations

from pathlib import Path

from tropical_wave_tools.sample_data import open_example_olr

HERE = Path(__file__).resolve().parent


def main() -> None:
    olr = open_example_olr().sortby("lat").transpose("time", "lat", "lon")
    row = olr.sel(lat=0.0, method="nearest")
    row = row - row.mean("time")
    out = HERE / "input_equator.nc"
    row.astype("float64").rename("olr").to_netcdf(out)
    print(f"wrote {out}  shape={row.shape}")


if __name__ == "__main__":
    main()
