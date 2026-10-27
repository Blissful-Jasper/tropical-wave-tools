"""Write the seeded random inputs shared by the NCL and Python runs.

``filter_input.nc``  (time, lat, lon): white noise plus a few injected plane waves.
``real_input.nc``    (time, lat, lon): the packaged 1979 OLR sample, equatorial rows.

White noise carries power at every (k, omega), so every filter band and band edge is
exercised; the injected waves make a failure easy to attribute to one band.

Usage: python validation/make_input.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
SEED = 20240601
NT, LATS, NLON = 360, (-5.0, 0.0, 5.0), 72


def synthetic_filter_input() -> xr.DataArray:
    rng = np.random.default_rng(SEED)
    lon = np.arange(NLON) * 360.0 / NLON
    t = np.arange(NT)[:, None, None]
    x = rng.standard_normal((NT, len(LATS), NLON))
    # (k, period in days); k > 0 propagates eastward
    for k, period, amp in [(5, 8.0, 2.0), (-3, 20.0, 2.0), (2, 40.0, 2.0), (-10, 4.0, 2.0)]:
        x += amp * np.cos(2 * np.pi * (k * lon[None, None, :] / 360.0 - t / period))
    return xr.DataArray(
        x,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=NT, freq="D"),
            "lat": list(LATS),
            "lon": lon,
        },
        name="x",
    )


def real_input() -> xr.DataArray:
    from tropical_wave_tools.sample_data import open_example_olr

    olr = open_example_olr().sortby("lat").transpose("time", "lat", "lon")
    rows = olr.sel(lat=list(LATS), method="nearest")
    return (rows - rows.mean("time")).astype("float64").rename("x").load()


def main() -> None:
    for name, data in (("filter_input.nc", synthetic_filter_input()), ("real_input.nc", real_input())):
        out = HERE / name
        data.to_netcdf(out, encoding={"x": {"zlib": True, "complevel": 5}})
        print(f"wrote {out}  shape={data.shape}  {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
