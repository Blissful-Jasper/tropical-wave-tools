"""Write the seeded random inputs shared by the NCL and Python runs.

``filter_input.nc``  (time, lat, lon): white noise plus a few injected plane waves.
``real_input.nc``    (time, lat, lon): the packaged 1979 OLR sample, equatorial rows.
``wk_input.nc``      (time, lat, lon), variable ``olr``: seeded noise + waves on the
                     15S-15N grid, shaped like the input of ``wk_time4.ncl`` (no missing values).

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


def synthetic_wk_input() -> xr.DataArray:
    """Small OLR-like cube: 13 latitudes (15S-15N, 2.5 deg), symmetric/antisymmetric waves."""
    rng = np.random.default_rng(SEED + 1)
    nt, nlon = 384, 72
    lat = np.arange(-15.0, 15.1, 2.5)
    lon = np.arange(nlon) * 360.0 / nlon
    t = np.arange(nt)[:, None, None]
    env = np.exp(-((lat[None, :, None] / 10.0) ** 2))
    x = 240.0 + 8.0 * rng.standard_normal((nt, lat.size, nlon))
    # symmetric Kelvin-like and antisymmetric MRG-like waves
    x += 10.0 * env * np.cos(2 * np.pi * (5 * lon[None, None, :] / 360.0 - t / 8.0))
    x += 10.0 * (lat[None, :, None] / 10.0) * env * np.cos(
        2 * np.pi * (-4 * lon[None, None, :] / 360.0 - t / 5.0)
    )
    return xr.DataArray(
        x.astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=nt, freq="D"), "lat": lat, "lon": lon},
        name="olr",
    )


def real_input() -> xr.DataArray:
    from tropical_wave_tools.sample_data import open_example_olr

    olr = open_example_olr().sortby("lat").transpose("time", "lat", "lon")
    rows = olr.sel(lat=list(LATS), method="nearest")
    return (rows - rows.mean("time")).astype("float64").rename("x").load()


def main() -> None:
    jobs = (
        ("filter_input.nc", synthetic_filter_input()),
        ("real_input.nc", real_input()),
        ("wk_input.nc", synthetic_wk_input()),
    )
    for name, data in jobs:
        out = HERE / name
        data.to_netcdf(out, encoding={data.name: {"zlib": True, "complevel": 5}})
        print(f"wrote {out}  shape={data.shape}  {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
