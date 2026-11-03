"""Synthetic Kelvin-wave cases for tests, demos and method validation.

The cases deliberately reproduce the situation motivating
:mod:`tropical_wave_tools.kelvin_events`: a Kelvin-wave circulation that is
symmetric about the equator while its convection is displaced poleward, plus
distractors that a good identification method must reject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from tropical_wave_tools.matsuno import EARTH_OMEGA, EARTH_RADIUS, GRAVITY

__all__ = ["SyntheticEvent", "make_synthetic_kelvin_case"]

_M_PER_DEG = EARTH_RADIUS * np.pi / 180.0


@dataclass(frozen=True)
class SyntheticEvent:
    """One Kelvin-wave packet.

    Attributes
    ----------
    t0_day, lon0:
        Time (days from the start) and longitude (deg) of the packet centre.
    speed:
        Phase and group speed (m s-1, non-dispersive).
    u_amp:
        Peak zonal-wind amplitude (m s-1) on the equator.
    olr_amp:
        Peak OLR amplitude (W m-2, magnitude; enhanced convection is negative).
    wavenumber:
        Zonal wavenumber of the carrier wave.
    sigma_t, sigma_x:
        Gaussian half-widths of the packet in days and degrees.
    conv_offset_deg:
        Longitude shift of the convective crest relative to the wind crest.
    """

    t0_day: float
    lon0: float
    speed: float = 15.0
    u_amp: float = 4.0
    olr_amp: float = 15.0
    wavenumber: int = 4
    sigma_t: float = 3.5
    sigma_x: float = 35.0
    conv_offset_deg: float = 0.0


def _wrapped(delta_deg: np.ndarray) -> np.ndarray:
    return (delta_deg + 180.0) % 360.0 - 180.0


def make_synthetic_kelvin_case(
    *,
    events: Optional[Sequence[SyntheticEvent]] = None,
    ntime: int = 200,
    dlon: float = 2.5,
    dlat: float = 2.5,
    lat_max: float = 20.0,
    conv_lat: float = 8.0,
    conv_sigma_lat: float = 3.0,
    equivalent_depth: float = 25.0,
    noise_u: float = 0.4,
    noise_olr: float = 4.0,
    add_distractors: bool = True,
    seed: int = 0,
) -> xr.Dataset:
    """Build ``u``, ``v``, ``z`` and ``olr`` containing Kelvin-wave packets.

    Kelvin circulation: ``u = A psi_0(y)``, ``z = u c / g``, ``v = 0`` with
    ``psi_0 = exp(-y^2 / 2 Ld^2)`` (equatorially symmetric).  Convection:
    ``OLR' = -B * exp(-(lat - conv_lat)^2 / 2 s^2)`` times the same carrier,
    i.e. displaced to ``conv_lat``.

    Distractors (``add_distractors``): a westward-moving equatorially
    symmetric Rossby-like packet and an *antisymmetric* eastward packet
    (``u ~ y psi_0``) which is orthogonal to the Kelvin structure.
    """
    if events is None:
        events = (
            SyntheticEvent(t0_day=40.0, lon0=130.0),
            SyntheticEvent(t0_day=110.0, lon0=200.0, speed=18.0, u_amp=5.0, conv_offset_deg=10.0),
        )
    rng = np.random.default_rng(seed)
    time = pd.date_range("2000-01-01", periods=ntime, freq="D")
    lon = np.arange(0.0, 360.0, dlon)
    lat = np.arange(-lat_max, lat_max + 0.5 * dlat, dlat)
    t = np.arange(ntime, dtype=float)[:, None, None]
    x = lon[None, None, :]
    y = lat[None, :, None]

    c = np.sqrt(GRAVITY * equivalent_depth)
    ld = np.sqrt(c / (2.0 * EARTH_OMEGA / EARTH_RADIUS))
    yy = _M_PER_DEG * y
    psi0 = np.exp(-0.5 * (yy / ld) ** 2)
    conv_env = np.exp(-0.5 * ((y - conv_lat) / conv_sigma_lat) ** 2)

    u = rng.normal(0.0, noise_u, (ntime, lat.size, lon.size))
    olr = rng.normal(0.0, noise_olr, (ntime, lat.size, lon.size))

    def packet(ev: SyntheticEvent, shift_deg: float = 0.0) -> np.ndarray:
        deg_per_day = ev.speed * 86400.0 / _M_PER_DEG
        centre = ev.lon0 + deg_per_day * (t - ev.t0_day)
        env_t = np.exp(-0.5 * ((t - ev.t0_day) / ev.sigma_t) ** 2)
        dx = _wrapped(x - centre - shift_deg)
        env_x = np.exp(-0.5 * (dx / ev.sigma_x) ** 2)
        carrier = np.cos(np.deg2rad(ev.wavenumber * dx))
        return env_t * env_x * carrier

    for ev in events:
        u += ev.u_amp * packet(ev) * psi0
        olr += -ev.olr_amp * packet(ev, ev.conv_offset_deg) * conv_env

    if add_distractors:
        west = SyntheticEvent(t0_day=70.0, lon0=250.0, speed=-6.0, u_amp=4.0, wavenumber=5, sigma_t=6.0)
        u += west.u_amp * packet(west) * psi0
        olr += -8.0 * packet(west) * np.exp(-0.5 * ((y - 6.0) / 4.0) ** 2)
        anti = SyntheticEvent(t0_day=150.0, lon0=60.0, speed=15.0, u_amp=5.0)
        u += anti.u_amp * packet(anti) * (yy / ld) * psi0

    z = u * c / GRAVITY
    v = rng.normal(0.0, 0.5 * noise_u, u.shape)

    coords = {"time": time, "lat": lat, "lon": lon}
    dims = ("time", "lat", "lon")
    return xr.Dataset(
        {
            "u": (dims, u, {"units": "m s-1"}),
            "v": (dims, v, {"units": "m s-1"}),
            "z": (dims, z, {"units": "m"}),
            "olr": (dims, 230.0 + olr, {"units": "W m-2"}),
        },
        coords=coords,
        attrs={"description": "Synthetic Kelvin-wave case with poleward-displaced convection."},
    )
