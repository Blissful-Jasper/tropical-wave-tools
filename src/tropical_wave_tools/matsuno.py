"""Matsuno (1966) dispersion relations for equatorial waves."""

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

PI = np.pi
EARTH_RADIUS = 6.371008e6
GRAVITY = 9.80665
EARTH_OMEGA = 7.292e-05
DEG2RAD = PI / 180.0
SEC2DAY = 1.0 / (24.0 * 60.0 * 60.0)


def beta_parameters(latitude: float) -> Tuple[float, float]:
    """Return beta-plane parameter and latitude-circle perimeter."""
    beta = 2.0 * EARTH_OMEGA * np.cos(abs(latitude) * DEG2RAD) / EARTH_RADIUS
    perimeter = 2.0 * PI * EARTH_RADIUS * np.cos(abs(latitude) * DEG2RAD)
    return beta, perimeter


def wn_array(max_wn: int, n_wn: int) -> np.ndarray:
    """Return the global wavenumber array."""
    return np.linspace(-abs(int(max_wn)), abs(int(max_wn)), abs(int(n_wn)))


def wn2k(wn: np.ndarray, perimeter: float) -> np.ndarray:
    """Convert global wavenumber to physical wavenumber in rad m-1."""
    wavelength = perimeter / wn
    return 2.0 * PI / wavelength


def afreq2freq(angular_frequency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert angular frequency to period and cycles per day."""
    period = (2.0 * PI / angular_frequency) * SEC2DAY
    frequency = 1.0 / period
    return period, frequency


def kelvin_mode(
    he: float,
    *,
    latitude: float = 0.0,
    max_wn: int = 50,
    n_wn: int = 500,
) -> pd.DataFrame:
    """Compute the Kelvin-wave dispersion curve."""
    _, perimeter = beta_parameters(latitude)
    wn = wn_array(max_wn, n_wn)
    k = wn2k(wn, perimeter)
    k[k <= 0] = np.nan
    angular_frequency = np.sqrt(GRAVITY * he) * k
    _, frequency = afreq2freq(angular_frequency)
    return pd.DataFrame({f"Kelvin(he={he}m)": frequency}, index=wn)


def mrg_mode(
    he: float,
    *,
    latitude: float = 0.0,
    max_wn: int = 50,
    n_wn: int = 500,
) -> pd.DataFrame:
    """Compute the mixed Rossby-gravity dispersion curve."""
    beta, perimeter = beta_parameters(latitude)
    wn = wn_array(max_wn, n_wn)
    k = wn2k(wn, perimeter)
    angular_frequency = k * np.sqrt(GRAVITY * he) * (
        0.5 - 0.5 * np.sqrt(1 + 4 * beta / (k**2 * np.sqrt(GRAVITY * he)))
    )
    _, frequency = afreq2freq(angular_frequency)
    return pd.DataFrame({f"MRG(he={he}m)": frequency}, index=wn)


def eig_n_0(
    he: float,
    *,
    latitude: float = 0.0,
    max_wn: int = 50,
    n_wn: int = 500,
) -> pd.DataFrame:
    """Compute the n=0 eastward inertia-gravity dispersion curve."""
    beta, perimeter = beta_parameters(latitude)
    wn = wn_array(max_wn, n_wn)
    k = wn2k(wn, perimeter)
    angular_frequency = np.sqrt(beta * np.sqrt(GRAVITY * he) + k**2 * GRAVITY * he)
    _, frequency = afreq2freq(angular_frequency)
    return pd.DataFrame({f"EIG(n=0,he={he}m)": frequency}, index=wn)


def er_n(
    he: float,
    n: int,
    *,
    latitude: float = 0.0,
    max_wn: int = 50,
    n_wn: int = 500,
) -> pd.DataFrame:
    """Compute the equatorial Rossby-wave dispersion curve."""
    beta, perimeter = beta_parameters(latitude)
    wn = wn_array(max_wn, n_wn)
    k = wn2k(wn, perimeter)
    angular_frequency = -beta * k / (k**2 + (2 * n + 1) * beta / np.sqrt(GRAVITY * he))
    _, frequency = afreq2freq(angular_frequency)
    return pd.DataFrame({f"ER(n={n},he={he}m)": frequency}, index=wn)


def dispersion(w: float, k: float, n: int, he: float, beta: float) -> float:
    """Characteristic equation of the Matsuno shallow-water problem."""
    c = np.sqrt(GRAVITY * he)
    return w**3 - (c**2 * k**2 + (2 * n + 1) * beta * c) * w - beta * c**2 * k


def eig_n(
    he: float,
    n: int,
    *,
    latitude: float = 0.0,
    max_wn: int = 50,
    n_wn: int = 500,
) -> pd.DataFrame:
    """Compute eastward inertia-gravity waves for meridional mode ``n``."""
    beta, perimeter = beta_parameters(latitude)
    wn = wn_array(max_wn, n_wn)
    k_values = wn2k(wn, perimeter)

    solutions = []
    for k_value in k_values:
        guess = abs(k_value) * np.sqrt(GRAVITY * he) + 1.0e-6
        root = fsolve(dispersion, x0=guess, args=(k_value, n, he, beta))[0]
        solutions.append(root)

    _, frequency = afreq2freq(np.asarray(solutions))
    return pd.DataFrame({f"EIG(n={n},he={he}m)": frequency}, index=wn)


def wig_n(
    he: float,
    n: int,
    *,
    latitude: float = 0.0,
    max_wn: int = 50,
    n_wn: int = 500,
) -> pd.DataFrame:
    """Compute westward inertia-gravity waves for meridional mode ``n``."""
    beta, perimeter = beta_parameters(latitude)
    wn = wn_array(max_wn, n_wn)
    k_values = wn2k(wn, perimeter)

    solutions = []
    for k_value in k_values:
        guess = -abs(k_value) * np.sqrt(GRAVITY * he) - 1.0e-6
        root = fsolve(dispersion, x0=guess, args=(k_value, n, he, beta))[0]
        solutions.append(root)

    _, frequency = afreq2freq(np.asarray(solutions))
    return pd.DataFrame({f"WIG(n={n},he={he}m)": frequency}, index=wn)


def matsuno_dataframe(
    he: float,
    *,
    n: Tuple[int, ...] = (1, 2, 3),
    latitude: float = 0.0,
    max_wn: int = 50,
    n_wn: int = 500,
) -> pd.DataFrame:
    """Combine several Matsuno branches into a single DataFrame."""
    frames = [kelvin_mode(he, latitude=latitude, max_wn=max_wn, n_wn=n_wn), mrg_mode(he, latitude=latitude, max_wn=max_wn, n_wn=n_wn), eig_n_0(he, latitude=latitude, max_wn=max_wn, n_wn=n_wn)]
    for mode_number in n:
        frames.append(er_n(he, mode_number, latitude=latitude, max_wn=max_wn, n_wn=n_wn))
        frames.append(eig_n(he, mode_number, latitude=latitude, max_wn=max_wn, n_wn=n_wn))
        frames.append(wig_n(he, mode_number, latitude=latitude, max_wn=max_wn, n_wn=n_wn))
    return reduce(lambda left, right: left.join(right), frames)


def matsuno_modes_wk(
    *,
    he: Tuple[float, ...] = (12.0, 25.0, 50.0),
    n: Tuple[int, ...] = (1,),
    latitude: float = 0.0,
    max_wn: int = 20,
    n_wn: int = 500,
) -> Dict[float, pd.DataFrame]:
    """Return Matsuno curves for each equivalent depth."""
    return {
        depth: matsuno_dataframe(depth, n=list(n), latitude=latitude, max_wn=max_wn, n_wn=n_wn)
        for depth in he
    }
