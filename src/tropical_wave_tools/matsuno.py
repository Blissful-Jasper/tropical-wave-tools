"""Matsuno (1966) dispersion relations for equatorial waves."""

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PI = np.pi
EARTH_RADIUS = 6.371008e6
GRAVITY = 9.80665
EARTH_OMEGA = 7.292e-05
DEG2RAD = PI / 180.0
SECONDS_PER_DAY = 24.0 * 60.0 * 60.0
ANGULAR_TO_CPD = SECONDS_PER_DAY / (2.0 * PI)


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
    return (2.0 * PI * wn) / perimeter


def afreq2freq(angular_frequency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert angular frequency to period and cycles per day."""
    frequency = np.asarray(angular_frequency, dtype=float) * ANGULAR_TO_CPD
    with np.errstate(divide="ignore", invalid="ignore"):
        period = np.where(np.abs(frequency) > 0.0, 1.0 / frequency, np.nan)
    return period, frequency


def _directional_frequency(
    wn: np.ndarray,
    angular_frequency: np.ndarray,
    *,
    direction: str,
) -> np.ndarray:
    """Return positive frequencies on the physically relevant propagation side."""
    _, frequency = afreq2freq(angular_frequency)
    values = np.abs(frequency)

    if direction == "eastward":
        values = np.where(wn >= 0.0, values, np.nan)
    elif direction == "westward":
        values = np.where(wn <= 0.0, values, np.nan)
    else:
        raise ValueError("`direction` must be 'eastward' or 'westward'.")
    return values


def _cubic_mode_roots(
    he: float,
    n: int,
    *,
    latitude: float = 0.0,
    max_wn: int = 50,
    n_wn: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return signed cubic roots for WIG, ER, and EIG branches."""
    beta, perimeter = beta_parameters(latitude)
    wn = wn_array(max_wn, n_wn)
    k_values = wn2k(wn, perimeter)
    c = np.sqrt(GRAVITY * he)

    roots = np.full((wn.size, 3), np.nan, dtype=float)
    for index, k_value in enumerate(k_values):
        coefficients = [
            1.0,
            0.0,
            -(c**2 * k_value**2 + (2 * n + 1) * beta * c),
            -beta * c**2 * k_value,
        ]
        root_values = np.roots(coefficients)
        real_roots = np.real(root_values[np.isclose(np.imag(root_values), 0.0, atol=1.0e-10)])
        if real_roots.size != 3:
            real_roots = np.sort(np.real(root_values))
        else:
            real_roots = np.sort(real_roots)
        roots[index, :] = real_roots

    return wn, roots[:, 0], roots[:, 1], roots[:, 2]


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
    angular_frequency = np.sqrt(GRAVITY * he) * k
    frequency = _directional_frequency(wn, angular_frequency, direction="eastward")
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
    c = np.sqrt(GRAVITY * he)
    # The positive n=0 branch is split into westward MRG and eastward EIG0 in WK plots.
    angular_frequency = 0.5 * (k * c + np.sqrt((k * c) ** 2 + 4.0 * beta * c))
    frequency = _directional_frequency(wn, angular_frequency, direction="westward")
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
    c = np.sqrt(GRAVITY * he)
    angular_frequency = 0.5 * (k * c + np.sqrt((k * c) ** 2 + 4.0 * beta * c))
    frequency = _directional_frequency(wn, angular_frequency, direction="eastward")
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
    wn, _, rossby_root, _ = _cubic_mode_roots(
        he,
        n,
        latitude=latitude,
        max_wn=max_wn,
        n_wn=n_wn,
    )
    frequency = _directional_frequency(wn, rossby_root, direction="westward")
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
    wn, _, _, eastward_root = _cubic_mode_roots(
        he,
        n,
        latitude=latitude,
        max_wn=max_wn,
        n_wn=n_wn,
    )
    frequency = _directional_frequency(wn, eastward_root, direction="eastward")
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
    wn, westward_root, _, eastward_root = _cubic_mode_roots(
        he,
        n,
        latitude=latitude,
        max_wn=max_wn,
        n_wn=n_wn,
    )
    # For negative wavenumbers, the physically relevant WIG branch is the
    # fast positive-frequency root, not the negative root whose absolute value
    # would otherwise fold an eastward branch into the westward side.
    wig_root = np.where(wn <= 0.0, eastward_root, westward_root)
    frequency = _directional_frequency(wn, wig_root, direction="westward")
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
