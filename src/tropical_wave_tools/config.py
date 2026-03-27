"""Project configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


DEFAULT_CONTOUR_LEVELS = (
    0.0,
    0.4,
    0.6,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.4,
    1.7,
    2.0,
    2.4,
    2.8,
    4.0,
)


@dataclass
class SpectralConfig:
    """Configuration for Wheeler-Kiladis spectral analysis."""

    window_size_days: int = 96
    window_skip_days: int = 30
    samples_per_day: float = 1.0
    freq_cutoff: Optional[float] = None
    wave_limit: int = 27
    contour_levels: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_CONTOUR_LEVELS)
    wavenumber_limit: int = 15
    tukey_alpha: float = 0.1

    @property
    def resolved_freq_cutoff(self) -> float:
        """Return the effective low-frequency cutoff used for annual-cycle removal."""
        if self.freq_cutoff is not None:
            return self.freq_cutoff
        return 1.0 / float(self.window_size_days)


@dataclass(frozen=True)
class WaveSpec:
    """Definition of one wave filter band."""

    period_days: Tuple[Optional[float], Optional[float]]
    wavenumber: Tuple[int, int]
    equivalent_depth: Tuple[Optional[float], Optional[float]]
    meridional_mode: Optional[int] = None
    dispersion_family: str = "none"


DEFAULT_WAVE_SPECS: Dict[str, WaveSpec] = {
    "kelvin": WaveSpec((3.0, 20.0), (2, 14), (8.0, 90.0), None, "kelvin"),
    "er": WaveSpec((9.0, 72.0), (-10, -1), (8.0, 90.0), 1, "er"),
    "mrg": WaveSpec((3.0, 10.0), (-10, -1), (8.0, 90.0), 0, "eig0_mrg"),
    "ig": WaveSpec((1.0, 14.0), (1, 5), (8.0, 90.0), 1, "ig"),
    "td": WaveSpec((2.5, 5.0), (-20, -6), (None, None), None, "none"),
    "mjo": WaveSpec((20.0, 100.0), (1, 5), (None, None), None, "none"),
    "eig0": WaveSpec((None, 1.0 / 0.55), (0, 15), (12.0, 50.0), 0, "eig0_mrg"),
}
