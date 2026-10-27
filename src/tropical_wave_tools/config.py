"""Project configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """Configuration for Wheeler-Kiladis spectral analysis.

    ``window_skip_days`` follows NCL ``wkSpaceTime``: it is the gap between the end of one
    window and the start of the next, and a negative value means the windows overlap. Window
    starts are therefore ``window_size_days + window_skip_days`` apart. The default of ``-60``
    is the Wheeler & Kiladis (1999) setting (96-day windows overlapping by 60 days).
    ``tukey_alpha`` is the fraction of each window that is tapered (split cosine bell, as in NCL).
    """

    window_size_days: int = 96
    window_skip_days: int = -60
    samples_per_day: float = 1.0
    freq_cutoff: float | None = None
    wave_limit: int = 27
    contour_levels: tuple[float, ...] = field(default_factory=lambda: DEFAULT_CONTOUR_LEVELS)
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

    period_days: tuple[float | None, float | None]
    wavenumber: tuple[int, int]
    equivalent_depth: tuple[float | None, float | None]
    meridional_mode: int | None = None
    dispersion_family: str = "none"


DEFAULT_WAVE_SPECS: dict[str, WaveSpec] = {
    "kelvin": WaveSpec((3.0, 20.0), (2, 14), (8.0, 90.0), None, "kelvin"),
    "er": WaveSpec((9.0, 72.0), (-10, -1), (8.0, 90.0), 1, "er"),
    # Synoptic gravity-wave boxes follow Mayta et al. (2024), summarizing the
    # Kiladis et al. (2009, 2016) frequency-wavenumber filters.
    "mrg": WaveSpec((2.5, 10.0), (-10, -1), (8.0, 90.0), 0, "eig0_mrg"),
    "eig": WaveSpec((1.0, 6.0), (0, 15), (12.0, 50.0), 0, "eig0_mrg"),
    "ig": WaveSpec((1.0, 6.0), (0, 15), (12.0, 50.0), 0, "eig0_mrg"),
    "wig": WaveSpec((1.25, 3.5), (-15, -1), (12.0, 90.0), 1, "ig"),
    "td": WaveSpec((2.5, 5.0), (-20, -6), (None, None), None, "none"),
    "mjo": WaveSpec((20.0, 100.0), (1, 5), (None, None), None, "none"),
    "eig0": WaveSpec((1.0, 6.0), (0, 15), (12.0, 50.0), 0, "eig0_mrg"),
}
