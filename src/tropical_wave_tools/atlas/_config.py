"""Case-specific constants of the atlas (waves, latitude bands, lags, quiver scales, ...)."""

from __future__ import annotations

COMPARISON_WAVE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("large_scale", ("kelvin", "er", "mjo")),
    ("westward", ("mrg", "td")),
)


CASE04_WAVES: tuple[str, ...] = ("kelvin", "er", "mjo", "mrg", "td")


CASE04_LAT_RANGE: tuple[float, float] = (-25.0, 25.0)


CASE05_WAVES: tuple[str, ...] = ("kelvin", "er", "mrg", "td")


CASE05_LAT_RANGE: tuple[float, float] = (-20.0, 20.0)


CASE05_REGION_RANGES: tuple[tuple[str, tuple[float, float]], ...] = (
    ("africa", (343.0, 50.0)),
    ("indian_ocean", (51.0, 94.0)),
    ("maritime_continent", (95.0, 150.0)),
    ("west_pacific", (151.0, 210.0)),
    ("east_pacific", (211.0, 270.0)),
    ("america", (271.0, 325.0)),
    ("atlantic", (326.0, 342.0)),
)


CASE05_REGION_LABELS: dict[str, str] = {
    "africa": "Africa",
    "indian_ocean": "Indian Ocean",
    "maritime_continent": "Maritime Continent",
    "west_pacific": "West Pacific",
    "east_pacific": "East Pacific",
    "america": "America",
    "atlantic": "Atlantic",
}


CASE05_PRECIP_VARIABLE = "precipitation"


CASE06_WAVE_QUIVER_STRIDES: dict[str, int] = {
    "kelvin": 3,
    "er": 3,
    "mjo": 4,
    "mrg": 3,
    "td": 2,
}


CASE06_WAVE_QUIVER_SCALES: dict[str, float] = {
    "kelvin": 20.0,
    "er": 22.0,
    "mjo": 26.0,
    "mrg": 18.0,
    "td": 10.0,
}


CASE07_WAVES: tuple[str, ...] = ("kelvin", "er", "mjo", "mrg", "td")


CASE07_WAVE_LAT_RANGES: dict[str, tuple[float, float]] = {
    "kelvin": (-12.5, 12.5),
    "er": (-18.0, 18.0),
    "mjo": (-20.0, 20.0),
    "mrg": (-18.0, 18.0),
    "td": (-22.5, 22.5),
}


CASE07_WAVE_PROJECTIONS: dict[str, str] = {
    "kelvin": "symmetric",
    "er": "symmetric",
    "mjo": "full",
    "mrg": "antisymmetric",
    "td": "full",
}


CASE07_WAVE_QUIVER_STRIDES: dict[str, int] = {
    "kelvin": 3,
    "er": 3,
    "mjo": 4,
    "mrg": 3,
    "td": 3,
}


CASE07_WAVE_QUIVER_SCALES: dict[str, float] = {
    "kelvin": 18.0,
    "er": 18.0,
    "mjo": 17.0,
    "mrg": 15.0,
    "td": 16.0,
}


CASE08_WAVE_LAGS: dict[str, tuple[int, ...]] = {
    "kelvin": (-6, -4, -2, 0, 2, 4, 6),
    "er": (-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12),
    "mjo": (-24, -18, -12, -6, 0, 6, 12, 18, 24),
    "mrg": (-3, -2, -1, 0, 1, 2, 3),
    "td": (-4, -2, 0, 2, 4),
}


CASE08_WAVE_MAP_LAGS: dict[str, tuple[int, ...]] = {
    "kelvin": (-4, -2, 0, 2, 4),
    "er": (-12, -6, 0, 6, 12),
    "mjo": (-24, -12, 0, 12, 24),
    "mrg": (-2, -1, 0, 1, 2),
    "td": (-4, -2, 0, 2, 4),
}


CASE08_WAVE_FOCUS_HALF_WIDTH: dict[str, float] = {
    "kelvin": 80.0,
    "er": 90.0,
    "mjo": 100.0,
    "mrg": 75.0,
    "td": 75.0,
}


CASE08_WAVE_FOCUS_CENTERS: dict[str, float] = {
    "kelvin": 200.0,
    "er": 172.5,
    "mrg": 172.5,
    "td": 133.0,
    "mjo": 150.0,
}


CASE08_EVENT_LAT_BANDS: dict[str, tuple[float, float]] = {
    "td": (5.0, 15.0),
}


CASE08_EVENT_LON_REFS: dict[str, float] = {
    "er": 172.5,
    "mrg": 172.5,
    "td": 133.0,
    "mjo": 150.0,
}


CASE08_EVENT_BASE_POINTS: dict[str, tuple[float, float]] = {
    "kelvin": (7.0, 200.0),
    "er": (7.5, 172.5),
    "mrg": (7.5, 172.5),
    "td": (7.5, 133.0),
}


CASE08_EVENT_MIN_SPACING: dict[str, int] = {
    "kelvin": 3,
    "er": 8,
    "mjo": 12,
    "mrg": 4,
    "td": 4,
}


CASE08_WAVE_QUIVER_STRIDES: dict[str, int] = {
    "kelvin": 2,
    "er": 2,
    "mjo": 3,
    "mrg": 2,
    "td": 2,
}


CASE08_WAVE_QUIVER_SCALES: dict[str, float] = {
    "kelvin": 8.5,
    "er": 10.5,
    "mjo": 8.0,
    "mrg": 7.5,
    "td": 6.5,
}


CASE08_WAVE_QUIVER_WIDTHS: dict[str, float] = {
    "kelvin": 0.0020,
    "er": 0.00195,
    "mjo": 0.00185,
    "mrg": 0.0020,
    "td": 0.00205,
}


CASE08_WAVE_OLR_LEVEL_COUNT: dict[str, int] = {
    "kelvin": 37,
    "er": 37,
    "mjo": 33,
    "mrg": 37,
    "td": 33,
}


CASE08_WAVE_OLR_RANGE_SCALE: dict[str, float] = {
    "kelvin": 1.5,
    "er": 1.45,
    "mjo": 1.25,
    "mrg": 1.5,
    "td": 1.6,
}


CASE08_WAVE_OLR_MIN_LIMIT: dict[str, float] = {
    "kelvin": 12.0,
    "er": 10.0,
    "mjo": 12.0,
    "mrg": 10.0,
    "td": 5.0,
}


CASE09_WAVES: tuple[str, ...] = ("kelvin", "er", "mjo", "mrg", "td")


CASE10_WAVES: tuple[str, ...] = ("kelvin", "er", "mjo", "mrg", "td")


CASE10_BASE_POINTS: dict[str, tuple[float, float]] = {
    "kelvin": (7.0, 200.0),
    "er": (7.5, 172.5),
    "mjo": (0.0, 90.0),
    "mrg": (7.5, 172.5),
    "td": (7.5, 133.0),
}


CASE10_LAGS: dict[str, tuple[int, ...]] = {
    "kelvin": tuple(range(-15, 16)),
    "er": tuple(range(-20, 21)),
    "mjo": tuple(range(-35, 36)),
    "mrg": tuple(range(-10, 11)),
    "td": tuple(range(-8, 9)),
}


CASE10_WINDOW_VARIANCE_DAYS: dict[str, int] = {
    "kelvin": 17,
    "er": 72,
    "mjo": 96,
    "mrg": 10,
    "td": 10,
}


CASE10_LON_WINDOWS: dict[str, tuple[float, float]] = {
    "kelvin": (90.0, 320.0),
    "er": (30.0, 285.0),
    "mjo": (30.0, 270.0),
    "mrg": (70.0, 255.0),
    "td": (75.0, 225.0),
}


CASE10_XTICKS: dict[str, tuple[float, ...]] = {
    "kelvin": (90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
    "er": (30.0, 90.0, 150.0, 210.0, 270.0),
    "mjo": (30.0, 90.0, 150.0, 210.0, 270.0),
    "mrg": (75.0, 120.0, 165.0, 210.0, 255.0),
    "td": (75.0, 112.5, 150.0, 187.5, 225.0),
}


CASE10_SHADING_RANGE_SCALE: dict[str, float] = {
    "kelvin": 1.65,
    "er": 1.7,
    "mjo": 1.35,
    "mrg": 1.65,
    "td": 1.8,
}


CASE10_SHADING_MIN_LIMIT: dict[str, float] = {
    "kelvin": 12.0,
    "er": 10.0,
    "mjo": 10.0,
    "mrg": 10.0,
    "td": 8.0,
}


CASE10_CONTOUR_TARGET_STEPS: dict[str, int] = {
    "kelvin": 6,
    "er": 6,
    "mjo": 7,
    "mrg": 5,
    "td": 5,
}


CASE10_FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "kelvin": (7.4, 5.15),
    "er": (7.8, 5.2),
    "mjo": (8.1, 5.35),
    "mrg": (7.5, 5.0),
    "td": (7.2, 4.95),
}


CASE03_HOV_LAT_BANDS: dict[str, tuple[float, float]] = {
    # Lubis & Jacobi (2015) diagnose MRG time-longitude evolution from an
    # off-equatorial base latitude near 7.5N rather than a north-minus-south
    # projection; this avoids confusing the westward phase with the eastward
    # group envelope in antisymmetric averages.
    "mrg": (5.0, 10.0),
}


CASE03_HOV_LON_REFS: dict[str, float] = {
    "mrg": 172.5,
}


WAVE_QUIVER_SCALES: dict[str, float] = {
    "kelvin": 42.0,
    "er": 45.0,
    "mjo": 28.0,
    "mrg": 38.0,
    "td": 40.0,
}


ANTISYMMETRIC_WAVES: frozenset[str] = frozenset({"mrg", "eig", "eig0", "ig", "wig"})


WAVE_PROPAGATION_DIRECTIONS: dict[str, str] = {
    "kelvin": "eastward",
    "mjo": "eastward",
    "eig": "eastward",
    "eig0": "eastward",
    "ig": "eastward",
    "er": "westward",
    "mrg": "westward",
    "td": "westward",
    "wig": "westward",
}


WAVE_HOVMOLLER_WINDOWS: dict[str, int] = {
    "kelvin": 120,
    "er": 144,
    "mjo": 240,
    "mrg": 96,
    "td": 72,
}
