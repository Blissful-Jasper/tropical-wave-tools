"""Plots checked against what they render, not just that they run.

The checks read the rendered pixels at known data positions, the artists' data (lines, arrows,
rectangles) and the colour bands of the filled contours, and compare them with the inputs and with
independent formulas (the nondimensional Matsuno relations, polyfit, numpy).
"""

from __future__ import annotations

import weakref

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.quiver import Quiver

from tropical_wave_tools import plotting
from tropical_wave_tools.config import DEFAULT_WAVE_SPECS
from tropical_wave_tools.plotting import _geo, _style, spectrum
from tropical_wave_tools.spectral import WKAnalysisResult

if tuple(int(part) for part in matplotlib.__version__.split(".")[:2]) < (3, 8):  # pragma: no cover
    pytest.skip("needs matplotlib >= 3.8 (ContourSet as a collection)", allow_module_level=True)

try:
    import cartopy.crs as ccrs
except ImportError:  # pragma: no cover - optional dependency
    ccrs = None

needs_cartopy = pytest.mark.skipif(ccrs is None, reason="cartopy is not installed")

LAT = np.arange(-30.0, 30.1, 2.5)
LON = np.arange(0.0, 360.0, 5.0)
LON0, LAT0 = 105.0, 12.5  # the strong feature; off every axis tick so no grid line crosses it
LAT_GRID, LON_GRID = np.meshgrid(LAT, LON, indexing="ij")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- pixel helpers
_RENDERED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _raster(figure) -> np.ndarray:
    """The figure drawn once (the figures are large; the tests never change one after reading it)."""
    if figure not in _RENDERED:
        figure.canvas.draw()
        _RENDERED[figure] = np.asarray(figure.canvas.buffer_rgba()).copy()
    return _RENDERED[figure]


def _pixel(figure, axes, x, y):
    """RGB of the rendered figure at data point (x, y); (lon, lat) for map axes."""
    buffer = _raster(figure)
    if hasattr(axes, "projection"):
        x, y = axes.projection.transform_point(x, y, ccrs.PlateCarree())
    px, py = axes.transData.transform((x, y))
    return buffer[int(buffer.shape[0] - 1 - round(py)), round(px), :3] / 255.0


def _is_hole(rgb, axes) -> bool:
    """True if a pixel shows the axes background: nothing was painted there."""
    face = np.array(mcolors.to_rgb(axes.get_facecolor()))
    return bool(np.allclose(rgb, face, atol=0.03) or np.allclose(rgb, (1.0, 1.0, 1.0), atol=0.03))


def _filled(axes):
    return [c for c in axes.collections if hasattr(c, "levels") and getattr(c, "filled", False)]


def _holes_around(figure, x, y, dx, dy) -> int:
    """Count unpainted samples in a 3 x 3 neighbourhood of a feature, over every contour panel."""
    holes = 0
    for axes in figure.axes:
        if not _filled(axes):
            continue
        for ox in (-dx, 0.0, dx):
            for oy in (-dy, 0.0, dy):
                holes += _is_hole(_pixel(figure, axes, x + ox, y + oy), axes)
    return holes


def _band_rgb(contour_set, value):
    """Colour a filled contour set gives ``value``, read from its own band colours (None: unpainted)."""
    levels = np.asarray(contour_set.levels, dtype=float)
    extend = contour_set.extend
    faces = np.asarray(contour_set.get_facecolor())[:, :3]
    first = 1 if extend in ("min", "both") else 0
    if value < levels[0]:
        return faces[0] if extend in ("min", "both") else None
    if value > levels[-1]:
        return faces[-1] if extend in ("max", "both") else None
    band = int(np.clip(np.searchsorted(levels, value, side="right") - 1, 0, len(levels) - 2))
    return faces[first + band]


def _safe_nodes(contour_set, field, candidates, margin=0.3):
    """Candidate (x, y) nodes whose value lies well inside a colour band (not on a band edge)."""
    levels = np.asarray(contour_set.levels, dtype=float)
    chosen = []
    for x, y, value in candidates:
        if value <= levels[0] or value >= levels[-1]:
            chosen.append((x, y, value))
            continue
        band = int(np.searchsorted(levels, value, side="right") - 1)
        width = levels[band + 1] - levels[band]
        if min(value - levels[band], levels[band + 1] - value) > margin * width:
            chosen.append((x, y, value))
    return chosen


def _assert_colours_follow_data(figure, axes, nodes):
    contour_set = _filled(axes)[0]
    chosen = _safe_nodes(contour_set, None, nodes)
    assert len(chosen) >= 3, "too few usable test nodes"
    for x, y, value in chosen:
        expected = _band_rgb(contour_set, value)
        assert expected is not None, f"value {value:.2f} at ({x}, {y}) is not covered by the colour scale"
        np.testing.assert_allclose(_pixel(figure, axes, x, y), expected, atol=0.02, err_msg=f"({x}, {y}) = {value:.3f}")


# ---------------------------------------------------------------------------- test fields
def _blob(amplitude=10.0, centre=(LON0, LAT0)):
    return amplitude * np.exp(-(((LON_GRID - centre[0]) / 12.0) ** 2 + ((LAT_GRID - centre[1]) / 7.0) ** 2))


def _latlon(amplitude=10.0, phase=0.0, positive=False):
    base = 1.5 * np.sin(np.deg2rad(2.0 * LON_GRID + phase)) * np.cos(np.deg2rad(3.0 * LAT_GRID))
    if positive:
        base = np.abs(base)
    return xr.DataArray(
        base + _blob(amplitude), dims=("lat", "lon"), coords={"lat": LAT, "lon": LON}, attrs={"units": "W m-2"}
    )


def _lagged(lags=(-4, 0, 4), amplitude=10.0):
    return xr.concat(
        [_latlon(amplitude * (1.0 - 0.1 * abs(lag)), phase=17.0 * lag).expand_dims(lag=[lag]) for lag in lags],
        dim="lag",
    )


def _zero_wind(template):
    return template * 0.0


def _hovmoller(amplitude=10.0):
    n_time = 100
    time_index, lon = np.meshgrid(np.arange(n_time), LON, indexing="ij")
    base = 1.5 * np.cos(np.deg2rad(1.0 * (lon - 6.0 * time_index)))
    blob = amplitude * np.exp(-(((time_index - 60) / 5.0) ** 2 + ((lon - LON0) / 12.0) ** 2))
    return xr.DataArray(
        base + blob,
        dims=("time", "lon"),
        coords={"time": pd.date_range("2000-01-01", periods=n_time), "lon": LON},
    )


def _lag_lon(amplitude=10.0):
    lag = np.arange(-10, 11)
    lag_grid, lon_grid = np.meshgrid(lag, LON, indexing="ij")
    base = 1.5 * np.cos(np.deg2rad(1.0 * (lon_grid - 10.0 * lag_grid)))
    blob = amplitude * np.exp(-((lag_grid / 3.0) ** 2 + ((lon_grid - LON0) / 12.0) ** 2))
    return xr.DataArray(base + blob, dims=("lag", "lon"), coords={"lag": lag, "lon": LON})


def _month_lon(amplitude=15.0):
    month = np.arange(1, 13)
    month_grid, lon_grid = np.meshgrid(month, LON, indexing="ij")
    base = 2.0 + np.abs(np.sin(np.deg2rad(3.0 * lon_grid)))
    blob = amplitude * np.exp(-(((month_grid - 6) / 1.5) ** 2 + ((lon_grid - LON0) / 12.0) ** 2))
    return xr.DataArray(base + blob, dims=("month", "lon"), coords={"month": month, "lon": LON})


# ============================================================================ holes
# Contour plots paint nothing outside their outermost level unless the colormap is extended. The
# scales come from robust quantiles, so the *strongest* signals used to be left blank.
def _map_cases():
    field = _latlon()
    std = abs(_latlon(positive=True))
    zero_wind = _zero_wind(field)
    lagged = _lagged()
    zero_wind_lagged = _zero_wind(lagged)
    eof_modes = xr.concat([_latlon(phase=40.0 * mode).expand_dims(mode=[mode]) for mode in (1, 2)], dim="mode")
    eof_wind = _zero_wind(eof_modes)
    return {
        "latlon_field": lambda c: plotting.plot_latlon_field(field, use_cartopy=c)[0],
        "spatial_std_triptych": lambda c: plotting.plot_spatial_std_triptych(std, std, std, use_cartopy=c)[0],
        "wave_spatial_comparison": lambda c: plotting.plot_wave_spatial_comparison(
            [std, std], titles=["a", "b"], use_cartopy=c
        )[0],
        "horizontal_structure": lambda c: plotting.plot_horizontal_structure(field, zero_wind, zero_wind, use_cartopy=c)[0],
        "wave_horizontal_structure_comparison": lambda c: plotting.plot_wave_horizontal_structure_comparison(
            [field, field], [zero_wind, zero_wind], [zero_wind, zero_wind], titles=["a", "b"], use_cartopy=c
        )[0],
        "lagged_horizontal_structure": lambda c: plotting.plot_lagged_horizontal_structure(
            lagged, zero_wind_lagged, zero_wind_lagged, ncols=3, use_cartopy=c
        )[0],
        "wave_evolution_comparison": lambda c: plotting.plot_wave_evolution_comparison(
            [lagged, lagged],
            [zero_wind_lagged, zero_wind_lagged],
            [zero_wind_lagged, zero_wind_lagged],
            wave_names=["kelvin", "er"],
            lags=(-4, 0, 4),
            use_cartopy=c,
        )[0],
        "wind_diagnostics_panel": lambda c: plotting.plot_wind_diagnostics_panel(
            field, field, zero_wind, zero_wind, use_cartopy=c, integer_colorbar=False
        )[0],
        "eof_patterns_and_pcs": lambda c: plotting.plot_eof_spatial_patterns_and_pcs(
            eof_modes,
            xr.DataArray(
                np.zeros((2, 5)),
                dims=("mode", "time"),
                coords={"mode": [1, 2], "time": pd.date_range("2000-01-01", periods=5)},
            ),
            [30.0, 20.0],
            use_cartopy=c,
        )[0],
        "eof_modes_with_wind": lambda c: plotting.plot_eof_modes_with_wind(
            eof_modes, eof_wind, eof_wind, [30.0, 20.0], integer_colorbar=False, use_cartopy=c
        )[0],
    }


MAP_CASES = _map_cases()


@pytest.mark.parametrize("name", sorted(MAP_CASES))
def test_strongest_map_signal_is_painted_not_left_blank(name: str) -> None:
    figure = MAP_CASES[name](False)
    assert _holes_around(figure, LON0, LAT0, dx=5.0, dy=2.5) == 0


@needs_cartopy
@pytest.mark.parametrize(
    "name", ["latlon_field", "horizontal_structure", "lagged_horizontal_structure", "eof_modes_with_wind"]
)
def test_strongest_map_signal_is_painted_with_cartopy(name: str) -> None:
    figure = MAP_CASES[name](True)
    assert _holes_around(figure, LON0, LAT0, dx=5.0, dy=2.5) == 0


def test_strongest_hovmoller_signal_is_painted_not_left_blank() -> None:
    field = _hovmoller()
    time_number = mdates.date2num(field.time.values[60])
    figure, _ = plotting.plot_hovmoller_comparison(field, field * 0.5)
    assert _holes_around(figure, LON0, time_number, dx=5.0, dy=1.5) == 0
    figure, _ = plotting.plot_hovmoller_triptych([field, field, field], titles=["a", "b", "c"])
    assert _holes_around(figure, LON0, time_number, dx=5.0, dy=1.5) == 0


def test_strongest_lag_longitude_signal_is_painted_not_left_blank() -> None:
    field = _lag_lon()
    figure, _ = plotting.plot_lag_longitude_evolution(field, title="t")
    assert _holes_around(figure, LON0, 0.0, dx=5.0, dy=1.0) == 0
    figure, _ = plotting.plot_paper_style_hovmoller(field, field * 0.5, title="t", base_point_label="b")
    assert _holes_around(figure, LON0, 0.0, dx=5.0, dy=1.0) == 0


def test_month_longitude_plots_paint_the_strongest_signal() -> None:
    field = _month_lon()
    figure, _ = plotting.plot_monthly_longitude_heatmap(field, title="t")
    assert _holes_around(figure, LON0, 6.0, dx=5.0, dy=0.5) == 0
    figure, _ = plotting.plot_wave_monthly_longitude_comparison([field, field * 0.8], wave_names=["a", "b"])
    assert _holes_around(figure, LON0, 6.0, dx=5.0, dy=0.5) == 0


def test_explicit_neither_still_gives_a_plain_colorbar() -> None:
    """``colorbar_extend="neither"`` remains available for those who want the old look."""
    field = _hovmoller()
    _, axes = plotting.plot_hovmoller_comparison(field, field * 0.5, colorbar_extend="neither")
    assert _filled(axes[0])[0].extend == "neither"
    _, axes = plotting.plot_hovmoller_comparison(field, field * 0.5)  # the default extends where values exceed the scale
    assert _filled(axes[0])[0].extend in ("max", "both")
    _, axes = plotting.plot_hovmoller_comparison(-field, field)  # exceeded on both sides
    assert _filled(axes[0])[0].extend == "both"


@pytest.mark.parametrize(
    ("extend", "values", "levels", "expected"),
    [
        (None, [0.0, 1.0], [-2.0, 2.0], "neither"),
        (None, [-3.0, 1.0], [-2.0, 2.0], "min"),
        (None, [0.0, 3.0], [-2.0, 2.0], "max"),
        (None, [-3.0, 3.0, np.nan], [-2.0, 2.0], "both"),
        (None, [np.nan, np.nan], [-2.0, 2.0], "neither"),
        ("neither", [-9.0, 9.0], [-2.0, 2.0], "neither"),
        ("both", [0.0], [-2.0, 2.0], "both"),
    ],
)
def test_resolve_extend(extend, values, levels, expected) -> None:
    assert _style._resolve_extend(extend, np.array(values), np.array(levels)) == expected
    assert _style._resolve_extend(extend, [xr.DataArray(values)], levels) == expected


# ====================================================================== colours / orientation
def test_hovmoller_colours_follow_the_data_at_each_lon_time() -> None:
    field = _hovmoller()
    figure, axes = plotting.plot_hovmoller_comparison(field, field * 0.5)
    for axis, scale in zip(axes, (1.0, 0.5)):
        nodes = [
            (float(lon), float(mdates.date2num(field.time.values[ti])), float(field.isel(time=ti).sel(lon=lon)) * scale)
            for ti in (10, 22, 35, 45, 75, 90)
            for lon in (40.0, 85.0, 205.0, 305.0)
        ]
        _assert_colours_follow_data(figure, axis, nodes)
    first, second = (_filled(axis)[0] for axis in axes)
    np.testing.assert_allclose(first.levels, second.levels)  # shared scale
    assert np.isclose(first.levels[0], -first.levels[-1])  # symmetric about zero


def test_paper_hovmoller_colours_follow_the_data_and_axes() -> None:
    field = _lag_lon()
    flat_contours = field * 0.0 + 1.0  # no contour lines to cross the probe pixels
    figure, axis = plotting.plot_paper_style_hovmoller(field, flat_contours, title="t", base_point_label="b")
    nodes = [(float(lon), float(lag), float(field.sel(lag=lag, lon=lon))) for lag in (-8, -6, -3, 3, 5, 8) for lon in (40.0, 65.0, 160.0, 255.0, 315.0)]
    _assert_colours_follow_data(figure, axis, nodes)
    assert axis.yaxis_inverted()  # lag runs downward in this style
    assert axis.get_xlabel() == "Longitude" and axis.get_ylabel() == "Lag (days)"


def test_latlon_colours_follow_the_data_for_any_dimension_or_latitude_order() -> None:
    field = _latlon()
    nodes = [(float(lon), float(lat), float(field.sel(lat=lat, lon=lon))) for lat in (-22.5, -12.5, 7.5, 12.5, 22.5) for lon in (25.0, 105.0, 215.0, 335.0)]
    for variant in (field, field.isel(lat=slice(None, None, -1)), field.transpose("lon", "lat")):
        figure, axes = plotting.plot_latlon_field(variant, use_cartopy=False)
        assert axes.get_xlabel() == "lon" and axes.get_ylabel() == "lat"
        _assert_colours_follow_data(figure, axes, nodes)
        plt.close(figure)


@needs_cartopy
def test_latlon_colours_follow_the_data_with_cartopy_for_any_dimension_order() -> None:
    field = _latlon()
    nodes = [(float(lon), float(lat), float(field.sel(lat=lat, lon=lon))) for lat in (-12.5, 7.5, 12.5, 22.5) for lon in (25.0, 105.0, 215.0)]
    for variant in (field, field.transpose("lon", "lat")):
        figure, axes = plotting.plot_latlon_field(variant, use_cartopy=True)
        _assert_colours_follow_data(figure, axes, nodes)
        plt.close(figure)


def test_nan_cells_are_left_unpainted() -> None:
    field = _latlon(amplitude=0.0).copy()
    field[10:13, 20:24] = np.nan
    figure, axes = plotting.plot_latlon_field(field, use_cartopy=False)
    lat, lon = float(LAT[11]), float(LON[21])
    assert _is_hole(_pixel(figure, axes, lon, lat), axes)
    assert not _is_hole(_pixel(figure, axes, float(LON[40]), float(LAT[11])), axes)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("fill", [0.0, 5.0, np.nan])
def test_degenerate_fields_do_not_crash(fill: float) -> None:
    flat = _latlon(positive=True) * 0.0 + fill
    plotting.plot_latlon_field(flat, use_cartopy=False)
    plotting.plot_spatial_std_comparison(flat, flat, wave_name="eig", use_cartopy=False)
    plotting.plot_spatial_std_triptych(flat, flat, flat, use_cartopy=False)
    plotting.plot_wave_spatial_comparison([flat, flat], titles=["a", "b"], use_cartopy=False)


# ---------------------------------------------------------------- dimension-order independence
def _rendered(figure) -> np.ndarray:
    return _raster(figure)


def test_map_plots_do_not_depend_on_dimension_order() -> None:
    """A (lon, lat) input used to draw a transposed map and quiver arrows with the wrong components."""
    field = _latlon()
    u = xr.DataArray(0.05 * LON_GRID - 5.0, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    v = xr.DataArray(0.2 * LAT_GRID, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    swap = lambda a: a.transpose(..., "lon", "lat")  # noqa: E731
    lagged, u_lag, v_lag = _lagged(), _lagged() * 0.0 + u, _lagged() * 0.0 + v
    cases = {
        "horizontal_structure": lambda t: plotting.plot_horizontal_structure(
            t(field), t(u), t(v), use_cartopy=False, wind_overlay="vectors"
        )[0],
        "wind_diagnostics_panel": lambda t: plotting.plot_wind_diagnostics_panel(
            t(field), t(field * 0.5), t(u), t(v), use_cartopy=False
        )[0],
        "lagged_horizontal_structure": lambda t: plotting.plot_lagged_horizontal_structure(
            t(lagged), t(u_lag), t(v_lag), ncols=3, wind_overlay="vectors", use_cartopy=False
        )[0],
        "spatial_std_triptych": lambda t: plotting.plot_spatial_std_triptych(
            t(abs(field)), t(abs(u)), t(abs(v)), use_cartopy=False
        )[0],
    }
    for name, make in cases.items():
        reference = _rendered(make(lambda a: a))
        plt.close("all")
        transposed = _rendered(make(swap))
        plt.close("all")
        assert reference.shape == transposed.shape, name
        assert np.array_equal(reference, transposed), f"{name}: output depends on the dimension order"


@pytest.mark.parametrize(
    "variant",
    ["normal", "descending latitude", "transposed", "winds on a latitude subset"],
)
def test_quiver_arrows_carry_the_wind_at_their_position(variant: str) -> None:
    """Each drawn arrow equals the (u, v) of the grid point it is drawn at."""
    field = _latlon()
    u = xr.DataArray(0.1 * LON_GRID - 3.0 + 0.0 * LAT_GRID, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    v = xr.DataArray(0.2 * LAT_GRID + 0.0 * LON_GRID, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    olr, wind_u, wind_v = field, u, v
    if variant == "descending latitude":
        olr, wind_u, wind_v = (a.isel(lat=slice(None, None, -1)) for a in (field, u, v))
    elif variant == "transposed":
        wind_u, wind_v = u.transpose("lon", "lat"), v.transpose("lon", "lat")
    elif variant == "winds on a latitude subset":
        wind_u, wind_v = u.sel(lat=slice(-20, 20)), v.sel(lat=slice(-20, 20))
    _, axes = plotting.plot_horizontal_structure(
        olr, wind_u, wind_v, use_cartopy=False, wind_overlay="vectors", quiver_stride=4
    )
    quivers = [c for c in axes.collections if isinstance(c, Quiver)]
    assert len(quivers) == 1
    quiver = quivers[0]
    positions = np.asarray(quiver.get_offsets())
    assert len(positions) > 50
    for (lon, lat), arrow_u, arrow_v in zip(positions, np.asarray(quiver.U).ravel(), np.asarray(quiver.V).ravel()):
        assert arrow_u == pytest.approx(float(u.sel(lon=lon, lat=lat)), abs=1e-6)
        assert arrow_v == pytest.approx(float(v.sel(lon=lon, lat=lat)), abs=1e-6)


@pytest.mark.parametrize("order", [("month", "lon"), ("lon", "month")])
def test_significance_hatching_sits_on_the_significant_cells(order) -> None:
    """The hatched region is the (month, lon) block with p < 0.05, whatever the dimension order."""
    field = _month_lon(amplitude=0.0)
    p_value = xr.ones_like(field)
    block = (p_value.month >= 3) & (p_value.month <= 5) & (p_value.lon >= 100) & (p_value.lon <= 150)
    p_value = p_value.where(~block, 0.01).transpose(*order)
    _, axes = plotting.plot_wave_monthly_longitude_comparison(
        [field], wave_names=["kelvin"], ncols=1, significance_fields=[p_value]
    )
    extents = []
    for collection in axes.flat[0].collections:
        if getattr(collection, "filled", False) and getattr(collection, "hatches", None) and any(collection.hatches):
            vertices = np.concatenate([path.vertices for path in collection.get_paths() if len(path.vertices)])
            extents.append((vertices[:, 0].min(), vertices[:, 0].max(), vertices[:, 1].min(), vertices[:, 1].max()))
    assert len(extents) == 1
    np.testing.assert_allclose(extents[0], [100.0, 150.0, 3.0, 5.0], atol=1e-6)


# ========================================================================== WK spectrum
def _wk_result(peaks_sym=(), peaks_asym=(), max_wn=20):
    frequency = np.arange(0.0, 0.5 + 1e-9, 1 / 96)
    wavenumber = np.arange(-max_wn, max_wn + 1)
    wavenumber_grid, frequency_grid = np.meshgrid(wavenumber, frequency)

    def build(peaks):
        total = np.ones_like(frequency_grid)
        for k, f, amplitude in peaks:
            total += amplitude * np.exp(-(((wavenumber_grid - k) / 1.5) ** 2 + ((frequency_grid - f) / 0.012) ** 2))
        return xr.DataArray(total, dims=("frequency", "wavenumber"), coords={"frequency": frequency, "wavenumber": wavenumber})

    background = xr.DataArray(
        np.ones_like(frequency_grid),
        dims=("frequency", "wavenumber"),
        coords={"frequency": frequency, "wavenumber": wavenumber},
    )
    return WKAnalysisResult(
        power_symmetric=build(peaks_sym), power_antisymmetric=build(peaks_asym), background=background
    )


def test_wk_peaks_are_painted_at_their_wavenumber_and_frequency() -> None:
    f_a, f_b, f_c = 9 / 96, 20 / 96, 25 / 96  # frequency grid nodes, so the peak values are exact
    result = _wk_result(peaks_sym=[(4, f_a, 1.4), (-8, f_b, 9.0)], peaks_asym=[(-3, f_c, 1.5)])
    figure, axes = plotting.plot_wk_spectrum(result, add_matsuno_lines=False, annotate_filter_boxes=False)
    sym, asym = (_filled(axis)[0] for axis in axes)
    # 9x background is far above the highest level (3.6): painted with the top colour, colourbar extended
    assert sym.extend == "max"
    assert not _is_hole(_pixel(figure, axes[0], -8.0, f_b), axes[0])
    np.testing.assert_allclose(_pixel(figure, axes[0], -8.0, f_b), _band_rgb(sym, 10.0), atol=0.02)
    # a moderate peak: band colour at its own place, background colour at the mirrored wavenumber
    # (probe points are kept off the axis ticks, where grid lines would tint the pixel)
    np.testing.assert_allclose(_pixel(figure, axes[0], 4.0, f_a), _band_rgb(sym, 2.4), atol=0.02)
    np.testing.assert_allclose(_pixel(figure, axes[0], -4.0, f_a), _band_rgb(sym, 1.0), atol=0.02)
    np.testing.assert_allclose(_pixel(figure, axes[1], -3.0, f_c), _band_rgb(asym, 2.5), atol=0.02)
    np.testing.assert_allclose(_pixel(figure, axes[1], 3.0, f_c), _band_rgb(asym, 1.0), atol=0.02)


def test_wk_colorbar_is_not_extended_when_every_peak_fits() -> None:
    result = _wk_result(peaks_sym=[(5, 0.09, 1.2)], peaks_asym=[(-3, 0.26, 1.5)])
    _, axes = plotting.plot_wk_spectrum(result, add_matsuno_lines=False, annotate_filter_boxes=False)
    assert _filled(axes[0])[0].extend == "neither" and _filled(axes[1])[0].extend == "neither"


def test_wk_period_guides_are_labelled_with_their_period() -> None:
    _, axes = plotting.plot_wk_spectrum(
        _wk_result(), cpd_lines=(2.5, 5.0, 30.0), add_matsuno_lines=False, annotate_filter_boxes=False
    )
    assert sorted(text.get_text() for text in axes[0].texts) == ["2.5d", "30d", "5d"]


_GRAVITY, _EARTH_RADIUS, _OMEGA = 9.8, 6371e3, 7.292e-5


def _nondimensional(k, frequency, depth):
    """(k, omega) of the equatorial beta-plane scales for an equivalent depth: x*sqrt(beta/c), t*sqrt(beta*c)."""
    beta = 2.0 * _OMEGA / _EARTH_RADIUS
    speed = np.sqrt(_GRAVITY * depth)
    return (k / _EARTH_RADIUS) * np.sqrt(speed / beta), (2.0 * np.pi * frequency / 86400.0) / np.sqrt(speed * beta)


def _solid_lines(axis, colour):
    return [
        line
        for line in axis.lines
        if mcolors.same_color(line.get_color(), colour) and len(line.get_xdata()) > 3 and line.get_linestyle() == "-"
    ]


@pytest.mark.parametrize(
    ("panel", "colour", "relation", "sign"),
    [
        (0, "#0f766e", "kelvin", +1),  # Kelvin: omega = k
        (0, "#c2410c", "n1", -1),  # ER n=1: omega^3 - (k^2 + 3) omega - k = 0
        (1, "#7c3aed", "n0", -1),  # MRG
        (1, "#b91c1c", "n0", +1),  # EIG n=0: omega^2 - k omega - 1 = 0
    ],
)
def test_dispersion_guides_satisfy_the_matsuno_relations(panel, colour, relation, sign) -> None:
    depths = (12.0, 25.0, 50.0)
    _, axes = plotting.plot_wk_spectrum(_wk_result(), equivalent_depths=depths)
    lines = _solid_lines(axes[panel], colour)
    assert len(lines) == len(depths)  # one curve per equivalent depth
    worst = 0.0
    for line in lines:
        k = np.asarray(line.get_xdata(), dtype=float)
        frequency = np.asarray(line.get_ydata(), dtype=float)
        assert (np.sign(k) == sign).all()  # Kelvin / EIG eastward, ER / MRG westward
        residuals = []
        for depth in depths:
            kk, omega = _nondimensional(k, frequency, depth)
            if relation == "kelvin":
                residuals.append(np.abs(omega - kk))
            elif relation == "n1":
                residuals.append(np.abs(omega**3 - (kk**2 + 3.0) * omega - kk))
            else:
                residuals.append(np.abs(omega**2 - kk * omega - 1.0))
        worst = max(worst, float(np.min(residuals, axis=0).max()))  # each point belongs to one of the depths
    assert worst < 5e-3


def test_filter_boxes_and_envelope_agree_with_the_configured_bands() -> None:
    _, axes = plotting.plot_wk_spectrum(_wk_result())
    for axis, wave in ((axes[0], "mjo"), (axes[1], "td")):
        spec = DEFAULT_WAVE_SPECS[wave]
        box = next(p for p in axis.patches if isinstance(p, Rectangle))
        x0, y0 = box.get_xy()
        np.testing.assert_allclose(
            [x0, x0 + box.get_width(), y0, y0 + box.get_height()],
            [*spec.wavenumber, 1.0 / spec.period_days[1], 1.0 / spec.period_days[0]],
        )
    spec = DEFAULT_WAVE_SPECS["kelvin"]
    (k_edge,), (f_edge,) = spectrum.get_cckw_envelope_curve(he=spec.equivalent_depth)
    cpd_per_k = lambda depth: np.sqrt(_GRAVITY * depth) / (2 * np.pi * _EARTH_RADIUS) * 86400.0  # noqa: E731
    shallow, deep = spec.equivalent_depth[0], spec.equivalent_depth[-1]
    np.testing.assert_allclose(f_edge[[1, 2]], k_edge[[1, 2]] * cpd_per_k(shallow))  # lower edge: shallowest depth
    np.testing.assert_allclose(f_edge[[4, 5]], k_edge[[4, 5]] * cpd_per_k(deep))  # upper edge: deepest depth
    assert (k_edge.min(), k_edge.max()) == tuple(spec.wavenumber)
    assert f_edge.min() == pytest.approx(1.0 / spec.period_days[1])
    assert f_edge.max() == pytest.approx(1.0 / spec.period_days[0])


# ============================================================================ cross-spectrum panel
def _cross_spectrum_results(peak=1.0, threshold=0.3):
    """A hand-made cross-spectrum result: one coherent blob, phase vectors (V1, V2) = (-sin, cos) of a known phase."""
    frequency = np.arange(0.0, 0.5 + 1e-9, 1 / 96)
    wavenumber = np.arange(-15, 16)
    wave_grid, frequency_grid = np.meshgrid(wavenumber, frequency)
    coherence = 0.05 + (peak - 0.05) * np.exp(-(((wave_grid + 4) / 2.0) ** 2 + ((frequency_grid - 0.2) / 0.03) ** 2))
    phase = 0.1 * wave_grid + 2.0 * frequency_grid
    components = {"COH2": coherence, "V1": -np.sin(phase), "V2": np.cos(phase)}
    stc = xr.DataArray(
        np.stack(list(components.values())),
        dims=("component", "frequency", "wavenumber"),
        coords={"component": list(components), "frequency": frequency, "wavenumber": wavenumber},
    )
    return {"exp": {"STC": stc, "wave": wavenumber, "freq": frequency, "prob_coh2": np.array([threshold])}}, stc


def test_cross_spectrum_panel_paints_perfect_coherence_and_draws_the_given_phase_vectors() -> None:
    from tropical_wave_tools.cross_spectrum_analysis import plot_cross_spectrum_panel

    results, stc = _cross_spectrum_results()
    figure, axes = plot_cross_spectrum_panel(
        results, experiments=["exp"], figsize=(8, 4), dpi=60, vector_skip=2, verbose=False
    )
    axis = axes[0]
    contour_set = _filled(axis)[0]
    assert contour_set.extend in ("max", "both")  # coherence squared reaches 1, above the 0.8 top level
    peak_frequency = float(stc.frequency.sel(frequency=0.2, method="nearest"))
    assert not _is_hole(_pixel(figure, axis, -4.0, peak_frequency), axis)
    np.testing.assert_allclose(_pixel(figure, axis, -4.0, peak_frequency), _band_rgb(contour_set, 1.0), atol=0.02)
    assert _is_hole(_pixel(figure, axis, 10.0, 0.4), axis)  # not significant: left unshaded
    quivers = [c for c in axis.collections if isinstance(c, Quiver)]
    assert len(quivers) == 1
    quiver = quivers[0]
    positions = np.asarray(quiver.get_offsets())
    arrow_u, arrow_v = np.asarray(quiver.U, dtype=float).ravel(), np.asarray(quiver.V, dtype=float).ravel()
    masked = np.ma.getmaskarray(np.ma.array(arrow_u, mask=quiver.Umask))  # a Quiver fills masked (NaN) arrows with 1
    coherence = stc.sel(component="COH2")
    drawn = 0
    for (k, f), u, v, is_masked in zip(positions, arrow_u, arrow_v, masked):
        if coherence.sel(wavenumber=k, frequency=f) >= 0.3:  # significant: the vector of the given phase
            assert not is_masked
            assert u == pytest.approx(float(stc.sel(component="V1", wavenumber=k, frequency=f)), abs=1e-9)
            assert v == pytest.approx(float(stc.sel(component="V2", wavenumber=k, frequency=f)), abs=1e-9)
            drawn += 1
        else:  # not significant: no arrow
            assert is_masked
    assert drawn > 5
    assert axis.get_title(loc="left") == "(a) EXP" and axis.get_xlim() == (-15.0, 15.0) and axis.get_ylim() == (0.0, 0.5)


def test_cross_spectrum_panel_colorbar_is_not_extended_when_coherence_fits_the_levels() -> None:
    from tropical_wave_tools.cross_spectrum_analysis import plot_cross_spectrum_panel

    results, _ = _cross_spectrum_results(peak=0.7)
    _, axes = plot_cross_spectrum_panel(results, experiments=["exp"], figsize=(6, 3), dpi=40, verbose=False)
    assert _filled(axes[0])[0].extend in ("neither", "min")  # nothing above 0.8; below 0.2 is possible near the threshold


# ============================================================================ line plots
def test_monthly_cycle_and_time_series_plot_their_data() -> None:
    months = np.arange(1, 13)
    first, second = np.sin(months / 2.0) + 2.0, np.cos(months / 3.0) + 3.0
    _, axis = plotting.plot_monthly_cycle(months, {"A": first, "B": second}, title="t", ylabel="y")
    np.testing.assert_allclose(axis.lines[0].get_ydata(), first)
    np.testing.assert_allclose(axis.lines[1].get_ydata(), second)
    np.testing.assert_allclose(axis.lines[0].get_xdata(), months)
    assert [t.get_text() for t in axis.get_legend().get_texts()] == ["A", "B"]
    series = xr.DataArray(
        np.random.default_rng(0).random(30), dims="time", coords={"time": pd.date_range("2000-01-01", periods=30)}
    )
    _, axis = plotting.plot_time_series(series, title="x")
    np.testing.assert_allclose(axis.lines[0].get_ydata(), series.values)


def test_annual_trend_panel_reports_the_fitted_slope_per_decade() -> None:
    years = np.arange(1980, 2000)
    values = 0.05 * (years - 1980) + 2.0 + np.random.default_rng(1).normal(0.0, 0.01, years.size)
    series = xr.DataArray(values, dims="year", coords={"year": years})
    _, axes = plotting.plot_wave_annual_trend_comparison(
        {"kelvin": series}, wave_names=["kelvin"], ncols=1, trend_pvalues={"kelvin": 0.0123}
    )
    axis = axes.flat[0]
    slope, intercept = np.polyfit(years, values, 1)
    assert f"{slope * 10.0:+.2f} per decade" in axis.get_title()
    assert "p=0.012" in axis.get_title() and axis.get_title().rstrip().endswith("*")
    np.testing.assert_allclose(axis.lines[1].get_ydata(), slope * years + intercept)


def test_monthly_comparison_marks_only_the_significant_months_on_the_curve() -> None:
    months = np.arange(1, 13)
    cycles = {"kelvin": {"OLR": np.linspace(1.0, 3.0, 12), "U850": np.linspace(2.0, 1.0, 12)}}
    pvalues = {
        "kelvin": {
            "OLR": xr.DataArray(np.where(np.isin(months, [2, 5, 9]), 0.01, 0.5), dims="month"),
            "U850": xr.DataArray(np.full(12, 0.5), dims="month"),
        }
    }
    _, axes = plotting.plot_wave_monthly_cycle_comparison(
        cycles, wave_names=["kelvin"], ncols=1, monthly_significance=pvalues
    )
    markers = np.concatenate([np.asarray(c.get_offsets()) for c in axes.flat[0].collections if len(c.get_offsets())])
    assert sorted(markers[:, 0]) == [2.0, 5.0, 9.0]
    np.testing.assert_allclose(markers[np.argsort(markers[:, 0]), 1], np.linspace(1.0, 3.0, 12)[[1, 4, 8]])
    _, axes = plotting.plot_wave_monthly_cycle_comparison(
        cycles, wave_names=["kelvin"], ncols=1, normalize_each_series=True
    )
    assert np.mean(axes.flat[0].lines[0].get_ydata()) == pytest.approx(1.0)


def test_case05_variance_cycles_plot_the_means_and_scale_to_mean_plus_std() -> None:
    months = np.arange(1, 13)
    mean = xr.DataArray(
        np.random.default_rng(2).random((2, 12)) * 8.0 + 1.0,
        dims=("wave", "month"),
        coords={"wave": ["kelvin", "er"], "month": months},
    )
    std = mean * 0.3
    _, axes = plotting.plot_case05_seasonal_variance_cycles(mean, std, wave_order=["kelvin", "er"])
    for axis, wave in zip(axes.flat, ("kelvin", "er")):
        line = next(line for line in axis.lines if len(line.get_xdata()) == 12)
        np.testing.assert_allclose(line.get_ydata(), mean.sel(wave=wave).values)
        assert axis.get_ylim()[1] >= float((mean.sel(wave=wave) + std.sel(wave=wave)).max())
    regions = ["africa", "west_pacific"]
    regional_mean = xr.concat([mean * (1.0 + index) for index in range(2)], dim="region").assign_coords(region=regions)
    regional_mean = regional_mean.transpose("region", "wave", "month")
    regional_std = regional_mean * 0.2
    _, axes = plotting.plot_case05_regional_variance_cycles(
        regional_mean, regional_std, region_order=regions, wave_order=["kelvin", "er"]
    )
    for panel, region in enumerate(regions):
        axis = axes.flat[panel]
        lines = [line for line in axis.lines if len(line.get_xdata()) == 12]
        for wave in ("kelvin", "er"):  # errorbar also draws cap lines, so look for the data line
            expected = regional_mean.sel(region=region, wave=wave).values
            assert any(np.allclose(line.get_ydata(), expected) for line in lines), (region, wave)
        assert axis.get_ylim()[1] >= float((regional_mean + regional_std).sel(region=region).max())


# ===================================================================== EOF figures
def test_eof_figure_titles_series_and_validation() -> None:
    rng = np.random.default_rng(0)
    eofs = xr.concat([_latlon(phase=30.0 * mode).expand_dims(mode=[mode]) for mode in (1, 2)], dim="mode")
    time = pd.date_range("2000-01-01", periods=60)
    pcs = xr.DataArray(rng.standard_normal((2, 60)), dims=("mode", "time"), coords={"mode": [1, 2], "time": time})
    _, axes = plotting.plot_eof_spatial_patterns_and_pcs(eofs, pcs, [37.2, 11.5], use_cartopy=False)
    assert axes[0, 0].get_title() == "EOF 1 (37.2%)" and axes[1, 0].get_title() == "EOF 2 (11.5%)"
    np.testing.assert_allclose(axes[1, 2].lines[0].get_ydata(), pcs.sel(mode=2).values)
    _, axes = plotting.plot_eof_spatial_patterns_and_pcs(eofs, pcs, [37.2, 11.5], modes=(2,), use_cartopy=False)
    assert axes[0, 0].get_title() == "EOF 2 (11.5%)"
    with pytest.raises(ValueError, match="mode/lat/lon"):
        plotting.plot_eof_spatial_patterns_and_pcs(eofs.isel(mode=0, drop=True), pcs, [1.0], use_cartopy=False)
    _, axes = plotting.plot_multiwave_eof_summary(
        [eofs.sel(mode=1), eofs.sel(mode=2)],
        [pcs.sel(mode=1), pcs.sel(mode=2)],
        [30.0, 20.0],
        wave_names=["kelvin", "er"],
        use_cartopy=False,
    )
    assert axes[0, 0].get_title() == "KELVIN EOF-1 (30.0%)" and axes[1, 0].get_title() == "ER EOF-1 (20.0%)"
    np.testing.assert_allclose(axes[1, 2].lines[0].get_ydata(), pcs.sel(mode=2).values)


# ============================================================================== helpers
@pytest.mark.parametrize(
    ("value", "label"),
    [
        (0, "0"), (360, "0"), (-360, "0"), (180, "180"), (-180, "180"), (30, "30E"), (270, "90W"),
        (-90, "90W"), (205.2, "155W"), (359.6, "0"), (0.4, "0"), (179.6, "180"), (180.4, "180"),
    ],
)  # fmt: skip
def test_longitude_labels(value: float, label: str) -> None:
    assert _geo._format_longitude_label(value) == label


def test_atlas_label_formatters_share_the_convention() -> None:
    from tropical_wave_tools import atlas

    for value in (0, 30, 180, 270, 359.6, 179.6):
        assert atlas._format_lon_label(value) == _geo._format_longitude_label(value)
    assert [atlas._format_lat_label(v) for v in (0.0, 5.0, -12.5, 7.25)] == ["0N", "5N", "12.5S", "7.2N"]


def test_nice_step_is_a_round_number_not_below_the_target_step() -> None:
    for span in (1e-9, 3e-4, 0.07, 0.9, 1.0, 3.3, 17.0, 250.0, 1e5):
        for steps in (4, 6, 8, 16):
            for integer in (False, True):
                step = _style._nice_step(span, target_steps=steps, integer=integer)
                mantissa = step / 10 ** np.floor(np.log10(step))
                allowed = (1, 2, 3, 5, 10) if integer else (1, 2, 2.5, 5, 10)
                assert (integer and step == 1.0) or any(np.isclose(mantissa, a) for a in allowed), (span, steps, integer)
                assert step >= span / steps * (1 - 1e-12) or (integer and step >= 1.0)
    assert _style._nice_step(0.0) == 1.0 and _style._nice_step(np.nan) == 1.0


@pytest.mark.parametrize("scale", [0.03, 0.37, 1.0, 4.4, 73.0])
def test_level_helpers_cover_the_data_and_are_symmetric(scale: float) -> None:
    values = np.random.default_rng(0).standard_normal(500) * scale
    peak = np.abs(values).max()
    levels, ticks = _style._integer_levels_from_data(values, symmetric=True)
    assert levels[0] <= -peak and levels[-1] >= peak and np.isclose(levels[0], -levels[-1])
    assert np.any(np.isclose(levels, 0.0)) and np.allclose(np.diff(levels), np.diff(levels)[0])
    assert ticks[0] == levels[0] and ticks[-1] == levels[-1]
    signed = _style._signed_levels_from_data(values)
    assert not np.any(np.isclose(signed, 0.0)) and signed[-1] >= peak - 1e-12 and np.allclose(signed, -signed[::-1])
    continuous, _, vmax = _style._continuous_symmetric_levels_from_data(xr.DataArray(values), quantile=0.98)
    assert np.isclose(continuous[0], -vmax) and vmax >= np.quantile(np.abs(values), 0.98) - 1e-12
    scientific, _, bound = _style._scientific_integer_levels_from_data(values)
    assert bound >= peak and np.isclose(scientific[0], -scientific[-1])
    lo, hi = _style._integer_levels_from_data(np.array([2.1, 7.8, 5.5]), symmetric=False)[0][[0, -1]]
    assert lo <= 2.1 and hi >= 7.8
    assert len(set(_style._filled_levels(2.0, 2.0))) > 1  # a degenerate range is widened


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_unit_labels_and_robust_limits() -> None:
    assert _style._format_units_for_mathtext("m s^-1") == "m s$^{-1}$"
    assert _style._format_units_for_mathtext("W m-2") == "W m-2"
    data = xr.DataArray(np.r_[-np.arange(1, 100), 250.0])
    assert _style._symmetric_limit(data, quantile=0.98) == pytest.approx(np.quantile(np.abs(data.values), 0.98))
    assert _style._symmetric_limit(xr.DataArray(np.zeros(5))) == 1.0
    assert _style._symmetric_limit(xr.DataArray(np.full(5, np.nan))) == 1.0


def test_longitude_window_keeps_each_value_with_its_wrapped_longitude() -> None:
    lat = np.arange(-10.0, 10.1, 5.0)
    lon = np.arange(0.0, 360.0, 10.0)
    data = xr.DataArray(
        np.arange(lat.size * lon.size, dtype=float).reshape(lat.size, lon.size),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    for centre, half_width in ((180, 60), (20, 70), (350, 40), (0, 90), (100, 85)):
        window = _geo._subset_longitude_window(data, center_lon=centre, half_width=half_width)
        assert np.all(np.diff(window.lon.values) > 0)
        assert np.all(np.abs(window.lon.values - centre) <= half_width + 1e-9)
        for new_lon in window.lon.values:
            np.testing.assert_array_equal(window.sel(lon=new_lon).values, data.sel(lon=new_lon % 360).values)
    assert _geo._subset_longitude_window(data, center_lon=0.0, half_width=1.0).sizes["lon"] == lon.size  # < 3 points


@needs_cartopy
def test_cyclic_point_is_added_once_and_only_for_global_grids() -> None:
    lat = np.arange(-10.0, 10.1, 5.0)
    lon = np.arange(0.0, 360.0, 10.0)
    data = xr.DataArray(
        np.arange(lat.size * lon.size, dtype=float).reshape(lat.size, lon.size),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    closed = _geo._prepare_projected_field(data)
    assert closed.sizes["lon"] == lon.size + 1
    np.testing.assert_array_equal(closed.isel(lon=-1).values, data.isel(lon=0).values)
    assert closed.lon.values[-1] == pytest.approx(360.0)
    again = _geo._prepare_projected_field(closed)  # a stored cyclic point must not be added a second time
    assert again.sizes["lon"] == closed.sizes["lon"]
    assert _geo._prepare_projected_field(data.sel(lon=slice(100, 200))).sizes["lon"] == data.sel(lon=slice(100, 200)).sizes["lon"]
    wrapped = data.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon")
    assert _geo._prepare_projected_field(wrapped).lon.values[-1] == pytest.approx(wrapped.lon.values[-1] + 10.0)
    assert _geo._prepare_projected_field(data.transpose("lon", "lat")).dims[-2:] == ("lat", "lon")


def test_lat_lon_last_only_reorders_the_horizontal_dimensions() -> None:
    cube = xr.DataArray(np.zeros((3, 4, 5)), dims=("lon", "time", "lat"), coords={"lon": np.arange(3)})
    assert _geo._lat_lon_last(cube).dims == ("time", "lat", "lon")
    vector = xr.DataArray(np.zeros(4), dims="time")
    assert _geo._lat_lon_last(vector) is vector


def test_weak_signal_note_can_be_switched_off() -> None:
    """A field in other units (here 1e-6 s^-1) is not 'weak'; the 0.1 threshold is a parameter."""
    small = abs(_latlon(amplitude=0.0, positive=True)) * 1e-6
    figure, _ = plotting.plot_wave_spatial_comparison([small, small], titles=["a", "b"], use_cartopy=False)
    assert any("Weak signal" in text.get_text() for axis in figure.axes for text in axis.texts)  # default keeps the note
    plt.close(figure)
    figure, _ = plotting.plot_wave_spatial_comparison(
        [small, small], titles=["a", "b"], use_cartopy=False, weak_signal_threshold=None
    )
    assert not any("Weak signal" in text.get_text() for axis in figure.axes for text in axis.texts)


def test_save_figure_creates_folders_and_honours_dpi(tmp_path) -> None:
    from PIL import Image

    figure, axis = plt.subplots(figsize=(4, 3))
    axis.plot([0, 1])
    path = plotting.save_figure(figure, tmp_path / "nested" / "figure.png", dpi=100)
    assert path.exists()
    width, height = Image.open(path).size
    assert 300 <= width <= 420 and 220 <= height <= 320  # 4 x 3 inches at 100 dpi, cropped by bbox_inches="tight"
