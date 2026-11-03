"""Reanalysis (ERA5) workflow around :mod:`tropical_wave_tools.kelvin_events`.

Kelvin waves circle the globe, so filtering and tracking always run on a
*global* longitude grid.  A basin such as the Pacific is selected afterwards,
by keeping events whose ridge spends enough time inside the longitude window.

The loader handles the usual ERA5 conventions: ``valid_time`` / ``latitude`` /
``longitude`` names, descending latitude, a ``pressure_level`` dimension,
geopotential (m2 s-2) instead of height, sub-daily data and 0.25 degree grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from tropical_wave_tools.io import (
    ensure_time_lat_lon,
    normalize_longitude,
    rename_standard_coordinates,
    sort_latitude,
    squeeze_singleton_extra_dims,
)
from tropical_wave_tools.kelvin_events import (
    KelvinEventCatalog,
    KelvinEventConfig,
    event_composite,
    identify_kelvin_events,
    kelvin_band_filter,
)
from tropical_wave_tools.matsuno import GRAVITY

__all__ = [
    "STRUCTURE_CLASSES",
    "WorkflowResult",
    "load_reanalysis_field",
    "plot_region_report",
    "plot_structure_report",
    "regrid_global",
    "run_regional_workflow",
    "structure_summary",
    "summarize_events",
]

STRUCTURE_CLASSES = ("northern", "equatorial", "symmetric_off_equator", "southern", "weak")

_LEVEL_DIMS = ("pressure_level", "level", "plev", "isobaricInhPa", "lev")
_SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


# --------------------------------------------------------------------------- #
# Loading and regridding
# --------------------------------------------------------------------------- #
def _pad_longitude(da: xr.DataArray) -> xr.DataArray:
    """Add one wrapped column on each side so interpolation is periodic."""
    left = da.isel(lon=-1).assign_coords(lon=float(da.lon[-1]) - 360.0)
    right = da.isel(lon=0).assign_coords(lon=float(da.lon[0]) + 360.0)
    return xr.concat([left, da, right], dim="lon")


def regrid_global(da: xr.DataArray, resolution: float, lat_limit: float) -> xr.DataArray:
    """Block-average (if finer than ``resolution``) then interpolate to a symmetric global grid.

    The target latitudes are ``-L ... +L`` in steps of ``resolution`` (``L`` is
    ``lat_limit`` reduced to what the source covers), so the equator is a grid
    row and the grid is mirror-symmetric, as the symmetry diagnostics require.
    """
    da = da.transpose("time", "lat", "lon")
    src_res = float(abs(np.median(np.diff(da.lon.values))))
    factor = round(resolution / src_res) if src_res > 0 else 1
    if factor > 1:
        lat_step = float(abs(np.median(np.diff(da.lat.values))))
        lat_factor = max(round(resolution / lat_step), 1)
        da = da.coarsen(lon=factor, lat=lat_factor, boundary="trim").mean()
    coverage = min(abs(float(da.lat.min())), abs(float(da.lat.max())))
    limit = np.floor(min(lat_limit, coverage) / resolution) * resolution
    lat_target = np.arange(-limit, limit + 0.5 * resolution, resolution)
    lon_target = np.arange(0.0, 360.0, resolution)
    padded = _pad_longitude(da.sortby("lon"))
    out = padded.interp(lat=lat_target, lon=lon_target)
    out.attrs.update(da.attrs)
    return out


def load_reanalysis_field(
    path: "str | Path",
    variable: Optional[str] = None,
    *,
    level: Optional[float] = None,
    lat_range: Tuple[float, float] = (-20.0, 20.0),
    time_range: Optional[Tuple[str, str]] = None,
    buffer_days: int = 30,
    factor: float = 1.0,
    offset: float = 0.0,
    geopotential: bool = False,
    regrid_deg: Optional[float] = None,
) -> xr.DataArray:
    """Load one reanalysis variable as a global ``(time, lat, lon)`` daily field.

    Parameters
    ----------
    level:
        Pressure level (hPa) to select when the file has a level dimension.
    time_range:
        ``(start, end)`` of the analysis window.  ``buffer_days`` extra days are
        read on each side so filter edge effects fall outside the window.
    factor, offset:
        ``value * factor + offset`` unit conversion (e.g. ``factor=-1/3600``
        to turn ERA5 hourly-accumulated ``ttr`` into OLR; check the ECMWF
        accumulation convention for your download).
    geopotential:
        Divide by g so ERA5 ``z`` (m2 s-2) becomes geopotential height (m).
    regrid_deg:
        Target resolution for a global, equator-symmetric grid.  Recommended
        (2.5) for 0.25 degree ERA5: 20 years of 0.25 degree data are ~14 GB per
        variable in memory.
    """
    dataset = xr.open_dataset(path)
    dataset = rename_standard_coordinates(dataset)
    name = variable or next(iter(dataset.data_vars))
    da = dataset[name]
    for dim in _LEVEL_DIMS:
        if dim in da.dims and level is not None:
            da = da.sel({dim: level}, method="nearest", drop=True)
    da = ensure_time_lat_lon(squeeze_singleton_extra_dims(da))
    da = sort_latitude(normalize_longitude(da, target="0_360"), ascending=True)

    if time_range is not None:
        start = pd.Timestamp(time_range[0]) - pd.Timedelta(days=buffer_days)
        end = pd.Timestamp(time_range[1]) + pd.Timedelta(days=buffer_days)
        da = da.sel(time=slice(start, end))
    pad = 3.0 if regrid_deg else 0.0
    da = da.sel(lat=slice(lat_range[0] - pad, lat_range[1] + pad))
    if da.sizes["time"] == 0 or da.sizes["lat"] == 0:
        raise ValueError(f"No data left in {path} after selecting time_range={time_range}, lat_range={lat_range}.")

    steps = np.diff(da.time.values)
    if steps.size and np.median(steps) < np.timedelta64(1, "D"):
        da = da.resample(time="1D").mean()
    attrs = dict(da.attrs)
    da = da.astype("float64") * factor + offset
    if geopotential:
        da = da / GRAVITY
        attrs["units"] = "m"
    da.attrs.update(attrs)
    if regrid_deg:
        da = regrid_global(da, regrid_deg, lat_limit=min(abs(lat_range[0]), abs(lat_range[1])))
    return da.load()


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summarize_events(events: pd.DataFrame, *, years: Optional[float] = None) -> pd.DataFrame:
    """Counts and medians by ``kind`` and season.

    ``events_per_year`` needs ``years`` (length of the analysis window).
    Medians of ``conv_axis_lat``, ``u_symmetry_index``, ``v_to_u_ratio``,
    ``coupling_corr`` and the axis/fixed-band peak ratio answer the central
    question directly: how far off the equator is the convection of the events
    whose wind pattern is equatorially symmetric?
    """
    if events.empty:
        return pd.DataFrame()
    table = events.copy()
    table["season"] = pd.to_datetime(table["peak_time"]).dt.month.map(_SEASONS)
    if {"conv_peak", "conv_peak_fixed_band"} <= set(table.columns):
        with np.errstate(invalid="ignore", divide="ignore"):
            table["peak_ratio_axis_over_fixed"] = table["conv_peak"] / table["conv_peak_fixed_band"].where(
                table["conv_peak_fixed_band"] > 0
            )
    value_columns = [
        c
        for c in (
            "speed_ms",
            "peak_amp",
            "conv_peak_lat",
            "conv_symmetry_index",
            "conv_nh_fraction",
            "struct_peak",
            "conv_axis_lat",
            "conv_offset_deg",
            "u_symmetry_index",
            "v_to_u_ratio",
            "coupling_corr",
            "peak_ratio_axis_over_fixed",
        )
        if c in table.columns
    ]
    groups = list(table.groupby(["kind", "season"]))
    groups += [((kind, "ALL"), g) for kind, g in table.groupby("kind")]
    rows = []
    for (kind, season), group in groups:
        row: Dict[str, object] = {"kind": kind, "season": season, "n": len(group)}
        if years:
            row["events_per_year"] = len(group) / years
        for column in value_columns:
            row[f"median_{column}"] = float(group[column].median())
        rows.append(row)
    order = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3, "ALL": 4}
    out = pd.DataFrame(rows)
    out["_s"] = out["season"].map(order)
    return out.sort_values(["kind", "_s"]).drop(columns="_s").reset_index(drop=True)


def structure_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Counts and fractions of each convective meridional-structure class.

    Rows are ``kind`` x season (plus ``ALL``).  ``conv_structure`` comes from
    the wave-following ``-OLR_K`` profile over the full latitude band, with no
    assumption about where the convection should be.  Fractions of small
    groups are noisy; look at ``n`` before reading them.
    """
    if events.empty or "conv_structure" not in events.columns:
        return pd.DataFrame()
    table = events.copy()
    table["season"] = pd.to_datetime(table["peak_time"]).dt.month.map(_SEASONS)
    groups = list(table.groupby(["kind", "season"])) + [((k, "ALL"), g) for k, g in table.groupby("kind")]
    rows = []
    for (kind, season), group in groups:
        counts = group["conv_structure"].value_counts()
        row: Dict[str, object] = {"kind": kind, "season": season, "n": len(group)}
        for cls in STRUCTURE_CLASSES:
            row[f"n_{cls}"] = int(counts.get(cls, 0))
            row[f"frac_{cls}"] = counts.get(cls, 0) / len(group)
        rows.append(row)
    order = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3, "ALL": 4}
    out = pd.DataFrame(rows)
    out["_s"] = out["season"].map(order)
    return out.sort_values(["kind", "_s"]).drop(columns="_s").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #
@dataclass
class WorkflowResult:
    """Outputs of :func:`run_regional_workflow`."""

    catalog: KelvinEventCatalog
    """Events restricted to the region and analysis window."""
    full_catalog: KelvinEventCatalog
    """All events on the global grid, including padding periods."""
    summary: pd.DataFrame
    filtered: Dict[str, xr.DataArray]
    figure: Optional[Path] = None
    structure: Optional[pd.DataFrame] = None
    structure_figure: Optional[Path] = None


def run_regional_workflow(
    *,
    u: xr.DataArray,
    z: Optional[xr.DataArray] = None,
    v: Optional[xr.DataArray] = None,
    olr: Optional[xr.DataArray] = None,
    config: Optional[KelvinEventConfig] = None,
    region_lon: Optional[Tuple[float, float]] = (120.0, 280.0),
    time_range: Optional[Tuple[str, str]] = None,
    min_points: int = 3,
    output_dir: "str | Path | None" = None,
    make_figures: bool = True,
) -> WorkflowResult:
    """Filter once on the global grid, identify events, then report one basin.

    The Kelvin-band fields are computed once and reused for identification and
    for the composites in the report figure.
    """
    cfg = config or KelvinEventConfig()
    fields = {"u": u, "z": z, "v": v, "olr": olr}
    filtered: Dict[str, xr.DataArray] = {}
    for key, da in fields.items():
        if da is None:
            continue
        filtered[key] = kelvin_band_filter(
            da,
            period_days=cfg.period_days,
            wavenumber=cfg.wavenumber,
            equivalent_depth=cfg.equivalent_depth,
            obs_per_day=cfg.obs_per_day,
            remove_seasonal_cycle=cfg.remove_seasonal_cycle,
            n_harm=cfg.n_harm,
        )
    full = identify_kelvin_events(
        u=filtered.get("u"),
        z=filtered.get("z"),
        v=filtered.get("v"),
        olr=filtered.get("olr"),
        config=cfg,
        prefiltered=True,
    )
    selected = full
    if time_range is not None:
        selected = selected.in_window(*time_range)
    if region_lon is not None:
        selected = selected.in_region(region_lon, min_points=min_points)

    if time_range is not None:
        years = (pd.Timestamp(time_range[1]) - pd.Timestamp(time_range[0])).days / 365.25
    else:
        ref = next(iter(filtered.values()))
        years = (pd.Timestamp(ref.time.values[-1]) - pd.Timestamp(ref.time.values[0])).days / 365.25
    summary = summarize_events(selected.events, years=years)
    structure = structure_summary(selected.events)

    result = WorkflowResult(
        catalog=selected, full_catalog=full, summary=summary, filtered=filtered, structure=structure
    )
    if output_dir is not None:
        out = selected.save(output_dir)
        summary.to_csv(out / "summary.csv", index=False)
        if not structure.empty:
            structure.to_csv(out / "structure_summary.csv", index=False)
        if make_figures and not selected.events.empty:
            result.figure = plot_region_report(result, out / "kelvin_region_report.png", region_lon=region_lon)
            if selected.conv_profiles is not None:
                result.structure_figure = plot_structure_report(
                    result, out / "kelvin_structure_report.png", region_lon=region_lon
                )
    return result


# --------------------------------------------------------------------------- #
# Report figure
# --------------------------------------------------------------------------- #
def plot_region_report(
    result: WorkflowResult,
    output: "str | Path",
    *,
    region_lon: Optional[Tuple[float, float]] = None,
    kinds: Tuple[str, ...] = ("coupled",),
    half_width_deg: float = 60.0,
) -> Path:
    """Composite maps and statistics answering: is the wind symmetric while convection is north?

    Panels: (a-c) wave-following lag-0 composites of Kelvin-band ``u``, ``v``
    and ``-OLR``; (d) latitude profiles of the composite peak anomalies;
    (e) distribution of the convective-axis latitude; (f) distribution of the
    ``u`` symmetry index.  Composites use the events of ``kinds`` only.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = result.catalog.events
    sel = events[events["kind"].isin(list(kinds))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    title = f"Kelvin events, {', '.join(kinds)} (N={len(sel)})"
    if region_lon is not None:
        title += f", {region_lon[0]:g}-{region_lon[1]:g}E"
    fig.suptitle(title)

    comps: Dict[str, xr.DataArray] = {}
    for key, sign in (("u", 1.0), ("v", 1.0), ("olr", -1.0)):
        if key in result.filtered and len(sel):
            comp = event_composite(
                sign * result.filtered[key], events, kinds=kinds, lags_days=(0.0,), half_width_deg=half_width_deg
            )
            comps[key] = comp.isel(lag=0)
    for ax, (key, label) in zip(axes[0], (("u", "u (m/s)"), ("v", "v (m/s, u colour scale)"), ("olr", "-OLR_K (W/m2)"))):
        ax.set_title(label)
        if key not in comps:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            continue
        c = comps[key]
        vmax = float(np.nanmax(np.abs(c.values))) or 1.0
        if key == "v" and "u" in comps:
            # Show v on the u colour scale so a negligible v looks negligible.
            vmax = float(np.nanmax(np.abs(comps["u"].values))) or vmax
        mesh = ax.pcolormesh(c.rel_lon, c.lat, c.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set(xlabel="Longitude relative to wave centre (deg)", ylabel="Latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.85)

    ax = axes[1, 0]
    u_peak = None
    if "u" in comps:
        u_peak = float(np.abs(comps["u"]).max("rel_lon").max()) or None
    for key, name, color in (
        ("u", "|u|", "#2a6fb0"),
        ("v", "|v| (relative to peak |u|)", "#6a9f3a"),
        ("olr", "-OLR (own peak)", "#c4561d"),
    ):
        if key not in comps:
            continue
        prof = np.abs(comps[key]).max("rel_lon") if key != "olr" else comps[key].max("rel_lon")
        # v is scaled by u's peak so its size is comparable to the circulation, not to itself.
        peak = (u_peak if key == "v" and u_peak else float(prof.max())) or 1.0
        ax.plot(prof.lat, prof / peak, lw=2, color=color, label=name)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set(title="Composite peak profile", xlabel="Latitude", ylabel="Normalised peak")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    lat_col = "conv_peak_lat" if "conv_peak_lat" in sel else "conv_axis_lat"
    if lat_col in sel and sel[lat_col].notna().any():
        strong = sel[sel.get("conv_structure", "") != "weak"] if "conv_structure" in sel else sel
        ax.hist(strong[lat_col].dropna(), bins=np.arange(-20, 21, 2.5), color="#c4561d")
        ax.axvline(strong[lat_col].median(), color="k", ls="--", lw=1)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set(title="Latitude of peak convection per event (no axis constraint)", xlabel="Latitude", ylabel="Events")

    ax = axes[1, 2]
    if "u_symmetry_index" in sel and sel["u_symmetry_index"].notna().any():
        ax.hist(sel["u_symmetry_index"].dropna(), bins=np.linspace(-1, 1, 21), color="#2a6fb0")
        ax.axvline(sel["u_symmetry_index"].median(), color="k", ls="--", lw=1)
    ax.set(title="u equatorial symmetry index (+1 = symmetric)", xlabel="Index", ylabel="Events")

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


_CLASS_COLORS = {
    "northern": "#c4561d",
    "equatorial": "#2a6fb0",
    "symmetric_off_equator": "#7b4fa3",
    "southern": "#6a9f3a",
    "weak": "#b0b0b0",
}


def plot_structure_report(
    result: WorkflowResult,
    output: "str | Path",
    *,
    region_lon: Optional[Tuple[float, float]] = None,
    kinds: Tuple[str, ...] = ("coupled", "dynamical_only"),
) -> Path:
    """Per-event convective structure: are the Kelvin waves' convection signals all north?

    (a) every event's wave-following ``-OLR_K`` latitude profile, each divided
    by its own peak and sorted by the latitude of that peak (weak events are
    drawn grey); (b) latitude of peak convection for non-weak events;
    (c) ``u`` symmetry versus convective symmetry per event; (d) share of
    each structure class by season.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    catalog = result.catalog
    events = catalog.events[catalog.events["kind"].isin(list(kinds))]
    profiles = catalog.conv_profiles
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    title = "Convective meridional structure of Kelvin events"
    if region_lon is not None:
        title += f" ({region_lon[0]:g}-{region_lon[1]:g}E)"
    fig.suptitle(f"{title}; N={len(events)} ({', '.join(kinds)})")

    ax = axes[0, 0]
    if profiles is not None and len(events):
        ids = [i for i in events.sort_values("conv_peak_lat")["event_id"] if i in set(profiles.event_id.values)]
        data = profiles.sel(event_id=ids)
        peak = data.max("lat").where(lambda x: x > 0)
        norm = (data / peak).values
        weak = (events.set_index("event_id").loc[ids, "conv_structure"] == "weak").to_numpy()
        norm = np.where(weak[:, None], np.nan, norm)
        mesh = ax.pcolormesh(
            data.lat, np.arange(len(ids)), np.clip(norm, 0, 1), cmap="magma_r", vmin=0, vmax=1, shading="auto"
        )
        ax.axvline(0.0, color="k", lw=0.6)
        ax.set(title="Per-event profile, own peak = 1 (sorted by peak latitude)", xlabel="Latitude", ylabel="Event (sorted)")
        fig.colorbar(mesh, ax=ax, shrink=0.85)

    ax = axes[0, 1]
    if "conv_peak_lat" in events:
        strong = events[events["conv_structure"] != "weak"]
        ax.hist(strong["conv_peak_lat"].dropna(), bins=np.arange(-17.5, 18, 2.5), color="#c4561d")
        if len(strong):
            ax.axvline(strong["conv_peak_lat"].median(), color="k", ls="--", lw=1)
        ax.axvline(0.0, color="0.6", lw=0.8)
        ax.set(title=f"Latitude of peak convection (non-weak, N={len(strong)})", xlabel="Latitude", ylabel="Events")

    ax = axes[1, 0]
    if {"u_symmetry_index", "conv_symmetry_index"} <= set(events.columns):
        for cls, group in events.groupby("conv_structure"):
            ax.scatter(group["u_symmetry_index"], group["conv_symmetry_index"], s=22, alpha=0.8,
                       color=_CLASS_COLORS.get(cls, "k"), label=f"{cls} ({len(group)})")
        ax.set(title="Wind vs convective equatorial symmetry", xlabel="u symmetry index", ylabel="OLR symmetry index",
               xlim=(-1.05, 1.05), ylim=(-0.05, 1.05))
        ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    table = result.structure
    if table is not None and not table.empty:
        sub = table[(table["kind"].isin(list(kinds)))]
        seasons = ["DJF", "MAM", "JJA", "SON", "ALL"]
        pooled = (
            sub.groupby("season")[[f"n_{c}" for c in STRUCTURE_CLASSES]].sum().reindex(seasons).fillna(0)
        )
        totals = pooled.sum(axis=1).replace(0, np.nan)
        bottom = np.zeros(len(seasons))
        for cls in STRUCTURE_CLASSES:
            frac = (pooled[f"n_{cls}"] / totals).fillna(0).to_numpy()
            ax.bar(seasons, frac, bottom=bottom, color=_CLASS_COLORS[cls], label=cls)
            bottom += frac
        for i in range(len(seasons)):
            ax.text(i, 1.02, f"n={int(totals.iloc[i]) if np.isfinite(totals.iloc[i]) else 0}", ha="center", fontsize=8)
        ax.set(title="Share of structure classes", ylabel="Fraction of events", ylim=(0, 1.1))
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
