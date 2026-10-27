"""Compare ``WaveFilter._kf_filter`` with NCL ``kf_filter`` output (golden files).

Regenerating the golden files needs NCL (see ``ncl_generate_reference.ncl``); comparing
them does not. Run ``python validation/compare_with_ncl.py`` for a per-band report;
the exit status is non-zero if any band exceeds the thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from tropical_wave_tools.filters import WaveFilter

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "reference"
INPUTS = {"filter": "filter_input.nc", "real": "real_input.nc"}
# Start loose, tighten once the observed agreement is known.
MIN_CORR = 0.9999
MAX_NRMSE = 0.01


def python_filter(wave: str, x: xr.DataArray) -> np.ndarray:
    """Filter every latitude row of ``x`` (time, lat, lon) with the Python implementation."""
    wave_filter = WaveFilter()
    spec = wave_filter.get_wave_params(wave)
    rows = [
        wave_filter._kf_filter(
            x.isel(lat=j).values,
            lon=x["lon"].values,
            obs_per_day=1,
            period_days=spec.period_days,
            wavenumber=spec.wavenumber,
            equivalent_depth=spec.equivalent_depth,
            wave_name=wave_filter._internal_wave_name(wave),
            meridional_mode=spec.meridional_mode,
        )
        for j in range(x.sizes["lat"])
    ]
    return np.stack(rows, axis=1)


def compare(wave: str, x: xr.DataArray, reference: xr.DataArray) -> dict[str, float]:
    mine = python_filter(wave, x)
    ref = np.asarray(reference.transpose("time", "lat", "lon").values, dtype=float)
    a, b = mine.ravel(), ref.ravel()
    return {
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "nrmse": float(np.sqrt(np.mean((a - b) ** 2)) / np.std(b)),
        "amp_ratio": float(np.std(a) / np.std(b)),
    }


def reference_files() -> list[Path]:
    """Golden files named ``ncl_<input>_<wave>.nc``."""
    return sorted(REFERENCE_DIR.glob("ncl_*_*.nc"))


def split_name(path: Path) -> tuple[str, str]:
    _, source, wave = path.stem.split("_", 2)
    return source, wave


def main() -> int:
    files = reference_files()
    if not files:
        print("No golden files in validation/reference; see ncl_generate_reference.ncl.")
        return 2
    failed = False
    print(f"{'input':8s} {'wave':8s} {'corr':>9s} {'nrmse':>9s} {'amp':>8s}")
    for path in files:
        source, wave = split_name(path)
        x = xr.open_dataarray(HERE / INPUTS[source])
        stats = compare(wave, x, xr.open_dataset(path)["filtered"])
        ok = stats["corr"] >= MIN_CORR and stats["nrmse"] <= MAX_NRMSE
        failed |= not ok
        print(
            f"{source:8s} {wave:8s} {stats['corr']:9.6f} {stats['nrmse']:9.6f} "
            f"{stats['amp_ratio']:8.5f}  {'OK' if ok else 'FAIL'}"
        )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
