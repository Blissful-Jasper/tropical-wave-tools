"""Compare ``WaveFilter._kf_filter`` with NCL ``kf_filter`` output.

Workflow
--------
1. ``python validation/make_input.py``
2. ``ncl validation/ncl_generate_reference.ncl``  (produces ``validation/reference/ncl_*.nc``)
3. ``python validation/compare_with_ncl.py``

Prints pattern correlation, normalised RMSE and amplitude ratio per wave band.
Exit status is non-zero if any band fails the thresholds, so it can gate CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from tropical_wave_tools.filters import WaveFilter

HERE = Path(__file__).resolve().parent
MIN_CORR = 0.999
MAX_NRMSE = 0.05


def compare(wave: str, x: xr.DataArray, reference: xr.DataArray) -> dict[str, float]:
    wave_filter = WaveFilter()
    spec = wave_filter.get_wave_params(wave)
    mine = wave_filter._kf_filter(
        x.values,
        lon=x["lon"].values,
        obs_per_day=1,
        period_days=spec.period_days,
        wavenumber=spec.wavenumber,
        equivalent_depth=spec.equivalent_depth,
        wave_name=wave,
        meridional_mode=spec.meridional_mode,
    )
    ref = np.asarray(reference.values, dtype=float)
    a, b = mine.ravel(), ref.ravel()
    return {
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "nrmse": float(np.sqrt(np.mean((a - b) ** 2)) / np.std(b)),
        "amp_ratio": float(np.std(a) / np.std(b)),
    }


def main() -> int:
    x = xr.open_dataarray(HERE / "input_equator.nc")
    failed = False
    print(f"{'wave':8s} {'corr':>8s} {'nrmse':>8s} {'amp':>8s}")
    for ref_path in sorted((HERE / "reference").glob("ncl_*.nc")):
        wave = ref_path.stem.removeprefix("ncl_")
        stats = compare(wave, x, xr.open_dataset(ref_path)["filtered"])
        ok = stats["corr"] >= MIN_CORR and stats["nrmse"] <= MAX_NRMSE
        failed |= not ok
        print(
            f"{wave:8s} {stats['corr']:8.5f} {stats['nrmse']:8.5f} "
            f"{stats['amp_ratio']:8.5f}  {'OK' if ok else 'FAIL'}"
        )
    if not any((HERE / "reference").glob("ncl_*.nc")):
        print("No NCL reference files found; see the module docstring.")
        return 2
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
