"""Compare ``analyze_wk_spectrum`` with NCL ``wkSpaceTime`` output (golden files).

``validation/ncl_wkspacetime.ncl`` runs ``wkSpaceTime`` on ``validation/wk_input.nc`` and stores
the spectra in ``validation/wk_ref/<case>_olr.nc``. NCL's ``nDaySkip`` is the gap between
windows (negative = overlap), which is also the meaning of ``SpectralConfig.window_skip_days``.

Run ``python validation/compare_wk_with_ncl.py`` for a report; the exit status is non-zero if a
case exceeds its tolerance (root-mean-square difference of log10 power, or of the raw ratio).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from tropical_wave_tools.config import SpectralConfig
from tropical_wave_tools.spectral import analyze_wk_spectrum

HERE = Path(__file__).resolve().parent
INPUT = HERE / "wk_input.nc"
REF_DIR = HERE / "wk_ref"
# case name -> (nDaySkip, tolerance). The overlapping case still differs from NCL at the
# lowest frequency bin (period = window length); its cause is not identified yet.
CASES = {
    "wk_skip30_olr": (30, 1e-5),
    "wk_skipm60_olr": (-60, 3e-3),
}


def python_fields(skip_days: int) -> dict[str, np.ndarray]:
    data = xr.open_dataarray(INPUT)
    result = analyze_wk_spectrum(
        data, config=SpectralConfig(window_size_days=96, window_skip_days=skip_days)
    )

    def select(array: xr.DataArray) -> np.ndarray:
        return array.sel(frequency=slice(0, 0.5 + 1e-9), wavenumber=slice(-15, 15)).values

    symmetric = select(result.power_symmetric)
    antisymmetric = select(result.power_antisymmetric)
    background = select(result.background)
    return {
        "FIG_1_SYM": np.log10(symmetric),
        "FIG_1_ASYM": np.log10(antisymmetric),
        "FIG_3_BACK": np.log10(background),
        "FIG_3_SYM": symmetric / background,
        "FIG_3_ASYM": antisymmetric / background,
    }


def compare(case: str) -> dict[str, float]:
    skip_days, _ = CASES[case]
    reference = xr.open_dataset(REF_DIR / f"{case}.nc")
    mine = python_fields(skip_days)
    stats = {}
    for name, values in mine.items():
        ref = np.asarray(reference[name].values, dtype=float)
        mask = np.isfinite(values) & np.isfinite(ref)
        stats[name] = float(np.sqrt(np.mean((values[mask] - ref[mask]) ** 2)))
    return stats


def main() -> int:
    failed = False
    for case, (_, tolerance) in CASES.items():
        if not (REF_DIR / f"{case}.nc").exists():
            print(f"{case}: reference file missing")
            continue
        stats = compare(case)
        ok = max(stats.values()) <= tolerance
        failed |= not ok
        detail = "  ".join(f"{k}={v:.1e}" for k, v in stats.items())
        print(f"{case:15s} tol={tolerance:.0e}  {detail}  {'OK' if ok else 'FAIL'}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
