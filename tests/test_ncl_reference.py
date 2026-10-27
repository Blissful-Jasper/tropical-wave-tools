"""Run the NCL cross-check when reference files have been generated."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "validation" / "reference"
REFERENCES = sorted(REFERENCE_DIR.glob("ncl_*.nc"))


@pytest.mark.skipif(not REFERENCES, reason="no NCL reference output in validation/reference")
@pytest.mark.parametrize("path", REFERENCES, ids=lambda p: p.stem)
def test_filter_matches_ncl(path: Path) -> None:
    from validation.compare_with_ncl import MAX_NRMSE, MIN_CORR, compare

    x = xr.open_dataarray(REFERENCE_DIR.parent / "input_equator.nc")
    stats = compare(path.stem.removeprefix("ncl_"), x, xr.open_dataset(path)["filtered"])
    assert stats["corr"] >= MIN_CORR
    assert stats["nrmse"] <= MAX_NRMSE
