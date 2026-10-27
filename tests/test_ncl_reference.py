"""Compare the Python filter against committed NCL ``kf_filter`` golden files."""

from __future__ import annotations

import pytest
import xarray as xr

from validation.compare_with_ncl import (
    HERE,
    INPUTS,
    MAX_NRMSE,
    MIN_CORR,
    compare,
    reference_files,
    split_name,
)

REFERENCES = reference_files()


@pytest.mark.skipif(not REFERENCES, reason="no NCL golden files in validation/reference")
@pytest.mark.parametrize("path", REFERENCES, ids=lambda p: p.stem)
def test_filter_matches_ncl(path) -> None:
    source, wave = split_name(path)
    x = xr.open_dataarray(HERE / INPUTS[source])
    stats = compare(wave, x, xr.open_dataset(path)["filtered"])
    assert stats["corr"] >= MIN_CORR, stats
    assert stats["nrmse"] <= MAX_NRMSE, stats
