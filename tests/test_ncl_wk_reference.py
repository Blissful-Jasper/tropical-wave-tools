"""Compare ``analyze_wk_spectrum`` with committed NCL ``wkSpaceTime`` golden files."""

from __future__ import annotations

import pytest

from validation.compare_wk_with_ncl import CASES, REF_DIR, compare


@pytest.mark.parametrize("case", sorted(CASES))
def test_wk_spectrum_matches_ncl(case: str) -> None:
    if not (REF_DIR / f"{case}.nc").exists():
        pytest.skip("no NCL wkSpaceTime golden file in validation/wk_ref")
    _, tolerance = CASES[case]
    stats = compare(case)
    assert max(stats.values()) <= tolerance, stats
