"""The atlas package keeps the names and behaviour of the former single-file ``atlas`` module."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PUBLIC_ATLAS_API = [
    "DEFAULT_LOCAL_PATHS",
    "WAVE_HOVMOLLER_WINDOWS",
    "compute_case10_regression_hovmoller",
    "compute_longitude_mean_monthly_rms_climatology_and_significance",
    "compute_monthly_climatology_and_significance",
    "compute_monthly_rms",
    "compute_monthly_rms_samples",
    "compute_monthly_variance_fraction_samples",
    "compute_wave_eof",
    "compute_yearly_rms",
    "detect_point_events",
    "detect_wave_events",
    "equatorial_mean",
    "fill_missing_with_time_mean",
    "generate_local_wave_atlas",
    "is_antisymmetric_wave",
    "lagged_composite",
    "load_local_wave_fields",
    "regress_field_onto_pcs",
    "select_hovmoller_window",
    "summarize_variance_fraction_cycle",
    "wave_longitude_projection",
]


@pytest.mark.parametrize("name", PUBLIC_ATLAS_API)
def test_public_atlas_names_are_importable_from_the_package(name: str) -> None:
    package = importlib.import_module("tropical_wave_tools.atlas")
    assert hasattr(package, name)
    assert name in package.__all__


def test_case_constants_are_consistent_across_waves() -> None:
    from tropical_wave_tools import atlas

    for wave in atlas.CASE10_WAVES:
        assert wave in atlas.CASE10_BASE_POINTS
        assert wave in atlas.CASE10_LAGS
        assert wave in atlas.CASE10_LON_WINDOWS
        assert wave in atlas.CASE10_WINDOW_VARIANCE_DAYS


def test_project_root_is_the_repository_root() -> None:
    """Moving the code into a subpackage changed the depth below the repository root."""
    from tropical_wave_tools import atlas

    root = Path(atlas._PROJECT_ROOT)
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "tropical_wave_tools").is_dir()
    assert atlas._resolve_local_project_path("data/local/README.md") == (root / "data/local/README.md").resolve()
    assert atlas._resolve_local_project_path("/an/absolute/path") == Path("/an/absolute/path")
    for path in atlas.DEFAULT_LOCAL_PATHS.values():
        assert isinstance(path, (str, Path))


def test_top_level_lazy_exports_resolve() -> None:
    import tropical_wave_tools as twt

    for name in ("compute_monthly_rms", "compute_wave_eof", "generate_local_wave_atlas", "load_local_wave_fields"):
        assert callable(getattr(twt, name))
