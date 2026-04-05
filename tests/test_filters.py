from __future__ import annotations

from tropical_wave_tools.config import DEFAULT_WAVE_SPECS
from tropical_wave_tools.filters import CCKWFilter, WaveFilter


def test_legacy_wave_filter_returns_expected_metadata(synthetic_wave_data) -> None:
    output = WaveFilter().extract_wave_signal(
        synthetic_wave_data,
        wave_name="kelvin",
        use_parallel=False,
        obs_per_day=1,
    )
    assert output.dims == synthetic_wave_data.dims
    assert output.attrs["waveName"] == "kelvin"


def test_cckw_filter_process_returns_dataarray(synthetic_wave_data) -> None:
    output = CCKWFilter(
        ds=synthetic_wave_data,
        wave_name="kelvin",
        spd=1,
        n_workers=1,
        verbose=False,
    ).process()
    assert output.dims == synthetic_wave_data.dims
    assert output.attrs["waveName"] == "kelvin"


def test_available_waves_include_empirical_and_wig_bands() -> None:
    filter_instance = WaveFilter()
    for wave_name in ("td", "mjo", "eig", "wig"):
        assert wave_name in filter_instance.get_available_waves()
        assert wave_name in DEFAULT_WAVE_SPECS


def test_wig_filter_returns_expected_metadata(synthetic_wave_data) -> None:
    output = WaveFilter().extract_wave_signal(
        synthetic_wave_data,
        wave_name="wig",
        use_parallel=False,
        obs_per_day=1,
    )
    assert output.dims == synthetic_wave_data.dims
    assert output.attrs["waveName"] == "wig"


def test_eig_filter_returns_expected_metadata(synthetic_wave_data) -> None:
    output = WaveFilter().extract_wave_signal(
        synthetic_wave_data,
        wave_name="eig",
        use_parallel=False,
        obs_per_day=1,
    )
    assert output.dims == synthetic_wave_data.dims
    assert output.attrs["waveName"] == "eig"


def test_published_ig_boxes_match_reference() -> None:
    eig = DEFAULT_WAVE_SPECS["eig"]
    wig = DEFAULT_WAVE_SPECS["wig"]
    mrg = DEFAULT_WAVE_SPECS["mrg"]

    assert eig.period_days == (1.0, 6.0)
    assert eig.wavenumber == (0, 15)
    assert eig.equivalent_depth == (12.0, 50.0)

    assert wig.period_days == (1.25, 3.5)
    assert wig.wavenumber == (-15, -1)
    assert wig.equivalent_depth == (12.0, 90.0)

    assert mrg.period_days == (2.5, 10.0)
    assert mrg.equivalent_depth == (8.0, 90.0)

    assert DEFAULT_WAVE_SPECS["ig"] == eig
    assert DEFAULT_WAVE_SPECS["eig0"] == eig
