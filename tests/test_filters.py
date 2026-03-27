from __future__ import annotations

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

