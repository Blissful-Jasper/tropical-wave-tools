from __future__ import annotations

import xarray as xr
import numpy as np

from tropical_wave_tools.atlas import (
    _case07_component_field,
    _group_comparison_waves,
    compute_longitude_mean_monthly_rms_climatology_and_significance,
    regress_field_onto_pcs,
    wave_longitude_projection,
)


def test_group_comparison_waves_splits_large_scale_and_synoptic_sets() -> None:
    grouped = _group_comparison_waves(["kelvin", "er", "mrg", "eig", "wig", "td", "mjo"])
    assert grouped == [
        ("large_scale", ["kelvin", "er", "mjo"]),
        ("westward", ["mrg", "td"]),
        ("other", ["eig", "wig"]),
    ]


def test_group_comparison_waves_keeps_unclassified_waves_last() -> None:
    grouped = _group_comparison_waves(["kelvin", "foo", "td"])
    assert grouped == [
        ("large_scale", ["kelvin"]),
        ("westward", ["td"]),
        ("other", ["foo"]),
    ]


def test_wave_longitude_projection_uses_antisymmetric_difference_for_mrg(synthetic_wave_data: xr.DataArray) -> None:
    output = wave_longitude_projection(synthetic_wave_data, wave_name="mrg")
    north = synthetic_wave_data.where(synthetic_wave_data["lat"] > 0.0, drop=True).mean("lat")
    south = synthetic_wave_data.where(synthetic_wave_data["lat"] < 0.0, drop=True).mean("lat")
    expected = north - south
    xr.testing.assert_allclose(output, expected)
    assert output.attrs["wave_projection"] == "antisymmetric_difference"


def test_wave_longitude_projection_uses_equatorial_mean_for_kelvin(synthetic_wave_data: xr.DataArray) -> None:
    output = wave_longitude_projection(synthetic_wave_data, wave_name="kelvin")
    expected = synthetic_wave_data.sel(lat=slice(-5.0, 5.0)).mean("lat")
    xr.testing.assert_allclose(output, expected)
    assert output.attrs["wave_projection"] == "equatorial_mean"


def test_case07_component_field_preserves_expected_equatorial_symmetry(synthetic_wave_data: xr.DataArray) -> None:
    lat_values = synthetic_wave_data["lat"].values.astype(float)
    symmetric_values = np.broadcast_to(np.abs(lat_values)[None, :, None], synthetic_wave_data.shape)
    antisymmetric_values = np.broadcast_to(lat_values[None, :, None], synthetic_wave_data.shape)
    symmetric = xr.DataArray(
        symmetric_values,
        dims=synthetic_wave_data.dims,
        coords=synthetic_wave_data.coords,
        name="symmetric",
    )
    antisymmetric = xr.DataArray(
        antisymmetric_values,
        dims=synthetic_wave_data.dims,
        coords=synthetic_wave_data.coords,
        name="antisymmetric",
    )

    xr.testing.assert_allclose(_case07_component_field(symmetric, projection="symmetric"), symmetric)
    xr.testing.assert_allclose(_case07_component_field(antisymmetric, projection="antisymmetric"), antisymmetric)


def test_regress_field_onto_pcs_can_standardize_principal_components() -> None:
    field = xr.DataArray(
        np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[2.0, 2.0], [2.0, 2.0]],
                [[4.0, 4.0], [4.0, 4.0]],
            ],
            dtype=float,
        ),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1, 2], "lat": [-5.0, 5.0], "lon": [0.0, 30.0]},
        name="olr",
    )
    pcs = xr.DataArray(
        np.array([[0.0, 2.0, 4.0]], dtype=float),
        dims=("mode", "time"),
        coords={"mode": [1], "time": [0, 1, 2]},
        name="pcs",
    )

    raw = regress_field_onto_pcs(field, pcs, standardize_pc=False)
    standardized = regress_field_onto_pcs(field, pcs, standardize_pc=True)
    expected_scale = float((pcs.sel(mode=1) - pcs.sel(mode=1).mean("time")).std("time").item())
    xr.testing.assert_allclose(standardized, raw * expected_scale)


def test_longitude_mean_monthly_rms_climatology_preserves_amplitude_when_zonal_mean_cancels(
    synthetic_wave_data: xr.DataArray,
) -> None:
    anomaly_like = synthetic_wave_data - synthetic_wave_data.mean("lon")
    climatology, pvalues = compute_longitude_mean_monthly_rms_climatology_and_significance(anomaly_like)
    assert "month" in climatology.dims
    assert "month" in pvalues.dims
    assert float(climatology.max()) > 0.0
