from __future__ import annotations

import numpy as np

from tropical_wave_tools.matsuno import (
    ANGULAR_TO_CPD,
    _cubic_mode_roots,
    beta_parameters,
    eig_n,
    eig_n_0,
    er_n,
    GRAVITY,
    kelvin_mode,
    mrg_mode,
    wig_n,
)


def _series_values(frame):
    return frame.iloc[:, 0]


def test_kelvin_curve_is_eastward_only():
    values = _series_values(kelvin_mode(25.0, max_wn=10, n_wn=21))
    assert np.isnan(values[values.index < 0.0]).all()
    assert np.isfinite(values[values.index > 0.0]).all()


def test_rossby_and_mrg_curves_are_westward_only():
    rossby = _series_values(er_n(25.0, 1, max_wn=10, n_wn=21))
    mrg = _series_values(mrg_mode(25.0, max_wn=10, n_wn=21))
    assert np.isfinite(rossby[rossby.index < 0.0]).all()
    assert np.isnan(rossby[rossby.index > 0.0]).all()
    assert np.isfinite(mrg[mrg.index < 0.0]).all()
    assert np.isnan(mrg[mrg.index > 0.0]).all()


def test_eastward_ig_curves_are_eastward_only():
    eig0 = _series_values(eig_n_0(25.0, max_wn=10, n_wn=21))
    eig1 = _series_values(eig_n(25.0, 1, max_wn=10, n_wn=21))
    assert np.isnan(eig0[eig0.index < 0.0]).all()
    assert np.isfinite(eig0[eig0.index > 0.0]).all()
    assert np.isnan(eig1[eig1.index < 0.0]).all()
    assert np.isfinite(eig1[eig1.index > 0.0]).all()


def test_westward_ig_curve_is_westward_only():
    wig2 = _series_values(wig_n(25.0, 2, max_wn=10, n_wn=21))
    assert np.isfinite(wig2[wig2.index < 0.0]).all()
    assert np.isnan(wig2[wig2.index > 0.0]).all()


def test_westward_ig_uses_fast_positive_root_on_negative_wavenumbers():
    wn, westward_root, _, eastward_root = _cubic_mode_roots(25.0, 2, max_wn=10, n_wn=21)
    wig2 = _series_values(wig_n(25.0, 2, max_wn=10, n_wn=21))
    expected = np.where(wn < 0.0, np.abs(eastward_root) * ANGULAR_TO_CPD, np.nan)
    mask = wn < 0.0
    assert np.allclose(wig2.values[mask], expected[mask], equal_nan=True)
    assert np.all(np.isfinite(eastward_root[mask]))
    assert np.all(eastward_root[mask] > 0.0)


def test_mrg_and_eig0_share_theoretical_frequency_at_zero_wavenumber():
    he = 25.0
    beta, _ = beta_parameters(0.0)
    expected = np.sqrt(beta * np.sqrt(GRAVITY * he)) * ANGULAR_TO_CPD

    mrg = _series_values(mrg_mode(he, max_wn=10, n_wn=21))
    eig0 = _series_values(eig_n_0(he, max_wn=10, n_wn=21))

    assert np.isclose(float(mrg.loc[0.0]), expected, rtol=1.0e-6)
    assert np.isclose(float(eig0.loc[0.0]), expected, rtol=1.0e-6)
