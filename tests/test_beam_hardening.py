"""Tests for beam-hardening removal and quality metrics."""

from __future__ import annotations

import numpy as np

from rockstone.beam_hardening import (
    _estimate_chords,
    correct_projections_chord,
    polynomial_remapped,
    remove_beam_hardening,
)
from rockstone.demo import make_phantom_projections, make_rock_stack
from rockstone.metrics import cupping_index, radial_profile
from rockstone.recon import reconstruct_fbp


def _interior_spread(sino):
    """Cupping spread of an FBP reconstruction from a phantom sinogram."""
    rec = reconstruct_fbp(sino[:, sino.shape[1] // 2, :], layout="angle",
                          window="none")
    c = (rec.shape[0] - 1) / 2
    yy, xx = np.mgrid[0:rec.shape[0], 0:rec.shape[1]]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    m = r < 0.8 * c
    rad, vals = r[m], rec[m]
    nb = 6
    edges = np.linspace(0, rad.max(), nb + 1)
    prof = [np.median(vals[(rad >= edges[i]) & (rad < edges[i + 1])])
            for i in range(nb)]
    return (max(prof) - min(prof)) / max(prof)


class TestChordEstimation:
    def test_range_and_shape(self):
        stack, _ = make_phantom_projections(n_projections=8, det_rows=16,
                                            det_cols=64, seed=0)
        ch = _estimate_chords(stack[0, 8])
        assert ch.shape == (64,)
        assert ch.max() <= 1.0 + 1e-9
        assert ch.max() > 0.9  # central ray passes through the full chord

    def test_empty_row(self):
        ch = _estimate_chords(np.zeros(32))
        assert ch.max() == 0.0


class TestChordCorrection:
    def test_flattens_phantom_reconstruction(self):
        stack, _ = make_phantom_projections(n_projections=90, det_rows=16,
                                            det_cols=96, hardening=0.4,
                                            seed=1)
        raw_spread = _interior_spread(stack)
        corr, info = correct_projections_chord(stack, degree=2)
        corr_spread = _interior_spread(corr)
        assert corr_spread < raw_spread / 2.0
        assert corr_spread < 0.10
        assert "model" in info and "gain_curve" in info

    def test_sinogram_layout_matches_detector(self):
        stack, _ = make_phantom_projections(n_projections=24, det_rows=12,
                                            det_cols=64, hardening=0.3,
                                            seed=2)
        sino = np.transpose(stack, (1, 2, 0))
        c1, _ = correct_projections_chord(stack, layout="detector")
        c2, _ = correct_projections_chord(sino, layout="sinogram")
        # results come back in the same layout as the input
        assert c2.shape == sino.shape
        np.testing.assert_allclose(c1, np.transpose(c2, (2, 0, 1)), atol=1e-8)


class TestRadialCorrection:
    def test_reduces_cupping_on_rock_slices(self):
        vol, _ = make_rock_stack(n_slices=5, size=128, seed=0, cupping=0.35)
        before = cupping_index(vol[2])
        corrected, res = remove_beam_hardening(vol, method="radial", degree=3)
        after = cupping_index(corrected[2])
        assert after < before / 2.0
        assert res.method == "radial"
        assert np.isfinite(res.cupping_before)
        assert np.isfinite(res.cupping_after)

    def test_single_slice_2d(self):
        vol, _ = make_rock_stack(n_slices=3, size=128, seed=1, cupping=0.3)
        corrected, res = remove_beam_hardening(vol[1], method="radial")
        assert corrected.shape == vol[1].shape


class TestPolyRemap:
    def test_identity_map(self):
        data = np.linspace(0, 1, 100).reshape(10, 10)
        # polyfit order (highest first): p(I) = 1*I + 0 is the identity
        out = polynomial_remapped(data, np.array([1.0, 0.0]))
        np.testing.assert_allclose(out, data, atol=1e-6)

    def test_monotonic_remap_preserves_order(self):
        data = np.linspace(0, 1, 100).reshape(10, 10)
        coeffs = np.array([0.5, -0.2, 1.0])
        out = polynomial_remapped(data, coeffs)
        order = np.argsort(data.ravel())
        sorted_out = out.ravel()[order]
        assert np.all(np.diff(sorted_out) >= -1e-9)


class TestRemoveBeamHardeningDispatch:
    def test_unknown_method(self):
        import pytest

        with pytest.raises(ValueError):
            remove_beam_hardening(np.random.default_rng(0).random((4, 16, 16)),
                                  method="nope")

    def test_poly_requires_coefficients(self):
        import pytest

        with pytest.raises(ValueError):
            remove_beam_hardening(np.random.default_rng(0).random((4, 16, 16)),
                                  method="poly")


class TestMetrics:
    def test_cupping_index_uniform(self):
        img = np.full((100, 100), 0.5)
        ci = cupping_index(img)
        assert np.isnan(ci) or abs(ci) < 0.05  # flat profile: no cupping

    def test_radial_profile_bins(self):
        img = np.full((100, 100), 1.0)
        img[:10] = 0.0
        r, p = radial_profile(img, center=(50, 50), phase="all", n_bins=8)
        assert r.shape == p.shape == (8,)
        assert np.isfinite(p).sum() >= 6
