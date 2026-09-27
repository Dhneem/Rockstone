"""Tests for slice alignment."""

from __future__ import annotations

import numpy as np

from rockstone.align import (
    align_projections,
    align_slices,
    estimate_rotation,
    pairwise_transform,
    subpixel_shift,
)
from rockstone.demo import make_projection_stack, make_rock_slice
from rockstone.evaluation import ground_truth_corrections
from rockstone.transforms import RigidTransform, apply_transform


def _texture(size=160, seed=7):
    return make_rock_slice(size, seed=seed)


def test_subpixel_shift_translation():
    base = _texture(128)
    shifted = apply_transform(base, RigidTransform(dx=7.4, dy=-3.2))
    T, q = pairwise_transform(base, shifted, estimate_theta=False)
    assert abs(T.dx + 7.4) < 0.5
    assert abs(T.dy - 3.2) < 0.5
    assert q > 0.3


def test_translation_only_pairwise_recovers_inverse():
    base = _texture(128, seed=3)
    moved = apply_transform(base, RigidTransform(dx=-6.0, dy=4.5))
    T, _ = pairwise_transform(base, moved, estimate_theta=False)
    assert abs(T.dx - 6.0) < 0.5
    assert abs(T.dy + 4.5) < 0.5


def test_rotation_estimate_and_correction():
    base = _texture(192, seed=7)
    rotated = apply_transform(base, RigidTransform(theta=np.radians(6.0)))
    T, q = pairwise_transform(base, rotated, estimate_theta=True)
    assert abs(np.degrees(T.theta) + 6.0) < 0.5
    rec = apply_transform(rotated, T)
    corr = np.corrcoef(base.ravel(), rec.ravel())[0, 1]
    assert corr > 0.98


def test_rotation_estimate_sign_both_directions():
    base = _texture(192, seed=11)
    for angle in (-13.0, 9.0):
        rot = apply_transform(base, RigidTransform(theta=np.radians(angle)))
        ang, _ = estimate_rotation(base, rot)
        # allow the 180-degree mirror lobe: fold the difference into [-90, 90)
        delta = (np.degrees(ang) - angle + 90.0) % 180.0 - 90.0
        assert delta < 0.5, f"angle {angle}: got {np.degrees(ang):.3f}"


def test_combined_rigid_recovery():
    base = _texture(192, seed=5)
    T2 = RigidTransform(theta=np.radians(-4.5), dx=3.1, dy=-2.2)
    moved = apply_transform(base, T2)
    T, q = pairwise_transform(base, moved, estimate_theta=True)
    rec = apply_transform(moved, T)
    corr = np.corrcoef(base.ravel(), rec.ravel())[0, 1]
    assert corr > 0.98
    assert q > 0.2


def test_align_slices_removes_drift():
    nz, size = 6, 128
    base = _texture(size, seed=7)
    true = [
        RigidTransform(dx=3.0 * i, dy=-1.5 * i, theta=np.radians(1.5 * i))
        for i in range(nz)
    ]
    vol = np.stack([
        apply_transform(base, t, cval=0.0) for t in true
    ])
    aligned, res = align_slices(vol, estimate_theta=True)
    # After alignment, consecutive slices should barely differ in shift.
    for i in range(1, nz):
        t, q = pairwise_transform(aligned[i - 1], aligned[i],
                                  estimate_theta=False)
        assert abs(t.dx) < 1.0 and abs(t.dy) < 1.0
    # Ground-truth comparison via evaluation helper.
    expected = ground_truth_corrections(true, reference=0)
    for est, exp in zip(res.transforms, expected):
        assert abs(est.dx - exp.dx) < 1.5
        assert abs(est.dy - exp.dy) < 1.5
        assert abs(np.degrees(est.theta - exp.theta)) < 1.5


def test_align_slices_smoothing_poly_runs():
    nz, size = 8, 96
    rng = np.random.default_rng(0)
    base = _texture(size, seed=2)
    vol = np.stack([
        apply_transform(base, RigidTransform(
            dx=2.0 * i + rng.normal(0, 0.5),
            dy=-1.0 * i,
            theta=np.radians(0.3 * i)),
            cval=0.0)
        for i in range(nz)
    ])
    aligned, res = align_slices(vol, estimate_theta=False, smoothing="poly",
                                smoothing_degree=2)
    assert res.smoothed
    assert aligned.shape == vol.shape


def test_align_projections_recovers_jitter():
    stack, true_shifts = make_projection_stack(
        n_projections=40, det_rows=32, det_cols=80, seed=3, jitter=2.0
    )
    aligned, shifts = align_projections(stack, layout="detector",
                                        smoothing="none")
    # The generator applies integer (rounded) rolls relative to view 0, and
    # the correction keeps view n//2 fixed, so compare against the
    # reference-relative ground truth.
    r = np.round(true_shifts)
    expected = -(r - r[len(stack) // 2])
    assert np.max(np.abs(shifts - expected)) < 1.0
    # residual motion between consecutive aligned projections
    from rockstone.align import _subpixel_shift_1d

    for p in range(1, stack.shape[0]):
        d, _ = _subpixel_shift_1d(aligned[p - 1].mean(axis=0),
                                  aligned[p].mean(axis=0))
        assert abs(d) < 0.5


def test_align_projections_sinogram_layout():
    stack, true_shifts = make_projection_stack(
        n_projections=30, det_rows=24, det_cols=64, seed=5, jitter=1.5
    )
    sino = np.transpose(stack, (1, 2, 0))
    aligned, shifts = align_projections(sino, layout="sinogram",
                                        smoothing="none")
    assert aligned.shape == sino.shape
    r = np.round(true_shifts)
    expected = -(r - r[len(stack) // 2])
    assert np.max(np.abs(shifts - expected)) < 1.2
