"""Tests for the rigid transform model."""

from __future__ import annotations

import numpy as np
import pytest

from rockstone.transforms import RigidTransform, apply_transform, from_matrix


def _smooth_image(size=48, seed=0):
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, size)
    x = np.linspace(0, 1, size)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    img = np.sin(3 * yy) * np.cos(2 * xx) + 0.5
    return img + 0.01 * rng.random((size, size))


def test_identity_is_identity():
    img = _smooth_image()
    out = apply_transform(img, RigidTransform())
    np.testing.assert_allclose(out, img)


def test_pure_translation_moves_content():
    img = np.zeros((64, 64))
    img[20:40, 30:50] = 1.0
    moved = apply_transform(img, RigidTransform(dx=5.0, dy=3.0))
    assert moved[23:43, 35:55].mean() > 0.99


def test_rotation_sign_image_coordinates():
    # With y pointing down, positive theta rotates content clockwise as
    # displayed: a point directly BELOW the center moves toward -x (left),
    # ending up beside the center (same row as the pivot).
    img = np.zeros((101, 101))
    c = 50.0
    img[100, 50] = 1.0  # directly below the image center
    out = apply_transform(img, RigidTransform(theta=np.pi / 2, pivot=(c, c)))
    yy, xx = np.nonzero(out > 0.5)
    assert xx[0] < 40   # moved to the left
    assert yy[0] < 60   # now beside the center (row ~ pivot row)


def test_roundtrip_composition():
    c = 24.0
    T1 = RigidTransform(theta=0.3, dx=4.0, dy=-2.0, pivot=(c, c))
    T2 = RigidTransform(theta=-0.1, dx=-1.0, dy=7.0, pivot=(c, c))
    # Noiseless image: the chained path interpolates twice, the composed
    # path once, so compare on a smooth signal at interpolation tolerance.
    y = np.linspace(0, 1, 48)
    yy, xx = np.meshgrid(y, y, indexing="ij")
    img = np.sin(3 * yy) * np.cos(2 * xx) + 0.5
    # Compare only where both warps sample inside the frame: a disk around
    # the pivot small enough that the chained path never hits the border.
    Y, X = np.mgrid[0:48, 0:48]
    interior = (Y - c) ** 2 + (X - c) ** 2 < 12.0 ** 2
    chained = apply_transform(apply_transform(img, T1), T2)
    composed = apply_transform(img, T1.composed_with(T2))
    np.testing.assert_allclose(chained[interior], composed[interior], atol=2e-3)
    roundtrip = apply_transform(apply_transform(img, T1), T1.inverse())
    np.testing.assert_allclose(roundtrip[interior], img[interior], atol=2e-3)


def test_from_matrix_rejects_non_rigid():
    M = np.array([[2.0, 0, 1], [0, 2.0, 1], [0, 0, 1]])
    with pytest.raises(ValueError):
        from_matrix(M)


def test_apply_transform_multichannel():
    img = np.zeros((32, 32, 3))
    img[10:20, 10:20] = 1.0
    out = apply_transform(img, RigidTransform(dx=4.0, dy=2.0))
    assert out.shape == (32, 32, 3)
    # Block [10:20, 10:20] shifted by dy=2, dx=4 -> rows 12:22, cols 14:24.
    assert out[12:22, 14:24].mean() > 0.99
