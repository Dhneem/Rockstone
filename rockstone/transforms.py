"""Rigid 2D transform model shared by the alignment code and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidTransform:
    """Rigid 2D transform: rotation about ``pivot`` plus translation.

    A pixel at ``p`` in the source slice maps to ``R @ (p - pivot) + pivot + t``
    in the target slice, where ``R`` is the 2x2 rotation matrix for angle
    ``theta`` (radians). With image arrays the y-axis points down, so a
    positive ``theta`` rotates content *clockwise* as displayed.
    """

    theta: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    pivot: tuple[float, float] = (0.0, 0.0)

    @property
    def rotation_matrix(self) -> np.ndarray:
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def as_matrix(self) -> np.ndarray:
        """3x3 homogeneous matrix M with ``p' = M @ [x, y, 1]``."""
        px, py = self.pivot
        R = self.rotation_matrix
        t = np.array([self.dx, self.dy])
        M = np.eye(3)
        M[:2, :2] = R
        M[:2, 2] = R @ np.array([-px, -py]) + np.array([px, py]) + t
        return M

    def composed_with(self, other: "RigidTransform") -> "RigidTransform":
        """Return the transform equivalent to applying ``self`` first,
        then ``other`` (i.e. ``other @ self`` in matrix order)."""
        M = other.as_matrix() @ self.as_matrix()
        return from_matrix(M)

    def inverse(self) -> "RigidTransform":
        return from_matrix(np.linalg.inv(self.as_matrix()))

    def scaled_translation(self, scale: float) -> "RigidTransform":
        """Same rotation/pivot, translation multiplied by ``scale``."""
        return RigidTransform(self.theta, self.dx * scale, self.dy * scale, self.pivot)


def from_matrix(M: np.ndarray) -> RigidTransform:
    """Build a :class:`RigidTransform` from a 3x3 (or 2x3) affine matrix.

    Raises :class:`ValueError` if the linear part is not a pure rotation.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (2, 3):
        full = np.vstack([M, [0.0, 0.0, 1.0]])
    elif M.shape == (3, 3):
        full = M
    else:
        raise ValueError(f"expected a 2x3 or 3x3 matrix, got shape {M.shape}")
    R = full[:2, :2]
    if not np.allclose(R @ R.T, np.eye(2), atol=1e-6):
        raise ValueError("matrix linear part is not orthogonal: not a rigid transform")
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    # Recover translation so that pivot stays at the origin of `full`.
    return RigidTransform(theta=theta, dx=float(full[0, 2]), dy=float(full[1, 2]))


def apply_transform(
    image: np.ndarray,
    transform: RigidTransform,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Warp ``image`` so that ``out[y, x] = image(transform^-1(p))``.

    ``transform`` maps source coordinates to target coordinates; the warp
    therefore applies the inverse mapping under the hood (an inverse warp,
    which keeps output pixels defined everywhere). Pixels whose source
    coordinate falls outside the image are set to ``cval``.
    """
    if output_shape is None:
        output_shape = image.shape[:2]
    h, w = output_shape
    Minv = transform.inverse().as_matrix()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ones = np.ones_like(xx)
    coords = np.stack([xx, yy, ones], axis=0).reshape(3, -1)
    src = Minv @ coords

    # Map back to the source frame's storage convention: the pivot and
    # translations are already expressed in (x, y) pixel coordinates.
    warped = _map_coordinates(image, src[1], src[0], order=order, cval=cval)
    # Honor cval for source coordinates outside the image (the sampler
    # itself clamps to the border).
    r, c = src[1], src[0]
    valid = (r >= 0.0) & (r <= image.shape[0] - 1) & \
            (c >= 0.0) & (c <= image.shape[1] - 1)
    if not valid.all():
        if image.ndim == 3:
            warped = warped.reshape(output_shape + (image.shape[2],))
            warped[~valid.reshape(output_shape)] = cval
            return warped
        warped = warped.reshape(output_shape)
        warped[~valid.reshape(output_shape)] = cval
        return warped
    if image.ndim == 3:
        return warped.reshape(output_shape + (image.shape[2],))
    return warped.reshape(output_shape)


def _map_coordinates(
    image: np.ndarray, row_coords: np.ndarray, col_coords: np.ndarray,
    order: int, cval: float,
) -> np.ndarray:
    """Minimal bilinear/nearest sampler to avoid importing scipy here."""
    h, w = image.shape[:2]
    flat = image.reshape(-1) if image.ndim == 2 else image.reshape(h, w, -1)
    multichannel = image.ndim == 3
    if multichannel:
        flat = flat.reshape(h * w, image.shape[2])

    r = row_coords.ravel()
    c = col_coords.ravel()
    if order == 0:
        ri = np.clip(np.rint(r).astype(np.int64), 0, h - 1)
        ci = np.clip(np.rint(c).astype(np.int64), 0, w - 1)
        idx = ri * w + ci
        if multichannel:
            return flat[idx, :]
        return flat[idx]

    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    dr = r - r0
    dc = c - c0
    r0c = np.clip(r0, 0, h - 1)
    c0c = np.clip(c0, 0, w - 1)
    r1c = np.clip(r0 + 1, 0, h - 1)
    c1c = np.clip(c0 + 1, 0, w - 1)

    idx00 = r0c * w + c0c
    idx01 = r0c * w + c1c
    idx10 = r1c * w + c0c
    idx11 = r1c * w + c1c
    if multichannel:
        v00, v01, v10, v11 = flat[idx00], flat[idx01], flat[idx10], flat[idx11]
        out = (
            v00 * ((1 - dr) * (1 - dc))[:, None]
            + v01 * ((1 - dr) * dc)[:, None]
            + v10 * (dr * (1 - dc))[:, None]
            + v11 * (dr * dc)[:, None]
        )
        return out
    out = (
        flat[idx00] * (1 - dr) * (1 - dc)
        + flat[idx01] * (1 - dr) * dc
        + flat[idx10] * dr * (1 - dc)
        + flat[idx11] * dr * dc
    )
    return out
