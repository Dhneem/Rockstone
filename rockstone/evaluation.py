"""Helpers for comparing estimated transforms against ground truth."""

from __future__ import annotations

import numpy as np

from .transforms import RigidTransform

__all__ = ["rigid_deviation", "ground_truth_corrections"]


def ground_truth_corrections(
    true_transforms: list[RigidTransform], reference: int = 0
) -> list[RigidTransform]:
    """Correction moves that align data from :func:`rockstone.demo.make_rock_stack`.

    Slice ``i`` of that synthetic data is ``warp(clean, T_i)``; aligning
    every slice into the frame of ``reference`` requires the content move
    ``M_i = T_i^-1 . T_ref`` (with ``M_ref = identity``). Returns the list
    of ``M_i`` to compare against ``AlignmentResult.transforms``.
    """
    ref = true_transforms[reference]
    return [ref.composed_with(t.inverse()) for t in true_transforms]


def rigid_deviation(
    estimated: list[RigidTransform],
    truth: list[RigidTransform],
    angle_in_degrees: bool = True,
) -> dict[str, float]:
    """Mean/max absolute deviation between two lists of content moves.

    ``estimated[i]`` should map corrupted slice ``i`` back to the common
    frame and ``truth[i]`` is the ground-truth correction (build it with
    :func:`ground_truth_corrections` for data from
    :func:`rockstone.demo.make_rock_stack`).
    """
    if len(estimated) != len(truth):
        raise ValueError("estimated and truth must have the same length")
    dtheta = np.array([e.theta - t.theta for e, t in zip(estimated, truth)])
    ddx = np.array([e.dx - t.dx for e, t in zip(estimated, truth)])
    ddy = np.array([e.dy - t.dy for e, t in zip(estimated, truth)])
    if angle_in_degrees:
        dtheta = np.degrees(dtheta)
    return {
        "dtheta_mean": float(np.mean(np.abs(dtheta))),
        "dtheta_max": float(np.max(np.abs(dtheta))),
        "dx_mean": float(np.mean(np.abs(ddx))),
        "dx_max": float(np.max(np.abs(ddx))),
        "dy_mean": float(np.mean(np.abs(ddy))),
        "dy_max": float(np.max(np.abs(ddy))),
    }
