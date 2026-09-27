"""Quality metrics for rock CT processing.

- :func:`radial_profile` — mean value vs radius from the sample center.
- :func:`cupping_index` — scalar severity of beam-hardening cupping.
- :func:`alignment_residual` — residual inter-slice motion after alignment.
"""

from __future__ import annotations

import numpy as np

__all__ = ["radial_profile", "cupping_index", "alignment_residual"]


def radial_profile(
    image: np.ndarray,
    center: tuple[float, float],
    phase: str = "bright",
    n_bins: int = 32,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity vs radius from ``center``.

    ``phase`` selects which population contributes to each bin:
    ``"bright"`` keeps values above the image median (the rock inside a
    darker air background), ``"dark"`` keeps values below it, and
    ``"all"`` keeps everything. Returns ``(radii, means)`` where invalid
    (empty) bins are ``nan``.
    """
    data = np.asarray(image, dtype=np.float64)
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)

    if r_max is None:
        r_max = min(center[0], center[1], w - 1 - center[0], h - 1 - center[1])
    med = np.nanmedian(data)
    if phase == "bright":
        mask = data > med
    elif phase == "dark":
        mask = data < med
    else:
        mask = np.ones_like(data, dtype=bool)

    edges = np.linspace(0.0, float(r_max), n_bins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    valid_bin = (idx >= 0) & (idx < n_bins)
    vals = data.ravel()
    msk = mask.ravel() & valid_bin & np.isfinite(vals)

    radii = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    sums = np.bincount(idx[msk], weights=vals[msk], minlength=n_bins)
    counts = np.bincount(idx[msk], minlength=n_bins)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    return radii, means


def cupping_index(
    image: np.ndarray,
    center: tuple[float, float] | None = None,
    background: str = "dark",
    phase: str = "bright",
) -> float:
    """Scalar cupping severity: ``1 - profile(R_inner) / profile(R_outer)``.

    A uniform, hardening-free sample gives ~0; strong cupping gives
    values approaching 1. Negative values indicate anti-cupping (edge
    darkening, e.g. from rings or scattering corrections).
    """
    data = np.asarray(image, dtype=np.float64)
    if center is None:
        from .beam_hardening import _center_from_threshold

        center = _center_from_threshold(data, background)
    radii, prof = radial_profile(data, center=center, phase=phase, n_bins=32)
    finite = np.isfinite(prof)
    if finite.sum() < 4:
        return float("nan")
    # Use robust outer/inner references from the valid bins.
    r_valid = radii[finite]
    p_valid = prof[finite]
    order = np.argsort(r_valid)
    r_valid, p_valid = r_valid[order], p_valid[order]
    inner = float(np.median(p_valid[: max(1, len(p_valid) // 4)]))
    outer = float(np.median(p_valid[-max(1, len(p_valid) // 4):]))
    if abs(outer) < 1e-12:
        return float("nan")
    return 1.0 - inner / outer


def alignment_residual(transforms: list) -> float:
    """Mean absolute residual motion (px/deg) of a corrected trajectory.

    Takes the per-slice transforms of an :class:`AlignmentResult` and
    reports the mean absolute deviation of the (smoothed) trajectory from
    its own linear trend — near zero when slices are well aligned.
    """
    thetas = np.array([t.theta for t in transforms])
    dxs = np.array([t.dx for t in transforms])
    dys = np.array([t.dy for t in transforms])
    n = len(transforms)
    x = np.linspace(0.0, 1.0, n)

    def resid_trend(v: np.ndarray) -> float:
        if n < 3:
            return 0.0
        c = np.polyfit(x, v, 1)
        return float(np.mean(np.abs(v - np.polyval(c, x))))

    return float(np.mean([resid_trend(dxs), resid_trend(dys),
                          np.degrees(resid_trend(thetas))]))
