"""Background correction: flatten the air level and smooth shading
around the sample.

Rock CT stacks often show a non-zero air level (detector offset or
reconstruction pedestal — the background around the rock is not 0) and
sometimes a smooth gradient across the field of view (scattering halos,
uneven illumination in scanned radiographs). This module estimates that
background surface per slice and removes it, either subtractively
(background flattens to zero) or multiplicatively (shading flattens to
a constant). It is complementary to the beam-hardening correction,
which models radial cupping *inside* the sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

__all__ = [
    "BackgroundResult",
    "correct_background",
]


@dataclass
class BackgroundResult:
    """Outcome of a background correction.

    Attributes
    ----------
    mode:
        ``"offset"`` (subtract) or ``"divide"``.
    per_slice:
        Whether each slice got its own surface (vs. one shared surface).
    air_levels:
        Estimated background ("air") level per slice.
    air_before / air_after:
        Median background level across slices, before and after.
    background_std_before / background_std_after:
        Median standard deviation of the background pixels across
        slices — the flatness metric the correction minimizes.
    """

    mode: str = "offset"
    per_slice: bool = True
    air_levels: np.ndarray = field(default_factory=lambda: np.array([]))
    air_before: float = float("nan")
    air_after: float = float("nan")
    background_std_before: float = float("nan")
    background_std_after: float = float("nan")


def _percentile_level(plane: np.ndarray, percentile: float) -> float:
    """Robust level: a percentile of the finite pixels."""
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, percentile))


def _sample_mask(plane: np.ndarray, background: str, air: float,
                 spread: float, dilate: int) -> np.ndarray:
    """Boolean mask of *background* pixels (True = background)."""
    finite = np.isfinite(plane)
    if spread <= 0:
        return finite  # no usable contrast: everything counts as bg
    frac = 0.25 * float(np.clip(spread, 0.0, None))
    if background == "dark":
        sample = plane > air + frac
    else:  # bright background (e.g. bright holder): sample is darker
        sample = plane < air - frac
    bg = finite & ~sample
    if dilate > 0 and sample.any():
        # keep a margin around the sample so its rim cannot bias the fit
        sample = ndimage.binary_dilation(sample, iterations=int(dilate))
        bg &= ~sample
    return bg


def _surface_from_plane(plane: np.ndarray, bg: np.ndarray, air: float,
                        block: tuple[int, int]) -> np.ndarray:
    """Fit a smooth background surface to the masked pixels.

    The plane is sampled on a coarse block grid (robust median of the
    background pixels per block), empty nodes are nearest-filled, and
    the grid is linearly upsampled back to full size.
    """
    h, w = plane.shape
    bh, bw = block
    nby = max(1, h // bh)
    nbx = max(1, w // bw)
    nodes = np.full((nby, nbx), np.nan)
    for i in range(nby):
        for j in range(nbx):
            sel = bg[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            if sel.any():
                vals = plane[i * bh:(i + 1) * bh,
                             j * bw:(j + 1) * bw][sel]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    nodes[i, j] = np.median(vals)
    if np.isnan(nodes).all():
        return np.full((h, w), air, dtype=np.float64)
    if np.isnan(nodes).any():  # nearest-fill empty blocks
        ind = ndimage.distance_transform_edt(
            np.isnan(nodes), return_distances=False, return_indices=True)
        nodes = nodes[tuple(ind)]
    surface = ndimage.zoom(nodes, (h / nodes.shape[0], w / nodes.shape[1]),
                           order=1)
    if surface.shape != (h, w):  # zoom rounding: force exact size
        fixed = np.full((h, w), float(np.nanmedian(surface)))
        fixed[:min(surface.shape[0], h), :min(surface.shape[1], w)] = \
            surface[:h, :w]
        surface = fixed
    return surface


def correct_background(
    data: np.ndarray,
    mode: str = "offset",
    per_slice: bool = True,
    air_percentile: float = 1.0,
    background: str = "dark",
    block: tuple[int, int] | None = None,
    dilate: int = 2,
) -> tuple[np.ndarray, BackgroundResult]:
    """Flatten the background of a stack ``(nz, ny, nx)``.

    Parameters
    ----------
    data:
        Stack of slices or projections; any numeric dtype (integer
        input is promoted to float64, NaNs are preserved).
    mode:
        ``"offset"`` subtracts the background surface (air flattens to
        zero — the usual choice for CT stacks); ``"divide"`` divides by
        it (multiplicative shading flattens to a constant, e.g. for
        optical scans where the signal itself is an intensity).
    per_slice:
        Estimate one surface per slice (default; adapts to z-dependent
        offsets) or a single shared surface (median across slices).
    air_percentile:
        Percentile used as the air/reference level (default 1.0). For
        ``background="bright"`` the mirrored high percentile is used.
    background:
        ``"dark"`` (air around a bright rock) or ``"bright"``.
    block:
        Coarse sampling block size ``(bh, bw)``; derived from the image
        size when omitted.
    dilate:
        Pixels of margin added around the sample mask before fitting.

    Returns
    -------
    ``(corrected, BackgroundResult)``
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D stack, got shape {data.shape}")
    if mode not in {"offset", "divide"}:
        raise ValueError(f"unknown mode: {mode!r}")
    nz, ny, nx = data.shape
    block = block or (max(8, ny // 12), max(8, nx // 12))

    air_levels = np.empty(nz)
    surfaces = np.empty((nz, ny, nx))
    masks: list[np.ndarray] = []
    for k in range(nz):
        plane = data[k]
        air = _percentile_level(
            plane, air_percentile if background == "dark"
            else 100.0 - air_percentile)
        air_levels[k] = air
        spread = (_percentile_level(plane, 99.5)
                  - _percentile_level(plane, 0.1))
        bg = _sample_mask(plane, background, air, spread, dilate)
        masks.append(bg)
        surfaces[k] = _surface_from_plane(plane, bg, air, block)
    if not per_slice and nz > 1:
        shared = np.nanmedian(surfaces, axis=0)
        surfaces[:] = shared

    if mode == "offset":
        corrected = data - surfaces
    else:  # divide: flatten multiplicative shading
        safe = np.where(np.abs(surfaces) > 1e-12, surfaces, 1.0)
        corrected = data / safe

    res = BackgroundResult(mode=mode, per_slice=per_slice,
                           air_levels=air_levels)
    stds_before, stds_after, airs_after = [], [], []
    for k in range(nz):
        bg = masks[k]
        if not bg.any():
            continue
        vals_b = data[k][bg]
        vals_a = corrected[k][bg]
        vals_b = vals_b[np.isfinite(vals_b)]
        vals_a = vals_a[np.isfinite(vals_a)]
        if vals_b.size:
            stds_before.append(float(np.std(vals_b)))
            airs_after.append(float(np.median(vals_a)))
        if vals_a.size:
            stds_after.append(float(np.std(vals_a)))
    if stds_before:
        res.background_std_before = float(np.median(stds_before))
        res.background_std_after = float(np.median(stds_after))
        res.air_after = float(np.median(airs_after))
    res.air_before = float(np.median(air_levels)) if nz else float("nan")
    return corrected, res
