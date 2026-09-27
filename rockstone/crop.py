"""Cropping helpers: rectangle and ellipse.

Both shapes take the same dragged selection — an axis-aligned
``(x0, y0, x1, y1)`` rectangle with inclusive corners. The rectangle
crop slices the array; the ellipse crop keeps the pixels inside the
ellipse *inscribed* in that rectangle and fills the rest (NaN, zero or
the air level), preserving the full image size — the classic
"cut the surrounding material away" operation for rock samples.

Masks are precomputed with an exact integer test
(``(dx/a)^2 + (dy/b)^2 <= 1``), so no pixel is guessed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "ELLIPSE_FILLS",
    "build_ellipse_mask",
    "ellipse_outline_mask",
    "crop_rectangle",
    "crop_ellipse",
]

#: what to write into the region outside the ellipse
ELLIPSE_FILLS = ["nan", "zero", "air"]


def build_ellipse_mask(shape: tuple[int, int],
                       rect: tuple[int, int, int, int]) -> np.ndarray:
    """Boolean mask, True inside the ellipse inscribed in ``rect``.

    ``shape`` is the full (h, w) of the plane; ``rect`` the selection
    in the same inclusive pixel coordinates as the rectangle crop.
    """
    h, w = shape
    x0, y0, x1, y1 = rect
    if x1 < x0 or y1 < y0:
        raise ValueError(f"degenerate rectangle: {rect!r}")
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        raise ValueError(f"rectangle {rect!r} outside image {w}x{h}")
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    a = (x1 - x0) / 2.0  # semi-axis along x
    b = (y1 - y0) / 2.0  # semi-axis along y
    if a <= 0 or b <= 0:
        raise ValueError("rectangle too small for an ellipse")
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0


def ellipse_outline_mask(shape: tuple[int, int],
                         rect: tuple[int, int, int, int],
                         thickness: int = 2) -> np.ndarray:
    """Boolean mask of the inscribed ellipse's outline (for drawing)."""
    mask = build_ellipse_mask(shape, rect)
    inner = ndimage.binary_erosion(mask,
                                   iterations=max(1, int(thickness)))
    return mask & ~inner


def crop_rectangle(
    data: np.ndarray,
    rect: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop the whole stack to ``rect`` (inclusive pixel corners)."""
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D stack, got shape {data.shape}")
    nz, ny, nx = data.shape
    x0, y0, x1, y1 = rect
    if x1 < x0 or y1 < y0:
        raise ValueError(f"degenerate rectangle: {rect!r}")
    if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny:
        raise ValueError(f"rectangle {rect!r} outside image {nx}x{ny}")
    return data[:, y0:y1 + 1, x0:x1 + 1].copy()


def crop_ellipse(
    data: np.ndarray,
    rect: tuple[int, int, int, int],
    fill: str = "nan",
    air_level: float | None = None,
) -> np.ndarray:
    """Keep the ellipse inscribed in ``rect`` on every slice.

    Parameters
    ----------
    data:
        Stack ``(nz, ny, nx)``.
    rect:
        Selection rectangle (inclusive corners) — the ellipse touches
        the four edges of this rectangle.
    fill:
        What to write outside the ellipse: ``"nan"`` (transparent for
        further statistics/exports), ``"zero"`` or ``"air"`` (the air
        level, computed from the data when ``air_level`` is not given).
    air_level:
        Fill value for ``fill="air"``; defaults to the standard CT air
        value of **-1000 HU** (``rockstone.constants.AIR_HU``).

    Returns
    -------
    Array with the same shape and float dtype as the input (float is
    required to hold NaN).
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D stack, got shape {data.shape}")
    if fill not in ELLIPSE_FILLS:
        raise ValueError(f"unknown fill {fill!r} (choose from "
                         f"{ELLIPSE_FILLS})")
    if data.shape[0] == 0:
        raise ValueError("empty stack")
    mask = build_ellipse_mask(data.shape[1:3], rect)

    arr = data.astype(np.float64)
    if fill == "nan":
        outside = np.nan
    elif fill == "zero":
        outside = 0.0
    else:  # air
        if air_level is None:
            from .constants import AIR_HU

            air_level = AIR_HU
        outside = float(air_level)

    out = arr.copy()
    out[:, ~mask] = outside
    return out
