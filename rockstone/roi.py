"""Region-of-interest (ROI) statistics.

An ROI is an axis-aligned rectangle ``(x0, y0, x1, y1)`` in slice
pixel coordinates (inclusive corners), measured on the *raw data* of
one slice or across every slice of the stack. All statistics ignore
NaN pixels; physical quantities use the volume metadata (pixel size in
mm) when available.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ROIStats",
    "roi_stats",
    "roi_stats_series",
    "write_roi_csv",
]


@dataclass
class ROIStats:
    """Statistics of one ROI on one slice.

    ``air_reference`` (optionally set by the caller) is the estimated
    air level of the slice; when present, the mean is reported both raw
    and air-subtracted (a rough attenuation proxy for CT data).
    """

    slice_index: int
    rect: tuple[int, int, int, int]
    n_pixels: int
    mean: float
    std: float
    minimum: float
    maximum: float
    median: float
    area_mm2: float | None = None
    air_reference: float | None = None

    @property
    def mean_above_air(self) -> float | None:
        if self.air_reference is None:
            return None
        return self.mean - self.air_reference

    def as_dict(self) -> dict:
        d = {
            "slice": self.slice_index + 1,  # 1-based for humans
            "x0": self.rect[0], "y0": self.rect[1],
            "x1": self.rect[2], "y1": self.rect[3],
            "n_pixels": self.n_pixels,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "median": round(self.median, 6),
        }
        if self.area_mm2 is not None:
            d["area_mm2"] = round(self.area_mm2, 4)
        if self.air_reference is not None:
            d["mean_above_air"] = round(self.mean_above_air, 6)
        return d


def _validate_rect(rect: tuple[int, int, int, int],
                   ny: int, nx: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    if x1 < x0 or y1 < y0:
        raise ValueError(f"degenerate ROI rectangle: {rect!r}")
    if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny:
        raise ValueError(
            f"ROI {rect!r} outside the slice (nx={nx}, ny={ny})")
    return rect


def roi_stats(
    data: np.ndarray,
    rect: tuple[int, int, int, int],
    slice_index: int,
    meta: dict | None = None,
    air_reference: float | None = None,
) -> ROIStats:
    """Measure one rectangular ROI on ``data[slice_index]``.

    Parameters
    ----------
    data:
        The full stack ``(nz, ny, nx)`` (the ROI is 2-D but the stack
        provides the physical context).
    rect:
        ``(x0, y0, x1, y1)`` pixel rectangle, inclusive corners.
    slice_index:
        Which slice to measure.
    meta:
        Volume metadata; ``pixel_size`` in mm gives the ROI area.
    air_reference:
        Air level of this slice (e.g. 1st percentile); when given, the
        result carries ``mean_above_air``.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D stack, got shape {data.shape}")
    nz, ny, nx = data.shape
    if not (0 <= slice_index < nz):
        raise IndexError(f"slice index {slice_index} out of range")
    x0, y0, x1, y1 = _validate_rect(rect, ny, nx)
    plane = np.asarray(data[slice_index], dtype=np.float64)
    vals = plane[y0:y1 + 1, x0:x1 + 1]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("ROI contains no finite pixels")

    px = (meta or {}).get("pixel_size")
    area = None
    if isinstance(px, (int, float)):
        area = float(px) ** 2 * vals.size
    elif isinstance(px, (list, tuple)) and len(px) >= 2:
        area = float(px[0]) * float(px[1]) * vals.size

    return ROIStats(
        slice_index=slice_index,
        rect=(x0, y0, x1, y1),
        n_pixels=int(vals.size),
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        minimum=float(np.min(vals)),
        maximum=float(np.max(vals)),
        median=float(np.median(vals)),
        area_mm2=area,
        air_reference=air_reference,
    )


def roi_stats_series(
    data: np.ndarray,
    rect: tuple[int, int, int, int],
    meta: dict | None = None,
    with_air: bool = False,
) -> list[ROIStats]:
    """Measure the same ROI on every slice of the stack.

    With ``with_air=True`` the standard CT air value of **-1000 HU**
    (``rockstone.constants.AIR_HU``) is used as the air reference for
    the mean-above-air statistic.
    """
    nz = np.asarray(data).shape[0]
    air_ref: float | None = None
    if with_air:
        from .constants import AIR_HU

        air_ref = AIR_HU
    return [roi_stats(data, rect, k, meta=meta, air_reference=air_ref)
            for k in range(nz)]


def write_roi_csv(stats: list[ROIStats], path: str | Path) -> Path:
    """Write a per-slice ROI measurement series as CSV."""
    if not stats:
        raise ValueError("no ROI measurements to write")
    fieldnames = list(stats[0].as_dict().keys())
    p = Path(path)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats:
            writer.writerow(s.as_dict())
    return p
