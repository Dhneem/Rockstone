"""Compare two datasets: difference metrics for QC of the processing.

Typical use: compare the original stack against the aligned/dehardened
result to quantify what the corrections changed, or compare two scans of
the same sample. Handles shape mismatches gracefully by comparing the
overlapping region (center-cropped), and reports which slices differ
most.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ComparisonResult", "compare_volumes", "crop_to_common"]


@dataclass
class ComparisonResult:
    """Outcome of comparing two stacks.

    Attributes
    ----------
    shape_a / shape_b:
        Input shapes.
    shapes_match:
        Whether the inputs had identical shapes (when not, only the
        overlapping region is compared).
    n_compared:
        Number of slices (leading-axis entries) compared.
    identical:
        True when shapes match and every value is equal (NaN == NaN).
    max_abs_diff / mean_abs_diff / rmse:
        Difference statistics over the compared region.
    psnr:
        Peak signal-to-noise ratio in dB (``inf`` for identical data).
    correlation:
        Pearson correlation between the two datasets (NaN if either is
        constant).
    worst_slices:
        Indices (in the *compared* region) of up to three slices with the
        largest mean absolute difference.
    """

    shape_a: tuple
    shape_b: tuple
    shapes_match: bool
    n_compared: int
    identical: bool
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    psnr: float
    correlation: float
    worst_slices: list[int] = field(default_factory=list)

    def summary(self) -> str:
        if self.identical:
            return "compare: datasets are identical"
        return (f"compare: max |diff| {self.max_abs_diff:.4g}, "
                f"mean {self.mean_abs_diff:.4g}, rmse {self.rmse:.4g}, "
                f"psnr {self.psnr:.1f} dB, corr {self.correlation:.4f}")

    def as_report(self) -> str:
        lines = [
            f"shapes match   : {self.shapes_match}"
            + ("" if self.shapes_match else
               f"  (a {self.shape_a} vs b {self.shape_b} — comparing the "
               f"overlapping region)"),
            f"slices compared: {self.n_compared}",
        ]
        if self.identical:
            lines.append("identical      : yes")
            return "\n".join(lines)
        lines += [
            f"identical      : no",
            f"max |diff|     : {self.max_abs_diff:.6g}",
            f"mean |diff|    : {self.mean_abs_diff:.6g}",
            f"rmse           : {self.rmse:.6g}",
            f"psnr           : {self.psnr:.2f} dB",
            f"correlation    : {self.correlation:.6f}",
        ]
        if self.worst_slices:
            lines.append("worst slices   : "
                         + ", ".join(str(i) for i in self.worst_slices))
        return "\n".join(lines)


def crop_to_common(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray,
                                                          np.ndarray]:
    """Center-crop two stacks to their overlapping leading-axis extent
    and common (h, w) shape, so they can be compared plane by plane."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 2:
        a = a[None]
    if b.ndim == 2:
        b = b[None]
    nz = min(a.shape[0], b.shape[0])
    ny = min(a.shape[1], b.shape[1])
    nx = min(a.shape[2], b.shape[2])

    def crop(x: np.ndarray) -> np.ndarray:
        z0 = (x.shape[0] - nz) // 2
        y0 = (x.shape[1] - ny) // 2
        x0 = (x.shape[2] - nx) // 2
        return x[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx]

    return crop(a), crop(b)


def compare_volumes(a: np.ndarray, b: np.ndarray) -> ComparisonResult:
    """Compare two stacks and return difference statistics.

    Shape mismatches are allowed: the overlapping region (center crop)
    is compared and ``shapes_match`` is reported as False.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    shapes_match = a.shape == b.shape

    A, B = crop_to_common(a, b)
    n = A.shape[0]

    diff = A - B
    finite = np.isfinite(diff)
    if not finite.all():
        diff = np.where(finite, diff, 0.0)

    identical = bool(shapes_match and np.array_equal(a, b, equal_nan=True))

    adiff = np.abs(diff)
    max_abs = float(adiff.max()) if adiff.size else 0.0
    mean_abs = float(adiff.mean()) if adiff.size else 0.0
    rmse = float(np.sqrt((diff[finite] ** 2).mean())) if finite.any() else 0.0

    vals = np.concatenate([A[finite], B[finite]])
    vrange = float(vals.max() - vals.min()) if vals.size else 0.0
    if rmse == 0.0:
        psnr = float("inf")
    elif vrange <= 0.0:
        psnr = float("nan")
    else:
        psnr = float(20.0 * np.log10(vrange / rmse))

    fa, fb = A[finite].ravel(), B[finite].ravel()
    if fa.size > 1 and fa.std() > 0 and fb.std() > 0:
        corr = float(np.corrcoef(fa, fb)[0, 1])
    else:
        corr = float("nan")

    worst: list[int] = []
    if n > 1 and adiff.size:
        per_slice = adiff.reshape(n, -1).mean(axis=1)
        worst = [int(i) for i in np.argsort(per_slice)[::-1][:min(3, n)]]

    return ComparisonResult(
        shape_a=tuple(a.shape), shape_b=tuple(b.shape),
        shapes_match=shapes_match, n_compared=n, identical=identical,
        max_abs_diff=max_abs, mean_abs_diff=mean_abs, rmse=rmse,
        psnr=psnr, correlation=corr, worst_slices=worst,
    )
