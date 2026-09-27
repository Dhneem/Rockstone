"""Rigid per-slice alignment for reconstructed stacks and raw projections.

Two building blocks:

- phase correlation on windowed images gives the sub-pixel translation
  between consecutive slices;
- a log-polar resampling turns rotation into a circular shift along the
  angular axis, which the same phase-correlation machinery measures
  (Fourier-Mellin style).

Per-slice transforms between neighbours are composed into one global
motion trajectory, optionally smoothed/regularized, and finally applied
to the stack so that all slices share a common frame.

For raw projection data the analog problem is lateral detector jitter
between projections: :func:`align_projections` estimates a lateral shift
per projection via 1-D phase correlation of column profiles and re-shifts
each projection, with optional polynomial smoothing of the motion.

Sign conventions (kept consistent throughout, and pinned by unit tests):

- :func:`rockstone.transforms.apply_transform` with ``T`` moves image
  *content* by ``T`` (rotation about the pivot, then translation);
- ``pairwise_transform(ref, mov)`` returns ``T`` such that
  ``apply_transform(mov, T)`` matches ``ref``;
- ``subpixel_shift(a, b)`` returns ``d`` such that ``b(p) = a(p - d)``
  ("b is a shifted by d").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transforms import RigidTransform, apply_transform

__all__ = [
    "AlignmentResult",
    "align_slices",
    "align_projections",
    "pairwise_transform",
    "estimate_rotation",
    "subpixel_shift",
]


@dataclass
class AlignmentResult:
    """Outcome of a stack alignment.

    Attributes
    ----------
    transforms:
        One :class:`RigidTransform` per slice, moving the original slice
        content into the aligned frame (slice ``reference`` is identity).
    rotations:
        Cumulative rotation angles in radians.
    shifts:
        ``(nz, 2)`` array of cumulative ``(dx, dy)`` per slice.
    residual_scores:
        Phase-correlation peak quality of each consecutive pair (higher is
        better; values < ~0.1 mean the pair could not be matched well).
    smoothed:
        Whether the trajectory was regularized before application.
    """

    transforms: list[RigidTransform]
    residual_scores: list[float]
    smoothed: bool = False
    rotations: list[float] = field(default_factory=list)
    shifts: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.rotations = [t.theta for t in self.transforms]
        self.shifts = np.array([[t.dx, t.dy] for t in self.transforms])


def _hanning2d(shape: tuple[int, int]) -> np.ndarray:
    wy = np.hanning(shape[0])[:, None]
    wx = np.hanning(shape[1])[None, :]
    return np.clip(wy @ wx, 1e-6, None)


def subpixel_shift(
    a: np.ndarray, b: np.ndarray, upsample: int = 20
) -> tuple[tuple[float, float], float]:
    """Sub-pixel translation between images ``a`` and ``b``.

    Returns ``((dy, dx), quality)`` such that ``b(p) = a(p - (dy, dx))``.
    Uses phase correlation with parabolic peak refinement; ``quality`` is
    the normalized peak height in [0, 1].
    """
    win = _hanning2d(a.shape)
    fa = np.fft.fft2(np.asarray(a, dtype=np.float64) * win)
    fb = np.fft.fft2(np.asarray(b, dtype=np.float64) * win)
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifft2(cross).real

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    quality = float(corr[peak])
    h, w = corr.shape
    dy, dx = int(peak[0]), int(peak[1])
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w

    if upsample > 1:
        for axis, idx in ((0, peak[0]), (1, peak[1])):
            lo = (idx - 1) % corr.shape[axis]
            hi = (idx + 1) % corr.shape[axis]
            if axis == 0:
                left, center, right = corr[lo, peak[1]], corr[peak], corr[hi, peak[1]]
            else:
                left, center, right = corr[peak[0], lo], corr[peak], corr[peak[0], hi]
            denom = left - 2 * center + right
            if abs(denom) > 1e-12:
                delta = 0.5 * (left - right) / denom
                if abs(delta) <= 0.5:
                    if axis == 0:
                        dy += float(delta)
                    else:
                        dx += float(delta)
    return (float(dy), float(dx)), quality


def _phase_corr_1d(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Circular shift ``d`` with ``b[k] = a[k - d]`` along axis 0.

    ``a`` and ``b`` are 2-D arrays; the correlation profile is averaged
    over the remaining axis. Returns ``(shift, quality)``.
    """
    fa = np.fft.fft(a, axis=0)
    fb = np.fft.fft(b, axis=0)
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifft(cross, axis=0).real
    profile = corr.mean(axis=1)
    n = profile.shape[0]
    peak = int(np.argmax(profile))
    shift = float(peak if peak <= n // 2 else peak - n)
    lo, hi = (peak - 1) % n, (peak + 1) % n
    denom = profile[lo] - 2 * profile[peak] + profile[hi]
    if abs(denom) > 1e-12:
        delta = 0.5 * (profile[lo] - profile[hi]) / denom
        if abs(delta) <= 0.5:
            shift += float(delta)
    return shift, float(profile[peak])


def _subpixel_shift_1d(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """1-D phase correlation: returns ``(shift, quality)`` with ``b = a`` shifted."""
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifft(cross).real
    n = corr.size
    peak = int(np.argmax(corr))
    shift = float(peak if peak <= n // 2 else peak - n)
    lo, hi = (peak - 1) % n, (peak + 1) % n
    denom = corr[lo] - 2 * corr[peak] + corr[hi]
    if abs(denom) > 1e-12:
        delta = 0.5 * (corr[lo] - corr[hi]) / denom
        if abs(delta) <= 0.5:
            shift += float(delta)
    return shift, float(corr[peak])


def _log_polar_mag(image: np.ndarray, theta_bins: int, log_bins: int,
                   min_radius: float, max_radius: float) -> np.ndarray:
    """Log-polar resampled Fourier log-magnitude, shape ``(log_bins, theta_bins)``.

    The spectrum is sampled around DC using wrapped indices, so rotation
    of the image content becomes a circular shift along the theta axis.
    """
    h, w = image.shape
    cy, cx = h // 2, w // 2  # DC bin after fftshift
    max_radius = min(max_radius, min(cy, cx) - 1)
    thetas = np.linspace(0.0, 2.0 * np.pi, theta_bins, endpoint=False)
    log_r = np.linspace(np.log(min_radius), np.log(max_radius), log_bins)
    rr = np.exp(log_r)
    tt, rr = np.meshgrid(thetas, rr)
    yy = cy + rr * np.sin(tt)
    xx = cx + rr * np.cos(tt)

    win = _hanning2d(image.shape)
    F = np.fft.fftshift(np.fft.fft2(np.asarray(image, dtype=np.float64) * win))
    log_mag = np.log(np.abs(F) + 1e-9)

    r0 = np.floor(yy).astype(np.int64) % h
    r1 = (r0 + 1) % h
    c0 = np.floor(xx).astype(np.int64) % w
    c1 = (c0 + 1) % w
    dr = yy - np.floor(yy)
    dc = xx - np.floor(xx)
    out = (
        log_mag[r0, c0] * (1 - dr) * (1 - dc)
        + log_mag[r0, c1] * (1 - dr) * dc
        + log_mag[r1, c0] * dr * (1 - dc)
        + log_mag[r1, c1] * dr * dc
    )
    # Remove the theta-independent radial background (the strong low-frequency
    # ring), which would otherwise dominate the correlation.
    out -= out.mean(axis=1, keepdims=True)
    return out


def _refine_rotation(ref: np.ndarray, mov: np.ndarray, theta0: float,
                     half_range_deg: float = 2.0, n_steps: int = 7,
                     iterations: int = 2) -> tuple[float, float]:
    """Refine a coarse rotation estimate by direct image correlation.

    Sweeps angles around ``theta0`` (shrinking the span each iteration)
    and keeps the angle whose de-rotated ``mov`` gives the best
    phase-correlation peak against ``ref``. This recovers sub-degree
    accuracy that the log-polar grid (1 degree/bin) cannot provide.
    """
    h, w = ref.shape
    pivot = ((w - 1) / 2.0, (h - 1) / 2.0)
    best_theta, best_q = theta0, -np.inf
    theta = theta0
    for it in range(iterations):
        span = np.radians(half_range_deg) / (2 ** it)
        for d in np.linspace(-span, span, n_steps):
            cand = theta + d
            derot = apply_transform(
                mov, RigidTransform(theta=-cand, pivot=pivot), cval=0.0
            )
            (_dy, _dx), q = subpixel_shift(ref, derot, upsample=5)
            if q > best_q:
                best_q, best_theta = q, cand
        theta = best_theta
    return best_theta, best_q


def estimate_rotation(a: np.ndarray, b: np.ndarray,
                      theta_bins: int = 360,
                      refine: bool = True) -> tuple[float, float]:
    """In-plane rotation between two images (Fourier-Mellin).

    Returns ``(angle, quality)`` such that ``b`` is ``a`` rotated by
    ``angle`` radians (content-wise, about the image center), angle in
    ``[-pi, pi)``. The coarse log-polar estimate is refined in the image
    domain when ``refine`` is True.
    """
    min_radius = max(8.0, min(a.shape) * 0.08)
    max_radius = min(a.shape) * 0.45
    la = _log_polar_mag(a, theta_bins, 128, min_radius, max_radius)
    lb = _log_polar_mag(b, theta_bins, 128, min_radius, max_radius)
    # If b = a rotated by phi, then lb(k) = la(k - phi * bins / 2pi).
    shift, q = _phase_corr_1d(la.T, lb.T)  # axis 0 = theta
    phi = shift * 2.0 * np.pi / theta_bins
    if phi >= np.pi:
        phi -= 2.0 * np.pi
    if phi < -np.pi:
        phi += 2.0 * np.pi
    if refine:
        phi_r, q_r = _refine_rotation(a, b, phi)
        if q_r > q:
            return float(phi_r), float(q_r)
    return float(phi), q


def pairwise_transform(
    ref: np.ndarray, mov: np.ndarray,
    estimate_theta: bool = True,
    upsample: int = 20,
) -> tuple[RigidTransform, float]:
    """Estimate the rigid transform taking ``mov`` onto ``ref``.

    Returns ``(T, quality)`` such that ``apply_transform(mov, T)``
    matches ``ref``. Translation is measured after de-rotating ``mov``,
    which decouples rotation from translation. A candidate rotated by an
    extra 180 degrees is also scored to guard against ambiguous peaks.
    """
    h, w = ref.shape
    pivot = ((w - 1) / 2.0, (h - 1) / 2.0)

    if estimate_theta:
        theta_c, _ = estimate_rotation(ref, mov)  # mov content = ref content rotated by theta_c
        candidates = [theta_c, theta_c + np.pi]
    else:
        candidates = [0.0]

    best: tuple[RigidTransform, float] | None = None
    for tc in candidates:
        # De-rotate mov content by -tc (apply_transform moves content by T).
        mov_derot = apply_transform(
            mov, RigidTransform(theta=-tc, pivot=pivot), output_shape=ref.shape
        )
        (dy, dx), q = subpixel_shift(ref, mov_derot, upsample=upsample)
        # mov_derot = ref shifted by +d  =>  T = S_{-d} . D  (module docstring)
        T = RigidTransform(theta=-tc, dx=-dx, dy=-dy, pivot=pivot)
        if best is None or q > best[1]:
            best = (T, q)
    assert best is not None
    return best


def _robust_poly_smooth(values: np.ndarray, degree: int = 2,
                        n_iter: int = 3) -> np.ndarray:
    """Polynomial smoothing with iterative outlier rejection."""
    n = values.size
    x = np.linspace(-1.0, 1.0, n)
    mask = np.ones(n, dtype=bool)
    out = values.copy()
    for _ in range(n_iter):
        coeffs = np.polyfit(x[mask], values[mask], deg=degree)
        out = np.polyval(coeffs, x)
        resid = values - out
        center = np.median(resid[mask])
        sigma = 1.4826 * np.median(np.abs(resid[mask] - center))
        if sigma < 1e-12:
            break
        new_mask = np.abs(resid - center) < 3.0 * sigma
        if new_mask.sum() < degree + 2 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return out


def _running_median(values: np.ndarray, window: int) -> np.ndarray:
    w = max(3, int(window) | 1)  # force odd
    half = w // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[k:k + w]) for k in range(values.size)])


def _smooth_trajectory(values: np.ndarray, mode: str, degree: int) -> np.ndarray:
    if mode == "poly":
        deg = max(1, min(int(degree), values.size - 2))
        return _robust_poly_smooth(values, degree=deg)
    if mode == "median":
        return _running_median(values, degree)
    raise ValueError(f"unknown smoothing mode: {mode!r}")


def _center_params(T: RigidTransform, center: np.ndarray) -> tuple[float, float, float]:
    """Re-express ``T`` as (theta, dx, dy) rotating about ``center``."""
    M = T.as_matrix()
    R = M[:2, :2]
    t0 = M[:2, 2]
    tc = t0 + (R - np.eye(2)) @ center
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    return theta, float(tc[0]), float(tc[1])


def align_slices(
    volume: np.ndarray,
    estimate_theta: bool = True,
    smoothing: str = "none",
    smoothing_degree: int = 3,
    reference: int | str = "first",
    interpolation_order: int = 1,
    progress: bool = False,
) -> tuple[np.ndarray, AlignmentResult]:
    """Rigidly align the slices of a reconstructed stack.

    Parameters
    ----------
    volume:
        ``(nz, ny, nx)`` stack.
    estimate_theta:
        Also correct in-plane rotation (Fourier-Mellin). Disable for speed
        when only drift is expected.
    smoothing:
        ``"none"``, ``"poly"`` (robust polynomial fit of the motion
        trajectory) or ``"median"`` (running median).
    smoothing_degree:
        Polynomial degree for ``"poly"`` or window for ``"median"``.
    reference:
        ``"first"`` (slice 0), ``"middle"`` (``nz // 2``) or an int index.
    interpolation_order:
        1 (bilinear, default) or 0 (nearest) when warping.
    progress:
        Print a per-slice progress line.

    Returns
    -------
    ``(aligned_volume, AlignmentResult)``
    """
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D stack, got shape {vol.shape}")
    nz = vol.shape[0]
    if nz < 2:
        return vol.copy(), AlignmentResult(
            transforms=[RigidTransform()], residual_scores=[1.0]
        )

    # Relative motion: rel_i moves slice i's content into slice i-1's frame.
    rel: list[RigidTransform] = []
    scores: list[float] = []
    for i in range(1, nz):
        t, q = pairwise_transform(vol[i - 1], vol[i], estimate_theta=estimate_theta)
        rel.append(t)
        scores.append(q)
        if progress:
            print(f"  slice {i}/{nz - 1}: dx={t.dx:+.2f} dy={t.dy:+.2f} "
                  f"rot={np.degrees(t.theta):+.3f} deg  (q={q:.3f})")

    # Cumulative content moves: cumulative[i] moves slice i into slice 0's
    # frame (apply rel_i first, then cumulative_{i-1}).
    cumulative = [RigidTransform()]
    for i in range(1, nz):
        cumulative.append(rel[i - 1].composed_with(cumulative[i - 1]))

    if reference == "first":
        ref_idx = 0
    elif reference == "middle":
        ref_idx = nz // 2
    else:
        ref_idx = int(reference) % nz
    # Rebase so the chosen reference slice is the identity: apply
    # cumulative[i] first, then the inverse of cumulative[ref_idx].
    inv_ref = cumulative[ref_idx].inverse()
    final = [t.composed_with(inv_ref) for t in cumulative]

    smoothed = False
    if smoothing != "none":
        smoothed = True
        center = np.array([(vol.shape[2] - 1) / 2.0, (vol.shape[1] - 1) / 2.0])
        params = np.array([_center_params(t, center) for t in final])
        thetas_s = _smooth_trajectory(params[:, 0], smoothing, smoothing_degree)
        dxs_s = _smooth_trajectory(params[:, 1], smoothing, smoothing_degree)
        dys_s = _smooth_trajectory(params[:, 2], smoothing, smoothing_degree)
        final = [
            RigidTransform(theta=float(thetas_s[i]), dx=float(dxs_s[i]),
                           dy=float(dys_s[i]),
                           pivot=(float(center[0]), float(center[1])))
            for i in range(nz)
        ]

    result = AlignmentResult(transforms=final,
                             residual_scores=[1.0] + scores,
                             smoothed=smoothed)

    aligned = np.empty_like(vol)
    for i in range(nz):
        aligned[i] = apply_transform(
            vol[i], final[i], output_shape=vol.shape[1:],
            order=interpolation_order, cval=0.0,
        )
    return aligned, result


def _shift_along_columns(arr: np.ndarray, shift: float) -> np.ndarray:
    """Shift a 2-D array's content along axis 1 by a sub-pixel amount."""
    if abs(shift) < 1e-9:
        return arr
    n_cols = arr.shape[1]
    src = np.arange(n_cols, dtype=np.float64) - shift
    src0 = np.clip(np.floor(src).astype(np.int64), 0, n_cols - 1)
    src1 = np.clip(src0 + 1, 0, n_cols - 1)
    w1 = np.clip(src - src0, 0.0, 1.0)
    return arr[:, src0] * (1 - w1)[None, :] + arr[:, src1] * w1[None, :]


def align_projections(
    projections: np.ndarray,
    layout: str = "detector",
    smoothing: str = "poly",
    smoothing_degree: int = 5,
    reference: int | str = "middle",
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct lateral detector jitter between projections.

    Each projection is compared with its neighbour by phase correlation of
    their column profiles (mean over detector rows); the resulting shift
    curve over projection index is optionally smoothed and applied with
    sub-pixel interpolation along the detector column axis.

    Parameters
    ----------
    projections:
        Raw projection stack. ``layout="detector"`` (default) expects
        ``(n_projections, det_rows, det_cols)``; ``layout="sinogram"``
        expects ``(det_rows, det_cols, n_projections)``.
    smoothing:
        ``"none"``, ``"poly"`` or ``"median"`` for the shift curve
        (detector jitter is smooth, so ``"poly"`` is the default).
    smoothing_degree:
        Polynomial degree / median window for the smoothing.
    reference:
        Projection index (or ``"middle"``/``"first"``) whose position is
        kept fixed.
    progress:
        Print a per-projection progress line.

    Returns
    -------
    ``(aligned, shifts)`` with ``aligned`` in the same layout as the input
    and ``shifts[p]`` the lateral shift applied to projection ``p``.
    """
    data = np.asarray(projections, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D projection stack, got {data.shape}")
    if layout == "sinogram":
        stack = np.transpose(data, (2, 0, 1))  # -> (n, rows, cols)
    elif layout == "detector":
        stack = data
    else:
        raise ValueError(f"unknown layout: {layout!r}")

    n, n_rows, n_cols = stack.shape
    profiles = stack.mean(axis=1)  # (n, n_cols) column profiles

    if reference == "first":
        ref_idx = 0
    elif reference == "middle":
        ref_idx = n // 2
    else:
        ref_idx = int(reference) % n

    # Chain pairwise shifts between adjacent projections (adjacent views
    # are similar, so this is robust to the object's angular evolution)
    # and integrate them into a drift curve relative to the reference.
    drift = np.zeros(n)
    for p in range(ref_idx + 1, n):
        d, q = _subpixel_shift_1d(profiles[p - 1], profiles[p])
        drift[p] = drift[p - 1] + d
        if progress:
            print(f"  projection {p}/{n - 1}: d={d:+.2f} (q={q:.3f})")
    for p in range(ref_idx - 1, -1, -1):
        d, q = _subpixel_shift_1d(profiles[p + 1], profiles[p])
        drift[p] = drift[p + 1] + d
        if progress:
            print(f"  projection {p}/{n - 1}: d={d:+.2f} (q={q:.3f})")

    if smoothing == "poly":
        deg = max(1, min(int(smoothing_degree), n - 2))
        drift = _robust_poly_smooth(drift, degree=deg)
    elif smoothing == "median":
        drift = _running_median(drift, smoothing_degree)

    # Undo the drift: shift each projection's content by -drift.
    aligned = np.empty_like(stack)
    for p in range(n):
        aligned[p] = _shift_along_columns(stack[p], -drift[p])

    if layout == "sinogram":
        aligned = np.transpose(aligned, (1, 2, 0))
    return aligned, -drift
