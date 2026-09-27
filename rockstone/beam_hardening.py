"""Beam-hardening removal for rock CT data.

Beam hardening: a polychromatic X-ray beam traversing the sample gets
progressively "hardened" (its mean energy rises), so the effective
attenuation coefficient drops with thickness. In reconstructions this
shows up as *cupping* — dense material near the sample edge appears
brighter than the same material in the center. In raw projections it
appears as a nonlinearity between chord length and measured line
integral.

Two complementary strategies are provided:

- ``"radial"`` (default): fit a smooth polynomial of radius to the
  per-slice radial intensity profile of the *background phase* (air for
  stacks without an outer holder) and flatten it. This directly removes
  the classic cupping artifact on reconstructed slices.
- ``"chord"``: work on raw projections before reconstruction. The
  measured line integral is modeled as a polynomial of the true chord
  length through a (cylindrical) sample; inverting that polynomial and
  renormalizing produces linearized projections that reconstruct without
  cupping. Chords are estimated from the support width of each
  projection row.
- ``"poly"``: generic intensity remap ``I -> p(I)`` with a
  monotonicity-safe inversion, for data where a calibration
  (e.g. step wedge / phantom scan) already gives the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import cupping_index, radial_profile

__all__ = [
    "BeamHardeningResult",
    "remove_beam_hardening",
    "correct_cupping_radial",
    "correct_projections_chord",
    "polynomial_remapped",
]


@dataclass
class BeamHardeningResult:
    """Outcome of a beam-hardening correction.

    Attributes
    ----------
    method:
        ``"radial"``, ``"chord"`` or ``"poly"``.
    cupping_before / cupping_after:
        Scalar cupping severity (see :func:`rockstone.metrics.cupping_index`);
        ``nan`` when not measurable (e.g. chord mode on projections).
    coefficients:
        Fitted polynomial coefficients (method dependent).
    centers:
        Sample center ``(cx, cy)`` used per correction (radial mode).
    degrees_applied:
        Polynomial degree actually used after clamping.
    """

    method: str
    cupping_before: float = float("nan")
    cupping_after: float = float("nan")
    coefficients: np.ndarray | None = None
    centers: list[tuple[float, float]] = field(default_factory=list)
    degree: int = 0


def _fit_poly(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    if x.size < degree + 2:
        degree = max(1, x.size - 2)
    return np.polyfit(x, y, deg=degree)


def _center_from_threshold(image: np.ndarray, background: str) -> tuple[float, float]:
    """Estimate the sample center as the centroid of the sample support."""
    data = np.asarray(image, dtype=np.float64)
    if background == "dark":
        mask = data > np.nanmedian(data)
    elif background == "bright":
        mask = data < np.nanmedian(data)
    else:  # auto
        med = np.nanmedian(data)
        mask = (data > med) if np.nanmean(data > med) <= 0.5 else (data < med)
    yy, xx = np.mgrid[0: data.shape[0], 0: data.shape[1]]
    total = mask.sum()
    if total == 0:
        h, w = data.shape
        return ((w - 1) / 2.0, (h - 1) / 2.0)
    cx = float((xx * mask).sum() / total)
    cy = float((yy * mask).sum() / total)
    return (cx, cy)


def correct_cupping_radial(
    image: np.ndarray,
    degree: int = 3,
    background: str = "dark",
    center: tuple[float, float] | None = None,
    apply_to_phase: str = "bright",
) -> tuple[np.ndarray, dict]:
    """Remove cupping from a reconstructed slice via radial polynomial fit.

    The mean intensity of the phase given by ``apply_to_phase`` (the rock,
    assumed brighter than air for typical CT values) is measured in
    radial bins; a polynomial of radius is fitted to that profile and
    divided out so the profile becomes flat across the sample.

    Returns ``(corrected, info)`` where ``info`` holds the fitted
    coefficients, the center used, and the radial profiles before/after.
    """
    data = np.asarray(image, dtype=np.float64)
    if center is None:
        center = _center_from_threshold(data, background)
    r, prof = radial_profile(data, center=center, phase=apply_to_phase, n_bins=48)
    valid = np.isfinite(prof) & (r > 0)
    if valid.sum() < 4:
        return data.copy(), {"coefficients": np.array([0.0, 1.0]),
                             "center": center, "profile_before": prof,
                             "profile_after": prof, "radii": r}

    # Normalize profile by its outer value (the far field is the least
    # hardened reference in a cupped reconstruction).
    ref = np.nanmedian(prof[valid][np.argsort(r[valid])[-3:]])
    if not np.isfinite(ref) or abs(ref) < 1e-12:
        return data.copy(), {"coefficients": np.array([0.0, 1.0]),
                             "center": center, "profile_before": prof,
                             "profile_after": prof, "radii": r}
    prof_norm = prof / ref

    coeff = _fit_poly(r[valid], prof_norm[valid], degree=degree)
    coeff[degree + 1:] = 0.0
    # Cupping dips toward the center (prof_norm < 1 there), so the gain that
    # flattens the profile is the *inverse* of the fitted profile.
    gain = np.divide(1.0, np.polyval(coeff, r),
                     out=np.ones_like(r), where=np.abs(np.polyval(coeff, r)) > 1e-12)
    gain[~valid] = 1.0
    gain = np.clip(gain, 0.2, 5.0)

    yy, xx = np.mgrid[0: data.shape[0], 0: data.shape[1]]
    radius = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_map = 1.0 / np.polyval(coeff, radius)
    gain_map = np.where(np.isfinite(gain_map), gain_map, 1.0)
    gain_map = np.clip(gain_map, 0.2, 5.0)

    corrected = data * gain_map
    _, prof_after = radial_profile(corrected, center=center,
                                   phase=apply_to_phase, n_bins=48)
    info = {"coefficients": coeff, "center": center,
            "profile_before": prof, "profile_after": prof_after, "radii": r}
    return corrected, info


def _estimate_chords(projection_row: np.ndarray) -> np.ndarray:
    """Chord length per detector column for one projection row.

    The support (columns where the row's value exceeds a fraction of its
    peak) defines the sample silhouette; the chord through a parallel-ray
    projection at column ``c`` is proportional to the chord of the circle
    with that support width. Returned in normalized units where the
    maximum chord is 1.
    """
    row = np.asarray(projection_row, dtype=np.float64)
    row = row - row.min()
    peak = row.max()
    if peak <= 0:
        return np.zeros_like(row)
    support = row > 0.05 * peak
    if not support.any():
        return np.zeros_like(row)
    # Widest contiguous run of the support (robust to isolated spikes).
    padded = np.concatenate(([0], support.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    b = int(np.argmax(ends - starts))
    s, e = int(starts[b]), int(ends[b])
    support = np.zeros_like(support)
    support[s:e] = True
    width = float(e - s)
    half = width / 2.0
    # Position of columns relative to the support center.
    idx = np.arange(row.size, dtype=np.float64)
    center_idx = (s + e - 1) / 2.0
    u = (idx - center_idx) / half  # in [-1, 1] over the support
    chord = np.sqrt(np.clip(1.0 - u ** 2, 0.0, None))  # max chord = 1
    chord[~support] = 0.0
    return chord


def correct_projections_chord(
    projections: np.ndarray,
    degree: int = 3,
    layout: str = "detector",
) -> tuple[np.ndarray, dict]:
    """Linearize raw projections against chord length (beam-hardening fix).

    The measured line integral is modeled directly from the data: the
    (chord, value) point cloud is binned and aggregated with medians into
    a measured hardening curve ``P(L)``. The correction gain per chord is
    the ratio of a linear reference ``ref(L)`` (anchored to the measured
    value at the largest, least-hardened chord) to ``P(L)``; the gain is
    applied per pixel by interpolating over its estimated chord. Because
    the gain is built from measured bins with linear edge extrapolation,
    it cannot blow up where data is sparse (unlike polynomial inversion).

    Parameters
    ----------
    projections:
        ``(n, rows, cols)`` detector layout (or sinogram layout with
        ``layout="sinogram"``).
    degree:
        Polynomial degree fitted to the measured curve for reporting in
        ``info["coefficients"]`` (the correction itself is non-parametric).

    Returns
    -------
    ``(corrected, info)`` with fitted coefficients and the sample points.
    """
    data = np.asarray(projections, dtype=np.float64)
    if layout == "sinogram":
        stack = np.transpose(data, (2, 0, 1))
    elif layout == "detector":
        stack = data
    else:
        raise ValueError(f"unknown layout: {layout!r}")

    n, n_rows, n_cols = stack.shape
    # Aggregate line integrals per chord across (subsampled) projections.
    chords_all: list[np.ndarray] = []
    values_all: list[np.ndarray] = []
    step = max(1, n // 32)
    for p in range(0, n, step):
        for r in range(n_rows):
            row = stack[p, r]
            chord = _estimate_chords(row)
            if chord.max() <= 0:
                continue
            chords_all.append(chord)
            values_all.append(row)
    if not chords_all:
        raise ValueError("could not estimate sample support in projections")
    chords = np.concatenate(chords_all)
    values = np.concatenate(values_all)

    # Exclude near-zero chords (edge/noise-dominated) from the fit.
    fit_mask = (chords > 0.15) & np.isfinite(values)
    if fit_mask.sum() < 10:
        raise ValueError("too few chord samples to fit the hardening model")

    # Measured hardening curve P(L): median value per chord bin.
    bins = np.linspace(0.15, 1.0, 35)
    idx = np.digitize(chords[fit_mask], bins) - 1
    bc, bv = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 3:
            bc.append(chords[fit_mask][m].mean())
            bv.append(np.median(values[fit_mask][m]))
    bc = np.asarray(bc)
    bv = np.asarray(bv)
    if bc.size < 4:
        raise ValueError("too few populated chord bins to model hardening")
    order = np.argsort(bc)
    bc, bv = bc[order], bv[order]

    # Fit a physically-constrained model P(u) = a*u - b*u^2 (concave,
    # through origin) on the reliable interior chords only. Near-edge
    # chords are dominated by noise/partial-volume and would poison the
    # fit; the model extrapolates to them with the correct shape.
    u_min = 0.35
    rel = bc >= u_min
    if rel.sum() < 4:
        rel = np.ones_like(bc, dtype=bool)
    A = np.stack([bc[rel], -(bc[rel] ** 2)], axis=1)
    (a_coef, b_coef), *_ = np.linalg.lstsq(A, bv[rel], rcond=None)
    if a_coef <= 0:
        raise ValueError("degenerate hardening model (non-positive slope)")
    b_coef = max(0.0, float(b_coef))  # keep the model concave

    u_max = float(bc[rel][-1])
    scale = (a_coef * u_max - b_coef * u_max ** 2) / u_max
    grid = np.linspace(0.0, 1.0, 512)
    p_model = scale * grid                      # linear reference
    p_fit = a_coef * grid - b_coef * grid ** 2  # hardened model
    gain_curve = np.divide(p_model, p_fit, out=np.ones_like(grid),
                           where=np.abs(p_fit) > 1e-12)
    gain_curve = np.clip(gain_curve, 0.2, 5.0)

    # Polynomial of the measured curve, kept for reporting/inspection.
    coeff = _fit_poly(bc, bv, degree=degree)

    corrected = np.empty_like(stack)
    for p in range(stack.shape[0]):
        for r in range(n_rows):
            row = stack[p, r]
            chord = _estimate_chords(row)
            if chord.max() <= 0:
                corrected[p, r] = row
                continue
            corrected[p, r] = row * np.interp(chord, grid, gain_curve)

    out = np.transpose(corrected, (1, 2, 0)) if layout == "sinogram" else corrected
    info = {"coefficients": coeff, "chord_samples": bc,
            "value_samples": bv, "gain_curve": gain_curve,
            "model": (float(a_coef), float(b_coef), float(u_max))}
    return out, info


def polynomial_remapped(
    data: np.ndarray,
    coefficients: np.ndarray,
    monotonic_safe: bool = True,
) -> np.ndarray:
    """Apply a calibrated polynomial intensity remap ``I -> p(I)``.

    ``coefficients`` are in :func:`numpy.polyfit` order (highest first).
    With ``monotonic_safe`` the map is evaluated on a dense grid and
    linearly inverted where it is locally non-monotonic, avoiding the
    fold-back that naive polynomial evaluation produces.
    """
    data = np.asarray(data, dtype=np.float64)
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    grid = np.linspace(lo, hi, 4096)
    vals = np.polyval(coefficients, grid)
    if monotonic_safe:
        # Enforce non-decreasing output by cumulative maximum.
        vals = np.maximum.accumulate(vals)
    out = np.interp(data.ravel(), grid, vals).reshape(data.shape)
    return out


def remove_beam_hardening(
    data: np.ndarray,
    method: str = "radial",
    degree: int = 3,
    background: str = "dark",
    center: tuple[float, float] | None = None,
    layout: str = "detector",
    coefficients: np.ndarray | None = None,
    progress: bool = False,
) -> tuple[np.ndarray, BeamHardeningResult]:
    """Remove beam-hardening artifacts from rock CT data.

    Parameters
    ----------
    data:
        Either a reconstructed stack ``(nz, ny, nx)`` (``method="radial"``
        or ``"poly"``) or raw projections (``method="chord"``).
    method:
        - ``"radial"``: per-slice radial cupping flattening.
        - ``"chord"``: chord-length linearization of raw projections.
        - ``"poly"``: apply an externally supplied polynomial remap
          (``coefficients`` must be given).
    degree:
        Polynomial degree for the radial/chord models (2-4 works best;
        cupping is dominantly quadratic).
    background:
        ``"dark"`` (air around rock) or ``"bright"`` (bright holder) —
        used only for center estimation in radial mode.
    center:
        Explicit sample center ``(cx, cy)`` for radial mode; estimated
        per slice when omitted.
    layout:
        ``"detector"`` or ``"sinogram"`` when ``method="chord"``.
    coefficients:
        Polynomial for ``method="poly"`` (numpy.polyfit order).
    progress:
        Print a per-slice progress line (radial mode).

    Returns
    -------
    ``(corrected, BeamHardeningResult)``
    """
    data = np.asarray(data)

    if method == "poly":
        if coefficients is None:
            raise ValueError("method='poly' requires `coefficients`")
        corrected = polynomial_remapped(data, np.asarray(coefficients))
        return corrected, BeamHardeningResult(method="poly", coefficients=np.asarray(coefficients))

    if method == "radial":
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim != 3:
            raise ValueError(f"expected a 2D slice or 3D stack, got shape {data.shape}")

        cup_before = cupping_index(data[len(data) // 2], background=background) \
            if len(data) else float("nan")
        out = np.empty_like(data, dtype=np.float64)
        centers: list[tuple[float, float]] = []
        coeffs: list[np.ndarray] = []
        for i in range(data.shape[0]):
            corrected, info = correct_cupping_radial(
                data[i], degree=degree, background=background, center=center
            )
            out[i] = corrected
            centers.append(tuple(info["center"]))
            coeffs.append(info["coefficients"])
            if progress:
                print(f"  slice {i + 1}/{data.shape[0]} corrected (radial cupping)")
        cup_after = cupping_index(out[len(out) // 2], background=background) \
            if len(out) else float("nan")
        result = BeamHardeningResult(
            method="radial",
            cupping_before=float(cup_before),
            cupping_after=float(cup_after),
            coefficients=coeffs[-1] if coeffs else None,
            centers=centers,
            degree=degree,
        )
        if out.shape[0] == 1:
            out = out[0]
        return out, result

    if method == "chord":
        corrected, info = correct_projections_chord(
            data, degree=degree, layout=layout
        )
        return corrected, BeamHardeningResult(
            method="chord", coefficients=info["coefficients"], degree=degree
        )

    raise ValueError(f"unknown method: {method!r}")
