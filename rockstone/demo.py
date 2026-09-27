"""Synthetic rock-sample data for tests and the ``rockstone demo`` command.

The synthetic stack uses the *same* base rock texture in every slice so
that alignment has real structure to lock onto; per-slice rigid jitter
and a physically-shaped cupping field are then applied. Raw projections
are generated with a concave chord/line-integral relation (the
beam-hardening signature) plus lateral detector jitter.
"""

from __future__ import annotations

import numpy as np

from .transforms import RigidTransform, apply_transform

__all__ = ["make_rock_slice", "make_rock_stack", "make_projection_stack",
           "run_demo"]


def _porous_rock(shape: tuple[int, int], seed: int,
                 n_minerals: int = 4) -> np.ndarray:
    """A grayscale 'rock' texture: mineral grains + pores, values in [0, 1]."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.full((h, w), 0.55)

    # Mineral grains: smooth random blobs.
    for _ in range(n_minerals):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        amp = rng.uniform(0.05, 0.2)
        sigma = rng.uniform(0.1, 0.35) * min(h, w)
        base += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))

    # Pores: small dark blobs.
    n_pores = int(min(h, w) ** 2 / 900)
    for _ in range(n_pores):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(1.5, 4.5)
        base[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] -= 0.3

    return np.clip(base, 0.05, 1.0)


def make_rock_slice(size: int = 256, seed: int = 0,
                    radius: float | None = None) -> np.ndarray:
    """A cylindrical rock slice: porous texture inside a circular core."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    if radius is None:
        radius = size * 0.42
    inside = (yy - c) ** 2 + (xx - c) ** 2 < radius ** 2
    img = np.zeros((size, size))
    img[inside] = _porous_rock((size, size), seed)[inside]
    return img


def make_rock_stack(n_slices: int = 24, size: int = 160, seed: int = 0,
                    jitter: float = 2.5, jitter_rot: float = 0.5,
                    cupping: float = 0.35,
                    static_texture: bool = True) -> tuple[np.ndarray, list[RigidTransform]]:
    """A synthetic micro-CT stack of one rock sample.

    Every slice shows the same rock (``static_texture``) with independent
    rigid jitter (translation + rotation about the center), then a
    multiplicative cupping dip toward the center, plus noise.

    Returns
    -------
    ``(volume, true_transforms)`` — the corrupted stack and the per-slice
    content moves ``T_i`` that were applied (``slice_i = warp(clean, T_i)``).
    """
    rng = np.random.default_rng(seed)
    true: list[RigidTransform] = []
    vol = np.empty((n_slices, size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / (size * 0.42)
    gain = 1.0 - cupping * (1.0 - np.clip(r, 0, 1) ** 2)

    for i in range(n_slices):
        seed_i = seed if static_texture else seed + 1000 + i
        clean = make_rock_slice(size, seed=seed_i)
        T = RigidTransform(
            theta=float(np.radians(rng.uniform(-jitter_rot, jitter_rot))),
            dx=float(rng.uniform(-jitter, jitter)),
            dy=float(rng.uniform(-jitter, jitter)),
        )
        true.append(T)
        img = apply_transform(clean, T, order=1, cval=0.0)
        img = img * gain
        img += rng.normal(0, 0.01, img.shape)
        vol[i] = np.clip(img, 0.0, 2.0)
    return vol, true


def make_projection_stack(n_projections: int = 120, det_rows: int = 64,
                          det_cols: int = 96, seed: int = 0,
                          jitter: float = 1.5,
                          hardening: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic raw projections of a cylindrical rock with beam hardening.

    A parallel-beam forward projector is simulated: the 2-D rock texture
    is rotated by each view angle and summed along the ray direction.
    Beam hardening is applied per ray as a concave function of the
    geometric chord length ``L`` through the cylinder,
    ``P = mu * L * ((1 + k) - k * L / L_max)``, so long (central) chords
    are under-attenuated — the classic hardening signature that
    reconstructs to a cupped slice. Lateral detector jitter is added per
    projection.

    Returns
    -------
    ``(stack, true_shifts)`` with ``stack`` shaped
    ``(n_projections, det_rows, det_cols)``.
    """
    rng = np.random.default_rng(seed)
    k = float(hardening)

    # Rock object: mostly uniform matrix + moderate texture (a cylinder of
    # radius R sampled on a grid slightly larger than the detector width).
    size = int(det_cols * 1.3)
    yy, xx = np.mgrid[0:size, 0:size]
    c_obj = (size - 1) / 2.0
    radius = det_cols * 0.4
    mu = np.zeros((size, size))
    texture = _porous_rock((size, size), seed=seed)
    mu_obj = 0.55 + 0.15 * (texture - texture.mean())
    inside = (yy - c_obj) ** 2 + (xx - c_obj) ** 2 < radius ** 2
    mu[inside] = mu_obj[inside]

    c_det = (det_cols - 1) / 2.0
    # Geometric chord (in pixels) for each detector column at angle theta.
    cols = np.arange(det_cols, dtype=np.float64) - c_det
    chord_of = lambda s: 2.0 * np.sqrt(np.clip(radius ** 2 - s ** 2, 0, None))
    L_max = 2.0 * radius

    stack = np.zeros((n_projections, det_rows, det_cols))
    true_shifts = rng.uniform(-jitter, jitter, n_projections)
    angles = np.linspace(0.0, np.pi, n_projections, endpoint=False)
    for p, ang in enumerate(angles):
        # Rotate the object by the view angle and integrate along rows.
        rotated = apply_transform(mu, RigidTransform(theta=ang,
                                                     pivot=(c_obj, c_obj)),
                                  output_shape=(size, size), cval=0.0)
        profile = rotated.sum(axis=0)  # ray sums along y for each column
        # Pad/crop the profile to the detector width (centered).
        start = (size - det_cols) // 2
        profile = profile[start:start + det_cols]

        # Beam hardening: concave in the geometric chord of THIS view.
        L = chord_of(cols)
        gain = np.where(L > 0, (1.0 + k) - k * (L / L_max), 1.0)
        profile = profile * gain

        img = np.tile(profile, (det_rows, 1))
        img += rng.normal(0, 0.002 * img.max(), img.shape)
        stack[p] = np.roll(img, int(round(true_shifts[p])), axis=1)
    return stack, true_shifts


def make_phantom_projections(n_projections: int = 120, det_rows: int = 32,
                             det_cols: int = 96, seed: int = 0,
                             jitter: float = 0.0,
                             hardening: float = 0.4,
                             mu: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Projections of a *uniform* disc (calibration phantom).

    Ground truth is a flat reconstruction with attenuation ``mu``; any
    cupping after FBP is purely beam hardening, which makes this the
    reference dataset for validating chord correction.
    """
    rng = np.random.default_rng(seed)
    k = float(hardening)
    size = int(det_cols * 1.3)
    c_obj = (size - 1) / 2.0
    radius = det_cols * 0.4
    yy, xx = np.mgrid[0:size, 0:size]
    mu_map = np.where((yy - c_obj) ** 2 + (xx - c_obj) ** 2 < radius ** 2, mu, 0.0)

    c_det = (det_cols - 1) / 2.0
    cols = np.arange(det_cols, dtype=np.float64) - c_det
    chord_of = lambda s: 2.0 * np.sqrt(np.clip(radius ** 2 - s ** 2, 0, None))
    L_max = 2.0 * radius

    stack = np.zeros((n_projections, det_rows, det_cols))
    true_shifts = rng.uniform(-jitter, jitter, n_projections)
    angles = np.linspace(0.0, np.pi, n_projections, endpoint=False)
    for p, ang in enumerate(angles):
        rotated = apply_transform(mu_map, RigidTransform(theta=ang,
                                                         pivot=(c_obj, c_obj)),
                                  output_shape=(size, size), cval=0.0)
        profile = rotated.sum(axis=0)
        start = (size - det_cols) // 2
        profile = profile[start:start + det_cols]
        L = chord_of(cols)
        gain = np.where(L > 0, (1.0 + k) - k * (L / L_max), 1.0)
        profile = profile * gain
        img = np.tile(profile, (det_rows, 1))
        img += rng.normal(0, 0.002 * img.max(), img.shape)
        stack[p] = np.roll(img, int(round(true_shifts[p])), axis=1)
    return stack, true_shifts


def run_demo(out_dir: str = "rockstone_demo", n_slices: int = 24,
             size: int = 160) -> int:
    """Generate a synthetic rock stack, run align + deharden, print report."""
    import tifffile
    from pathlib import Path

    from .align import pairwise_transform
    from .beam_hardening import remove_beam_hardening
    from .metrics import cupping_index
    from .pipeline import process

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"generating synthetic rock stack ({n_slices} slices, {size}px) ...")
    vol, true = make_rock_stack(n_slices=n_slices, size=size, cupping=0.35)
    tifffile.imwrite(out / "rock_stack.tif", vol)

    report = process(
        out / "rock_stack.tif",
        output_path=out / "rock_processed.tif",
        align=True,
        deharden=True,
    )
    print(report.summary())

    # --- ground-truth check of the alignment -------------------------------
    from .evaluation import ground_truth_corrections

    expected = ground_truth_corrections(true, reference=0)
    devs = []
    for est, exp in zip(report.alignment.transforms, expected):
        devs.append(max(abs(est.dx - exp.dx), abs(est.dy - exp.dy),
                        abs(np.degrees(est.theta - exp.theta))))
    print(f"alignment vs ground truth: max deviation "
          f"{max(devs):.2f} px/deg, mean {float(np.mean(devs)):.2f}")

    # --- residual drift between consecutive aligned slices ------------------
    aligned = tifffile.imread(out / "rock_processed.tif")
    resid = []
    for i in range(1, len(aligned)):
        t, _ = pairwise_transform(aligned[i - 1], aligned[i], estimate_theta=False)
        resid.append(np.hypot(t.dx, t.dy))
    print(f"residual inter-slice drift after alignment: "
          f"mean {np.mean(resid):.3f} px, max {np.max(resid):.3f} px")

    # --- cupping metrics -----------------------------------------------------
    cb = cupping_index(vol[len(vol) // 2])
    ca = cupping_index(aligned[len(aligned) // 2])
    print(f"cupping index: {cb:.3f} -> {ca:.3f}")

    print(f"demo written to {out}/")
    return 0
