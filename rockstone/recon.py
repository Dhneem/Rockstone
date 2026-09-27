"""Minimal parallel-beam filtered backprojection (FBP) reconstruction.

This is a small utility (no third-party tomography dependencies) that
serves two purposes:

- reconstruct corrected projections so beam-hardening fixes can be
  verified end-to-end (cupping index on the reconstruction);
- let users turn processed raw data back into slices.
"""

from __future__ import annotations

import numpy as np

__all__ = ["reconstruct_fbp", "ram_lak_filter"]


def ram_lak_filter(n: int, window: str = "hann") -> np.ndarray:
    """Ramp filter response for odd-length FFT of size ``n``."""
    f = np.fft.rfftfreq(n)
    ramp = 2.0 * np.abs(f)
    if window == "hann":
        w = 0.5 * (1.0 + np.cos(2.0 * np.pi * f / (f.max() + 1e-12)))
        ramp = ramp * w
    elif window not in ("none", "ram-lak"):
        raise ValueError(f"unknown window: {window!r}")
    return ramp


def reconstruct_fbp(
    sinogram: np.ndarray,
    layout: str = "angle",
    window: str = "hann",
    output_size: int | None = None,
) -> np.ndarray:
    """Reconstruct a 2-D slice from a 2-D parallel-beam sinogram.

    Parameters
    ----------
    sinogram:
        ``(n_angles, n_cols)`` when ``layout="angle"`` (default) or
        ``(n_cols, n_angles)`` when ``layout="detector"``.
    window:
        ``"hann"`` (apodized ramp, fewer artifacts) or ``"none"``.
    output_size:
        Size of the square reconstruction; defaults to ``n_cols``.

    Returns
    -------
    The reconstructed slice ``(output_size, output_size)``. Scaling is
    arbitrary but consistent; contrast is what matters for QC.
    """
    sino = np.asarray(sinogram, dtype=np.float64)
    if layout == "detector":
        sino = sino.T
    elif layout != "angle":
        raise ValueError(f"unknown layout: {layout!r}")
    n_angles, n_cols = sino.shape
    size = output_size or n_cols

    # Zero-pad projections to a power of two for clean FFTs.
    pad = int(2 ** np.ceil(np.log2(2 * n_cols)))
    padded = np.zeros((n_angles, pad))
    start = (pad - n_cols) // 2
    padded[:, start:start + n_cols] = sino

    filt = ram_lak_filter(pad, window=window)
    filtered = np.fft.irfft(np.fft.rfft(padded, axis=1) * filt[None, :],
                            n=pad, axis=1)[:, start:start + n_cols]

    # Backprojection with linear interpolation (inverse mapping per angle).
    yy, xx = np.mgrid[0:size, 0:size]
    xc = (xx - (size - 1) / 2.0) / (n_cols / 2.0) * (n_cols - 1) / 2.0
    yc = (yy - (size - 1) / 2.0) / (n_cols / 2.0) * (n_cols - 1) / 2.0

    out = np.zeros((size, size))
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    for i, ang in enumerate(angles):
        c, s = np.cos(ang), np.sin(ang)
        t = xc * c + yc * s + (n_cols - 1) / 2.0
        t0 = np.clip(np.floor(t).astype(np.int64), 0, n_cols - 2)
        w = np.clip(t - t0, 0.0, 1.0)
        row = filtered[i]
        out += row[t0] * (1 - w) + row[t0 + 1] * w
    out *= np.pi / (2.0 * n_angles)
    return out
