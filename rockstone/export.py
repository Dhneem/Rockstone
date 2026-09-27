"""Export the currently displayed slice to PNG, SVG, PDF or CSV.

- **PNG / PDF**: the image exactly as rendered (color system, brightness
  and contrast window applied), tagged with the physical pixel size as
  DPI metadata so 1:1 prints come out at true scale.
- **SVG**: a *true vector* rendering — each pixel is a rectangle, runs
  of equal value in a row are merged into single rects. Faithful to the
  data at any magnification; sized in millimeters using the stored
  physical dimensions.
- **CSV**: the raw (float) values of the current slice, one row per
  image row — for spreadsheets or further scripted analysis.
"""

from __future__ import annotations

import csv
import xml.sax.saxutils as saxutils
from pathlib import Path

import numpy as np

__all__ = [
    "EXPORT_FORMATS",
    "export_slice",
    "physical_pixel_size",
]

#: GUI-visible format list: (label, extension)
EXPORT_FORMATS = [
    ("PNG image (*.png)", ".png"),
    ("SVG vector image (*.svg)", ".svg"),
    ("PDF document (*.pdf)", ".pdf"),
    ("CSV values (*.csv)", ".csv"),
]


def physical_pixel_size(meta: dict | None) -> float | None:
    """Pixel size in mm from volume metadata (mean of x/y, None if unset).

    A DPI scale only makes sense when both axes share one size; a
    square-pixel image is the normal case for the slice views here.
    """
    if not meta:
        return None
    px = meta.get("pixel_size")
    if isinstance(px, (int, float)):
        return float(px) or None
    if isinstance(px, (list, tuple)) and len(px) >= 2:
        try:
            x, y = float(px[0]), float(px[1])
        except (TypeError, ValueError):
            return None
        if x > 0 and y > 0 and np.isclose(x, y, rtol=0.05):
            return (x + y) / 2.0
        return None  # non-square pixels: no meaningful DPI
    return None


def export_slice(
    data: np.ndarray | None,
    u8: np.ndarray,
    path: str | Path,
    meta: dict | None = None,
    slice_index: int | None = None,
    colormap: str | None = None,
) -> Path:
    """Export one slice to ``path`` (format chosen by the extension).

    Parameters
    ----------
    data:
        The full stack; ``data[slice_index]`` provides the raw values
        for the CSV payload and the SVG metadata rows (``None`` exports
        only the rendered image formats).
    u8:
        The rendered uint8 image for the same slice (grayscale or RGB).
    path:
        Destination file; ``.png`` / ``.svg`` / ``.pdf`` / ``.csv``.
    meta:
        Physical-dimensions metadata (``pixel_size`` in mm) used for
        DPI tagging and the SVG size in millimeters.
    slice_index:
        Index of the exported slice in ``data`` (metadata rows only).
    colormap:
        Name of the active color system (recorded in SVG metadata).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".png":
        _export_png(u8, p, meta)
    elif suffix == ".pdf":
        _export_pdf(u8, p, meta)
    elif suffix == ".svg":
        _export_svg(u8, p, meta, data, slice_index, colormap)
    elif suffix == ".csv":
        _export_csv(data, slice_index, p)
    else:
        raise ValueError(f"unsupported export format: {p.suffix!r}")
    return p


# ---------------------------------------------------------------- PNG/PDF --

def _dpi(meta: dict | None) -> tuple[float, float] | None:
    size = physical_pixel_size(meta)
    if size is None:
        return None
    dpi = 25.4 / size
    return (dpi, dpi)


def _export_png(u8: np.ndarray, p: Path, meta: dict | None) -> None:
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(u8))
    img.save(p, format="PNG", dpi=_dpi(meta))


def _export_pdf(u8: np.ndarray, p: Path, meta: dict | None) -> None:
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(u8))
    img.save(p, format="PDF", resolution=(_dpi(meta) or (150.0, 150.0))[0])


# -------------------------------------------------------------------- SVG --

_GRAY_HEX = np.char.add("#", np.char.add(np.char.add(
    np.char.mod("%02x", np.arange(256)),
    np.char.mod("%02x", np.arange(256))),
    np.char.mod("%02x", np.arange(256))))


def _row_runs(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start/end column indices of runs of equal values in ``row``."""
    change = np.flatnonzero(np.diff(row, axis=0)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [row.shape[0]]))
    return starts, ends


def _export_svg(u8: np.ndarray, p: Path, meta: dict | None,
                data: np.ndarray | None, slice_index: int | None,
                colormap: str | None) -> None:
    """Write a true-vector SVG: one shape per run of equal pixels.

    Grayscale images emit one merged path per gray level (color becomes
    position); color images emit one rect per run. The canvas is sized
    in millimeters when physical dimensions are known.
    """
    h, w = u8.shape[:2]
    size_mm = physical_pixel_size(meta)
    width_attr = f"{w * size_mm:.4g}mm" if size_mm else str(w)
    height_attr = f"{h * size_mm:.4g}mm" if size_mm else str(h)

    slice_bit = (f" (slice {slice_index + 1})"
                 if slice_index is not None else "")
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_attr}" height="{height_attr}" '
        f'viewBox="0 0 {w} {h}" shape-rendering="crispEdges">',
        f"<title>Rockstone slice export{slice_bit}</title>",
    ]
    if colormap:
        out.append("<desc>color system: "
                   f"{saxutils.escape(colormap)}</desc>")

    if u8.ndim == 2:
        by_val: dict[int, list[str]] = {}
        for y in range(h):
            starts, ends = _row_runs(u8[y])
            for s, e in zip(starts, ends):
                by_val.setdefault(int(u8[y, s]), []).append(
                    f"M{s},{y}h{e - s}v1h-{e - s}z")
        out.append("<g>")
        for val in sorted(by_val):
            out.append(f'<path fill="{_GRAY_HEX[val]}" d="'
                       + " ".join(by_val[val]) + '"/>')
        out.append("</g>")
    else:
        out.append("<g>")
        rgb = np.ascontiguousarray(u8[..., :3])
        for y in range(h):
            row = rgb[y]
            change = np.flatnonzero(np.diff(row, axis=0).any(axis=1)) + 1
            starts = np.concatenate(([0], change))
            ends = np.concatenate((change, [row.shape[0]]))
            for s, e in zip(starts, ends):
                r, g, b = (int(c) for c in row[s])
                out.append(
                    f'<rect x="{s}" y="{y}" width="{e - s}" height="1" '
                    f'fill="#{r:02x}{g:02x}{b:02x}"/>')
        out.append("</g>")

    if data is not None and slice_index is not None \
            and 0 <= slice_index < len(data):
        plane = np.asarray(data[slice_index], dtype=np.float64)
        finite = plane[np.isfinite(plane)]
        if finite.size:
            out.append(f"<desc>raw values: min {finite.min():.6g}, "
                       f"max {finite.max():.6g}</desc>")
    out.append("</svg>")
    p.write_text("\n".join(out), encoding="utf-8")


# -------------------------------------------------------------------- CSV --

def _export_csv(data: np.ndarray | None, slice_index: int | None,
                p: Path) -> None:
    if data is None or slice_index is None:
        raise ValueError("CSV export needs the raw data and a slice index")
    if not (0 <= slice_index < len(data)):
        raise IndexError(f"slice index {slice_index} out of range")
    plane = np.asarray(data[slice_index], dtype=np.float64)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "value"])
        n_rows, n_cols = plane.shape
        for y in range(n_rows):
            row = plane[y]
            vals = ["" if not np.isfinite(v) else repr(float(v))
                    for v in row]
            writer.writerows(
                [y, x, vals[x]] for x in range(n_cols))
