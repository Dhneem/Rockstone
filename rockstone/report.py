"""Generate a self-contained report for the loaded dataset.

The report summarizes *what the data is* (shape, kind, value range,
physical dimensions), *what was measured* (cupping, air level,
background spread) and *what was done* (the modification history of
this session). The HTML variant embeds a PNG snapshot of the current
slice and its intensity histogram — no external files, easy to e-mail
or drop next to the data.
"""

from __future__ import annotations

import datetime
import io
import sys
from pathlib import Path

import numpy as np

__all__ = ["ReportData", "collect_report_data", "generate_report",
           "render_slice_u8"]


def _png_bytes(img: np.ndarray) -> bytes:
    """Serialize a uint8 image as PNG bytes (for PDF/DOCX embedding)."""
    import io as _io

    from PIL import Image

    buf = _io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG")
    return buf.getvalue()


def render_slice_u8(plane: np.ndarray) -> np.ndarray:
    """Headless grayscale rendering of a plane (1-99.5 percentile
    stretch) for reports written without the GUI."""
    plane = np.asarray(plane, dtype=np.float64)
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros(plane.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    out = (plane - lo) / (hi - lo)
    out = np.nan_to_num(out, nan=0.0)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def _percentile_level(plane: np.ndarray, percentile: float) -> float:
    finite = np.asarray(plane, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, percentile))


def collect_report_data(
    data: np.ndarray | None,
    *,
    kind: str | None = None,
    source: str | None = None,
    meta: dict | None = None,
    slice_index: int | None = None,
    u8: np.ndarray | None = None,
    colormap: str | None = None,
    history: list[tuple[str, str]] | None = None,
    processed: bool = False,
    compare_result=None,
) -> "ReportData":
    """Gather everything the report needs from the app state.

    ``history`` is a list of ``(label, action)`` tuples — ``action`` is
    ``"did"`` for modifications, ``"undid"`` / ``"re-applied"`` for
    history steps (as recorded by the GUI's undo stack).
    """
    import scipy  # for the environment section

    try:
        import tifffile
        tifffile_version = tifffile.__version__
    except Exception:  # noqa: BLE001 - optional for the report
        tifffile_version = "?"
    try:
        import PIL
        pil_version = PIL.__version__
    except Exception:  # noqa: BLE001
        pil_version = "?"

    rd = ReportData()
    rd.generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    rd.python = sys.version.split()[0]
    rd.versions = f"numpy {np.__version__}, scipy {scipy.__version__}, tifffile {tifffile_version}, Pillow {pil_version}"
    rd.source = source or "-"
    rd.history = list(history or [])
    rd.processed = processed
    rd.slice_index = slice_index
    rd.colormap = colormap
    rd.compare_result = compare_result

    if data is None:
        return rd

    data = np.asarray(data)
    rd.shape = tuple(int(n) for n in data.shape)
    rd.dtype = str(data.dtype)
    finite = data[np.isfinite(data)]
    rd.value_min = float(finite.min()) if finite.size else float("nan")
    rd.value_max = float(finite.max()) if finite.size else float("nan")

    from .metrics import cupping_index

    if kind is None:
        nz, ny, nx = data.shape
        kind = "projections" if nz > 1.5 * max(ny, nx) else "volume"
    rd.kind = kind
    if kind == "volume" and data.shape[0] > 0:
        rd.cupping = cupping_index(data[data.shape[0] // 2])

    px = (meta or {}).get("pixel_size")
    sp = (meta or {}).get("slice_spacing")
    if isinstance(px, (int, float)):
        px = [px, px]
    if px and len(px) >= 2 and sp:
        rd.voxel = f"{px[0]:g} x {px[1]:g} x {sp:g} mm"
        rd.voxel_volume_mm3 = float(px[0]) * float(px[1]) * float(sp)
    elif px and len(px) >= 2:
        rd.voxel = f"{px[0]:g} x {px[1]:g} mm"
    elif sp:
        rd.voxel = f"{sp:g} mm / slice"
    if rd.voxel_volume_mm3 is not None and rd.shape:
        rd.stack_extent_mm = (
            rd.shape[2] * float(px[0]),
            rd.shape[1] * float(px[1]),
            rd.shape[0] * float(sp),
        )

    # air level / background spread of the current state
    rd.air_level, rd.bg_spread = _air_metrics(data)
    return rd


class ReportData:
    """Plain container for the report payload (see
    :func:`collect_report_data`)."""

    def __init__(self) -> None:
        self.generated: str = ""
        self.python: str = ""
        self.versions: str = ""
        self.source: str = "-"
        self.shape: tuple | None = None
        self.dtype: str = ""
        self.kind: str | None = None
        self.value_min: float = float("nan")
        self.value_max: float = float("nan")
        self.cupping: float | None = None
        self.voxel: str | None = None
        self.voxel_volume_mm3: float | None = None
        self.stack_extent_mm: tuple | None = None
        self.slice_index: int | None = None
        self.colormap: str | None = None
        self.processed: bool = False
        self.history: list[tuple[str, str]] = []
        self.compare_result = None
        self.air_level: float | None = None
        self.bg_spread: float | None = None


def _air_metrics(data: np.ndarray) -> tuple[float | None, float | None]:
    """Air level and background spread (median across slices)."""
    from .background import _percentile_level

    if data is None or not data.size:
        return None, None
    airs, spreads = [], []
    for plane in data[: min(len(data), 16)]:
        air = _percentile_level(plane, 1.0)
        hi = _percentile_level(plane, 99.5)
        if hi <= air:
            continue
        bg = plane[plane < air + 0.25 * (hi - air)]
        bg = bg[np.isfinite(bg)]
        if bg.size:
            airs.append(air)
            spreads.append(float(np.std(bg)))
    if not airs:
        return None, None
    return float(np.median(airs)), float(np.median(spreads))


def generate_report(
    rd: ReportData,
    path: str | Path,
    data: np.ndarray | None = None,
    u8: np.ndarray | None = None,
) -> Path:
    """Write the report to ``path``.

    Formats: ``.html``/``.htm`` (self-contained, embedded images),
    ``.pdf``, ``.docx`` (both with the slice snapshot and histogram
    when ``u8`` is given) or ``.txt``.

    ``data`` and ``u8`` (the rendered slice) enable the metrics and the
    embedded images.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".html", ".htm"}:
        _write_html(rd, p, data, u8)
    elif suffix == ".pdf":
        _write_pdf(rd, p, data, u8)
    elif suffix == ".docx":
        _write_docx(rd, p, data, u8)
    elif suffix == ".txt":
        _write_txt(rd, p, data)
    else:
        raise ValueError(
            f"unsupported report format: {p.suffix!r} "
            f"(use .html, .pdf, .docx or .txt)")
    return p


def _shared_rows(rd: ReportData) -> list[tuple[str, str]]:
    """Dataset + metric rows shared by the PDF/DOCX writers."""
    rows = [
        ("shape", f"{_fmt(rd.shape)} (slices x rows x columns)"),
        ("data type", _fmt(rd.dtype)),
        ("kind", _fmt(rd.kind)),
        ("value range", f"{_fmt(rd.value_min, '{:.6g}')} .. "
                        f"{_fmt(rd.value_max, '{:.6g}')}"),
        ("voxel size", _fmt(rd.voxel)),
    ]
    if rd.voxel_volume_mm3 is not None:
        rows.append(("voxel volume", f"{rd.voxel_volume_mm3:.6g} mm^3"))
    if rd.stack_extent_mm is not None:
        x, y, z = rd.stack_extent_mm
        rows.append(("sample extent", f"{x:.6g} x {y:.6g} x {z:.6g} mm"))
    rows.append(("cupping index", _fmt(rd.cupping, '{:.3f}')))
    rows.append(("air level", _fmt(rd.air_level, '{:.6g}')))
    rows.append(("background spread", _fmt(rd.bg_spread, '{:.6g}')))
    return rows


# ------------------------------------------------------------------ HTML --

_HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Rockstone report</title>
<style>
body { font-family: Segoe UI, Arial, sans-serif; margin: 24px auto;
       max-width: 860px; color: #222; line-height: 1.45; }
h1 { font-size: 1.5em; border-bottom: 2px solid #4a7; padding-bottom: 6px; }
h2 { font-size: 1.1em; margin-top: 1.4em; color: #363; }
table { border-collapse: collapse; margin: 8px 0; }
td { padding: 3px 14px 3px 0; vertical-align: top; }
td.k { color: #666; white-space: nowrap; }
img { border: 1px solid #bbb; max-width: 100%; }
.hist { image-rendering: pixelated; }
.footer { color: #888; font-size: 0.85em; margin-top: 2em;
           border-top: 1px solid #ddd; padding-top: 8px; }
.warn { color: #a33; }
</style></head><body>
"""


def _fmt(v, fmt: str = "{}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return fmt.format(v)


def _write_html(rd: ReportData, p: Path, data: np.ndarray | None,
                u8: np.ndarray | None) -> None:
    out = [_HTML_HEAD]
    out.append("<h1>Rockstone report</h1>")
    out.append(f"<p>Generated {rd.generated} &nbsp;·&nbsp; "
               f"source: {rd.source}</p>")

    out.append("<h2>Dataset</h2><table>")
    out.append(f"<tr><td class='k'>shape</td><td>{_fmt(rd.shape)} "
               f"(slices × rows × columns)</td></tr>")
    out.append(f"<tr><td class='k'>data type</td><td>{_fmt(rd.dtype)}"
               "</td></tr>")
    out.append(f"<tr><td class='k'>kind</td><td>{_fmt(rd.kind)}</td></tr>")
    out.append(f"<tr><td class='k'>value range</td><td>"
               f"{_fmt(rd.value_min, '{:.6g}')} … "
               f"{_fmt(rd.value_max, '{:.6g}')}</td></tr>")
    out.append(f"<tr><td class='k'>voxel size</td><td>{_fmt(rd.voxel)}"
               "</td></tr>")
    if rd.voxel_volume_mm3 is not None:
        out.append(f"<tr><td class='k'>voxel volume</td><td>"
                   f"{rd.voxel_volume_mm3:.6g} mm³</td></tr>")
    if rd.stack_extent_mm is not None:
        x, y, z = rd.stack_extent_mm
        out.append(f"<tr><td class='k'>sample extent</td><td>"
                   f"{x:.6g} × {y:.6g} × {z:.6g} mm (x, y, z)</td></tr>")
    out.append("</table>")

    out.append("<h2>Metrics</h2><table>")
    out.append(f"<tr><td class='k'>cupping index</td><td>"
               f"{_fmt(rd.cupping, '{:.3f}')} (middle slice)</td></tr>")
    out.append(f"<tr><td class='k'>air level</td><td>"
               f"{_fmt(rd.air_level, '{:.6g}')}</td></tr>")
    out.append(f"<tr><td class='k'>background spread</td><td>"
               f"{_fmt(rd.bg_spread, '{:.6g}')} (std)</td></tr>")
    out.append("</table>")
    if rd.cupping is not None and rd.cupping > 0.15:
        out.append("<p class='warn'>Cupping above 0.15 — consider "
                   "'Remove beam hardening'.</p>")

    if rd.compare_result is not None:
        out.append("<h2>Comparison</h2><table>")
        for line in str(rd.compare_result.summary()).splitlines():
            out.append(f"<tr><td>{line}</td></tr>")
        out.append("</table>")

    out.append("<h2>Processing history</h2>")
    if rd.history:
        out.append("<ol>")
        for label, action in rd.history:
            out.append(f"<li>{action} <b>{label}</b></li>")
        out.append("</ol>")
    else:
        out.append("<p>No modifications in this session.</p>")

    if u8 is not None:
        out.append(f"<h2>Current slice"
                   f"{f' (#{rd.slice_index + 1})' if rd.slice_index is not None else ''}"
                   "</h2>")
        out.append("<img src=\"data:image/png;base64,{img}\" "
                   "alt=\"current slice\">")
        out.append("<p class='hist'><img "
                   "src=\"data:image/png;base64,{hist}\" "
                   "alt=\"histogram\" class=\"hist\"></p>")
        if rd.colormap:
            out.append(f"<p>color system: {rd.colormap}</p>")
    if rd.processed:
        out.append("<p class='warn'>The dataset has been modified in "
                   "this session and may have unsaved changes.</p>")

    out.append(f"<div class='footer'>Rockstone · Python {rd.python} · "
               f"{rd.versions}</div>")
    out.append("</body></html>")
    html = "\n".join(out)
    if u8 is not None:
        html = html.replace("{img}", _png_b64(_margin_image(u8)))
        html = html.replace("{hist}", _png_b64(_histogram_image(u8)))
    p.write_text(html, encoding="utf-8")


def _margin_image(u8: np.ndarray, frac: float = 0.04) -> np.ndarray:
    """Pad the rendered slice with a dark frame so bright edges show."""
    m = max(2, int(round(max(u8.shape[:2]) * frac)))
    h, w = u8.shape[:2]
    if u8.ndim == 3:
        out = np.zeros((h + 2 * m, w + 2 * m, u8.shape[2]), np.uint8)
        out[m:m + h, m:m + w] = u8
    else:
        out = np.zeros((h + 2 * m, w + 2 * m), np.uint8)
        out[m:m + h, m:m + w] = u8
    return out


def _histogram_image(u8: np.ndarray, width: int = 512, height: int = 128,
                     bins: int = 64) -> np.ndarray:
    """Render the intensity histogram of the slice as a uint8 image."""
    gray = u8[..., 0] if u8.ndim == 3 else u8
    counts, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 255))
    peak = max(1, int(counts.max()))
    img = np.full((height, width), 255, np.uint8)
    bar_w = width / bins
    for i, c in enumerate(counts):
        bh = int(round(c / peak * (height - 6)))
        if bh <= 0:
            continue
        x0 = int(round(i * bar_w))
        x1 = max(x0 + 1, int(round((i + 1) * bar_w)) - 1)
        img[height - bh:, x0:x1] = 70
    return img


def _png_b64(img: np.ndarray) -> str:
    import base64

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------- PDF --

def _write_pdf(rd: ReportData, p: Path, data: np.ndarray | None,
               u8: np.ndarray | None) -> None:
    """Lay the report out as a one-or-two page PDF (ReportLab)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, Paragraph,
                                    SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    accent = colors.HexColor("#2e6b4f")
    head = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=17,
                          leading=21, textColor=accent, spaceAfter=2)
    sub = ParagraphStyle("sub", fontName="Helvetica", fontSize=9,
                         leading=12, textColor=colors.HexColor("#555555"),
                         spaceAfter=10)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12,
                        leading=15, textColor=accent, spaceBefore=12,
                        spaceAfter=4)
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5,
                          leading=11)
    cellk = ParagraphStyle("cellk", parent=cell,
                           textColor=colors.HexColor("#666666"))
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9,
                          leading=12)
    warn = ParagraphStyle("warn", parent=body, textColor=colors.HexColor(
        "#a33333"))

    story = [Paragraph("Rockstone report", head),
             Paragraph(f"Generated {rd.generated} · source: {rd.source}",
                       sub)]

    def table(rows_):
        t = Table(
            [[Paragraph(k, cellk), Paragraph(v, cell)] for k, v in rows_],
            colWidths=[34 * mm, None], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        return t

    story.append(Paragraph("Dataset & metrics", h2))
    story.append(table(_shared_rows(rd)))

    if rd.compare_result is not None:
        story.append(Paragraph("Comparison", h2))
        story.append(table(
            [(line.split(":", 1)[0].strip(),
              line.split(":", 1)[1].strip() if ":" in line else "")
             for line in str(rd.compare_result.summary()).splitlines()
             if ":" in line]))

    story.append(Paragraph("Processing history", h2))
    if rd.history:
        items = [f"{action} <b>{label}</b>" for label, action in rd.history]
        story.append(Table(
            [[Paragraph(f"{i + 1}.", cell), Paragraph(it, cell)]
             for i, it in enumerate(items)],
            colWidths=[10 * mm, None], hAlign="LEFT"))
    else:
        story.append(Paragraph("No modifications in this session.", body))

    if rd.cupping is not None and rd.cupping > 0.15:
        story.append(Spacer(1, 4))
        story.append(Paragraph("Cupping above 0.15 — consider 'Remove "
                               "beam hardening'.", warn))
    if rd.processed:
        story.append(Spacer(1, 4))
        story.append(Paragraph("The dataset has been modified in this "
                               "session and may have unsaved changes.",
                               warn))

    def _fit(img: np.ndarray, max_w: float, max_h: float) -> tuple[float, float]:
        """Scale (w, h) to fit the box, preserving aspect."""
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h)
        return w * scale, h * scale

    if u8 is not None:
        story.append(Paragraph(
            f"Current slice"
            f"{f' (#{rd.slice_index + 1})' if rd.slice_index is not None else ''}",
            h2))
        slice_img = _margin_image(u8)
        w, h = _fit(slice_img, 120 * mm, 150 * mm)
        story.append(RLImage(io.BytesIO(_png_bytes(slice_img)),
                             width=w, height=h))
        hist_img = _histogram_image(u8)
        story.append(Spacer(1, 6))
        hw, hh = _fit(hist_img, 120 * mm, 40 * mm)
        story.append(RLImage(io.BytesIO(_png_bytes(hist_img)),
                             width=hw, height=hh))
        if rd.colormap:
            story.append(Spacer(1, 3))
            story.append(Paragraph(f"color system: {rd.colormap}", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Rockstone · Python {rd.python} · "
                           f"{rd.versions}", sub))

    doc = SimpleDocTemplate(
        str(p), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Rockstone report", author="Rockstone")
    doc.build(story)


# ------------------------------------------------------------------ DOCX --

def _write_docx(rd: ReportData, p: Path, data: np.ndarray | None,
                u8: np.ndarray | None) -> None:
    """Write the report as a Word document (python-docx)."""
    import docx
    from docx.shared import Inches, Pt, RGBColor

    accent = RGBColor(0x2E, 0x6B, 0x4F)
    grey = RGBColor(0x66, 0x66, 0x66)
    warn_color = RGBColor(0xA3, 0x33, 0x33)

    doc = docx.Document()
    doc.core_properties.title = "Rockstone report"
    doc.core_properties.author = "Rockstone"

    title = doc.add_heading("Rockstone report", level=0)
    for run in title.runs:
        run.font.color.rgb = accent
    meta_p = doc.add_paragraph()
    r = meta_p.add_run(f"Generated {rd.generated} · source: {rd.source}")
    r.font.size = Pt(9)
    r.font.color.rgb = grey

    doc.add_heading("Dataset & metrics", level=1)
    table = doc.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"
    for k, v in _shared_rows(rd):
        row = table.add_row().cells
        row[0].text = k
        row[1].text = v

    if rd.compare_result is not None:
        doc.add_heading("Comparison", level=1)
        for line in str(rd.compare_result.summary()).splitlines():
            doc.add_paragraph(line)

    doc.add_heading("Processing history", level=1)
    if rd.history:
        for label, action in rd.history:
            doc.add_paragraph(f"{action} {label}", style="List Number")
    else:
        doc.add_paragraph("No modifications in this session.")

    if rd.cupping is not None and rd.cupping > 0.15:
        w = doc.add_paragraph("Cupping above 0.15 — consider 'Remove "
                              "beam hardening'.")
        w.runs[0].font.color.rgb = warn_color
    if rd.processed:
        w = doc.add_paragraph("The dataset has been modified in this "
                              "session and may have unsaved changes.")
        w.runs[0].font.color.rgb = warn_color

    if u8 is not None:
        doc.add_heading(
            f"Current slice"
            f"{f' (#{rd.slice_index + 1})' if rd.slice_index is not None else ''}",
            level=1)
        doc.add_picture(io.BytesIO(_png_bytes(_margin_image(u8))),
                        width=Inches(4.3))
        doc.add_picture(io.BytesIO(_png_bytes(_histogram_image(u8))),
                        width=Inches(4.3))
        if rd.colormap:
            cap = doc.add_paragraph(f"color system: {rd.colormap}")
            cap.runs[0].font.size = Pt(9)
            cap.runs[0].font.color.rgb = grey

    foot = doc.add_paragraph(f"Rockstone · Python {rd.python} · "
                             f"{rd.versions}")
    foot.runs[0].font.size = Pt(8)
    foot.runs[0].font.color.rgb = grey
    doc.save(p)


# ------------------------------------------------------------------- TXT --

def _write_txt(rd: ReportData, p: Path, data: np.ndarray | None) -> None:
    L = []
    L.append("Rockstone report")
    L.append("=" * 60)
    L.append(f"generated : {rd.generated}")
    L.append(f"source    : {rd.source}")
    L.append("")
    L.append("Dataset")
    L.append("-" * 60)
    L.append(f"shape     : {_fmt(rd.shape)}  (slices x rows x columns)")
    L.append(f"dtype     : {_fmt(rd.dtype)}")
    L.append(f"kind      : {_fmt(rd.kind)}")
    L.append(f"range     : {_fmt(rd.value_min, '{:.6g}')} .. "
             f"{_fmt(rd.value_max, '{:.6g}')}")
    L.append(f"voxel     : {_fmt(rd.voxel)}")
    if rd.voxel_volume_mm3 is not None:
        L.append(f"voxel vol : {rd.voxel_volume_mm3:.6g} mm^3")
    if rd.stack_extent_mm is not None:
        x, y, z = rd.stack_extent_mm
        L.append(f"extent    : {x:.6g} x {y:.6g} x {z:.6g} mm")
    L.append("")
    L.append("Metrics")
    L.append("-" * 60)
    L.append(f"cupping   : {_fmt(rd.cupping, '{:.3f}')} (middle slice)")
    L.append(f"air level : {_fmt(rd.air_level, '{:.6g}')}")
    L.append(f"bg spread : {_fmt(rd.bg_spread, '{:.6g}')} (std)")
    if rd.compare_result is not None:
        L.append("")
        L.append("Comparison")
        L.append("-" * 60)
        L.append(str(rd.compare_result.summary()))
    L.append("")
    L.append("Processing history")
    L.append("-" * 60)
    if rd.history:
        for label, action in rd.history:
            L.append(f"- {action} {label}")
    else:
        L.append("(no modifications in this session)")
    if rd.processed:
        L.append("")
        L.append("NOTE: dataset modified in this session; may have "
                 "unsaved changes.")
    L.append("")
    L.append(f"Rockstone · Python {rd.python} · {rd.versions}")
    p.write_text("\n".join(L), encoding="utf-8")
