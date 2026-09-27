"""Tests for report generation: library (HTML/TXT), CLI and GUI."""

from __future__ import annotations

import base64
import time

import numpy as np
import pytest

from rockstone.compare import compare_volumes
from rockstone.demo import make_rock_stack
from rockstone.io import save_volume
from rockstone.report import (ReportData, collect_report_data,
                              generate_report, render_slice_u8)


@pytest.fixture()
def dataset():
    vol, _ = make_rock_stack(n_slices=6, size=48, seed=0, cupping=0.3)
    u8 = np.tile(np.arange(0, 256, 16, dtype=np.uint8), (48, 1))
    meta = {"pixel_size": [0.05, 0.05], "slice_spacing": 0.05}
    return vol, u8, meta


def _collect(vol, u8=None, meta=None, **kw):
    return collect_report_data(
        vol, kind="volume", source="vol.tif", meta=meta,
        slice_index=2 if u8 is not None else None,
        u8=u8, colormap="Gray" if u8 is not None else None,
        history=[("align slices", "did"), ("crop", "did")],
        processed=True, **kw)


# ------------------------------------------------------------ library --

def test_html_report_contents(tmp_path, dataset):
    vol, u8, meta = dataset
    rd = _collect(vol, u8, meta)
    p = generate_report(rd, tmp_path / "r.html", data=vol, u8=u8)
    html = p.read_text(encoding="utf-8")
    assert "<h1>Rockstone report</h1>" in html
    assert "(6, 48, 48)" in html and "slices × rows × columns" in html
    assert "float64" in html and "volume" in html
    assert "0.05 x 0.05 x 0.05 mm" in html
    assert "2.4 × 2.4 × 0.3 mm" in html          # sample extent
    assert "0.000125 mm³" in html                # voxel volume
    assert "cupping index" in html
    assert "air level" in html and "background spread" in html
    assert "<li>did <b>align slices</b></li>" in html
    assert "<li>did <b>crop</b></li>" in html
    assert "Current slice" in html and "#3" in html
    assert html.count("data:image/png;base64,") == 2  # slice + histogram
    # the embedded payload really is a PNG
    b64 = html.split("base64,")[1].split('"')[0]
    assert base64.b64decode(b64)[:4] == b"\x89PNG"
    assert "Rockstone · Python" in html          # footer


def test_html_report_warnings(tmp_path, dataset):
    vol, u8, meta = dataset
    rd = _collect(vol, u8, meta)  # demo data has cupping ~0.26 > 0.15
    p = generate_report(rd, tmp_path / "r.html", data=vol, u8=u8)
    html = p.read_text(encoding="utf-8")
    assert "Remove beam hardening" in html       # cupping hint
    assert "unsaved changes" in html             # processed=True


def test_html_report_compare_section(tmp_path, dataset):
    vol, u8, meta = dataset
    other = vol * 1.1
    res = compare_volumes(vol, other)
    rd = _collect(vol, u8, meta, compare_result=res)
    p = generate_report(rd, tmp_path / "c.html", data=vol, u8=u8)
    html = p.read_text(encoding="utf-8")
    assert "Comparison" in html
    assert "max" in html.lower() and "rmse" in html.lower()


def test_html_report_minimal(tmp_path):
    rd = ReportData()
    rd.generated = "now"
    p = generate_report(rd, tmp_path / "m.html")
    html = p.read_text(encoding="utf-8")
    assert "—" in html          # placeholder for missing values
    assert "No modifications" in html
    assert "data:image/png" not in html          # no images requested


def test_txt_report_contents(tmp_path, dataset):
    vol, _u8, meta = dataset
    rd = _collect(vol, None, meta)
    p = generate_report(rd, tmp_path / "r.txt", data=vol, u8=None)
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "Rockstone report"
    assert "shape     : (6, 48, 48)" in text
    assert "voxel     : 0.05 x 0.05 x 0.05 mm" in text
    assert "- did align slices" in text
    assert "may have" in text and "unsaved changes." in text


def test_report_rejects_unknown_format(tmp_path, dataset):
    vol, _u8, _meta = dataset
    with pytest.raises(ValueError, match="use .html, .pdf, .docx or .txt"):
        generate_report(ReportData(), tmp_path / "r.pptx", data=vol)


def test_render_slice_u8_stretch():
    plane = np.tile(np.linspace(0.2, 0.8, 64), (16, 1))
    u8 = render_slice_u8(plane)
    assert u8.dtype == np.uint8
    assert u8.min() == 0 and u8.max() == 255
    nan_plane = np.full((8, 8), np.nan)
    assert (render_slice_u8(nan_plane) == 0).all()


# ------------------------------------------------------- PDF and DOCX --

def test_pdf_report_contents(tmp_path, dataset):
    vol, u8, meta = dataset
    rd = _collect(vol, u8, meta)
    p = generate_report(rd, tmp_path / "r.pdf", data=vol, u8=u8)
    assert p.read_bytes()[:8] == b"%PDF-1.4"
    from pypdf import PdfReader

    reader = PdfReader(str(p))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Rockstone report" in text
    assert "cupping index" in text and "air level" in text
    assert "align slices" in text and "crop" in text  # history
    assert "unsaved changes" in text                   # processed warning
    assert "Remove beam hardening" in text             # cupping hint


def test_pdf_report_minimal(tmp_path):
    p = generate_report(ReportData(), tmp_path / "m.pdf")
    assert p.exists()
    from pypdf import PdfReader

    assert "No modifications" in "\n".join(
        page.extract_text() for page in PdfReader(str(p)).pages)


def test_pdf_report_portrait_image_fits(tmp_path, dataset):
    # tall aspect ratio must not overflow the page (LayoutError guard)
    vol, _u8, meta = dataset
    tall_u8 = np.zeros((200, 40), np.uint8)
    tall_u8[10:190, 10:30] = 200
    rd = _collect(vol, tall_u8, meta)
    p = generate_report(rd, tmp_path / "tall.pdf", data=vol, u8=tall_u8)
    assert p.exists()


def test_docx_report_contents(tmp_path, dataset):
    import zipfile

    vol, u8, meta = dataset
    rd = _collect(vol, u8, meta)
    p = generate_report(rd, tmp_path / "r.docx", data=vol, u8=u8)
    assert p.read_bytes()[:2] == b"PK"  # OOXML zip
    z = zipfile.ZipFile(p)
    assert "word/document.xml" in z.namelist()
    # slice snapshot + histogram embedded
    assert sum(1 for n in z.namelist()
               if n.startswith("word/media/")) == 2
    doc = z.read("word/document.xml").decode("utf-8")
    assert "Rockstone report" in doc
    assert "cupping index" in doc


def test_docx_report_minimal(tmp_path):
    import zipfile

    p = generate_report(ReportData(), tmp_path / "m.docx")
    z = zipfile.ZipFile(p)
    doc = z.read("word/document.xml").decode("utf-8")
    assert "No modifications" in doc
    assert "word/media" not in doc      # no images requested


# ----------------------------------------------------------------- CLI --

def test_cli_report_html_and_txt(tmp_path):
    from rockstone.cli import main

    src = tmp_path / "in.tif"
    save_volume(make_rock_stack(n_slices=4, size=32, seed=0,
                                cupping=0.3)[0], src)
    out_h = tmp_path / "rep.html"
    out_t = tmp_path / "rep.txt"
    assert main(["report", str(src), "-o", str(out_h)]) == 0
    assert main(["report", str(src), "--txt", "-o", str(out_t)]) == 0
    html = out_h.read_text(encoding="utf-8")
    assert "<h1>Rockstone report</h1>" in html
    assert html.count("data:image/png;base64,") == 2
    assert "Rockstone report" in out_t.read_text(encoding="utf-8")


def test_cli_report_default_name_and_no_image(tmp_path):
    from rockstone.cli import main

    src = tmp_path / "scan.tif"
    save_volume(make_rock_stack(n_slices=3, size=24, seed=0,
                                cupping=0.3)[0], src)
    assert main(["report", str(src), "--no-image"]) == 0
    default = tmp_path / "scan_report.html"
    assert default.exists()
    html = default.read_text(encoding="utf-8")
    assert "data:image/png" not in html  # --no-image skipped the snapshot


def test_cli_report_pdf_and_docx(tmp_path):
    import zipfile

    from rockstone.cli import main

    src = tmp_path / "scan.tif"
    save_volume(make_rock_stack(n_slices=3, size=24, seed=0,
                                cupping=0.3)[0], src)
    assert main(["report", str(src), "--pdf"]) == 0
    pdf = tmp_path / "scan_report.pdf"
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"
    assert main(["report", str(src), "--docx"]) == 0
    docx = tmp_path / "scan_report.docx"
    assert docx.exists() and zipfile.ZipFile(docx).namelist()
    # explicit -o overrides the extension guessing
    custom = tmp_path / "custom.pdf"
    assert main(["report", str(src), "--pdf", "-o", str(custom),
                 "--no-image"]) == 0
    assert custom.exists()


# ----------------------------------------------------------------- GUI --

def _pump(root, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _find(widget, cls):
    out = []
    for c in widget.winfo_children():
        if isinstance(c, cls):
            out.append(c)
        out.extend(_find(c, cls))
    return out


@pytest.fixture()
def rep_app(tk_root, tmp_path):
    vol, _ = make_rock_stack(n_slices=4, size=32, seed=0, cupping=0.3)
    src = tmp_path / "vol.tif"
    save_volume(vol, src)
    from rockstone.gui import RockstoneApp

    return RockstoneApp(tk_root), src


def test_report_button_states(rep_app):
    app, src = rep_app
    assert str(app.report_btn.cget("state")) == "disabled"
    app.load_path(src)
    assert str(app.report_btn.cget("state")) == "normal"


def test_gui_report_flow_html(rep_app, tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import ttk

    from rockstone import gui as gui_module

    app, src = rep_app
    app.load_path(src)
    app.rotate_volume(1)  # one history entry to report on
    target = tmp_path / "out.html"
    monkeypatch.setattr(
        gui_module.filedialog, "asksaveasfilename",
        staticmethod(lambda **k: str(target)))
    app.generate_report_as()
    dlg = next(w for w in app.root.winfo_children()
               if isinstance(w, tk.Toplevel)
               and "report" in w.title().lower())
    next(b for b in _find(dlg, ttk.Button)
         if str(b.cget("text")) == "Generate").invoke()
    assert _pump(app.root, lambda: "report written"
                 in app.status.get()), app.status.get()
    html = target.read_text(encoding="utf-8")
    assert "<h1>Rockstone report</h1>" in html
    assert "did <b>rotate 90°</b>" in html       # session history
    assert html.count("data:image/png;base64,") == 2
    assert not bool(dlg.winfo_exists())          # dialog closed


def test_gui_report_flow_txt(rep_app, tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import ttk

    from rockstone import gui as gui_module

    app, src = rep_app
    app.load_path(src)
    target = tmp_path / "out.txt"
    monkeypatch.setattr(
        gui_module.filedialog, "asksaveasfilename",
        staticmethod(lambda **k: str(target)))
    app.generate_report_as()
    dlg = next(w for w in app.root.winfo_children()
               if isinstance(w, tk.Toplevel)
               and "report" in w.title().lower())
    _find(dlg, ttk.Combobox)[0].set("Plain text (*.txt)")
    next(b for b in _find(dlg, ttk.Button)
         if str(b.cget("text")) == "Generate").invoke()
    assert _pump(app.root, lambda: "report written"
                 in app.status.get()), app.status.get()
    text = target.read_text(encoding="utf-8")
    assert "Rockstone report" in text
    assert "(no modifications in this session)" in text


def test_gui_report_flow_pdf(rep_app, tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import ttk

    from rockstone import gui as gui_module

    app, src = rep_app
    app.load_path(src)
    target = tmp_path / "out.pdf"
    monkeypatch.setattr(
        gui_module.filedialog, "asksaveasfilename",
        staticmethod(lambda **k: str(target)))
    app.generate_report_as()
    dlg = next(w for w in app.root.winfo_children()
               if isinstance(w, tk.Toplevel)
               and "report" in w.title().lower())
    combos = _find(dlg, ttk.Combobox)
    combos[0].set("PDF document, with images (*.pdf)")
    next(b for b in _find(dlg, ttk.Button)
         if str(b.cget("text")) == "Generate").invoke()
    assert _pump(app.root, lambda: "report written"
                 in app.status.get()), app.status.get()
    assert target.read_bytes()[:5] == b"%PDF-"
    from pypdf import PdfReader

    assert "Rockstone report" in "\n".join(
        page.extract_text() for page in PdfReader(str(target)).pages)


def test_gui_report_flow_docx(rep_app, tmp_path, monkeypatch):
    import tkinter as tk
    import zipfile
    from tkinter import ttk

    from rockstone import gui as gui_module

    app, src = rep_app
    app.load_path(src)
    target = tmp_path / "out.docx"
    monkeypatch.setattr(
        gui_module.filedialog, "asksaveasfilename",
        staticmethod(lambda **k: str(target)))
    app.generate_report_as()
    dlg = next(w for w in app.root.winfo_children()
               if isinstance(w, tk.Toplevel)
               and "report" in w.title().lower())
    _find(dlg, ttk.Combobox)[0].set("Word document, with images (*.docx)")
    next(b for b in _find(dlg, ttk.Button)
         if str(b.cget("text")) == "Generate").invoke()
    assert _pump(app.root, lambda: "report written"
                 in app.status.get()), app.status.get()
    z = zipfile.ZipFile(target)
    assert sum(1 for n in z.namelist()
               if n.startswith("word/media/")) == 2  # slice + histogram


def test_gui_report_cancelled_keeps_dialog(rep_app, tmp_path,
                                           monkeypatch):
    from rockstone import gui as gui_module
    import tkinter as tk

    app, src = rep_app
    app.load_path(src)
    monkeypatch.setattr(gui_module.filedialog, "asksaveasfilename",
                        staticmethod(lambda **k: ""))
    app.generate_report_as()
    dlg = next(w for w in app.root.winfo_children()
               if isinstance(w, tk.Toplevel)
               and "report" in w.title().lower())
    assert bool(dlg.winfo_exists())
    dlg.destroy()
