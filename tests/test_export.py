"""Tests for slice export: PNG, SVG, PDF and CSV."""

from __future__ import annotations

import csv
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rockstone.export import (EXPORT_FORMATS, export_slice,
                              physical_pixel_size)
from rockstone.io import save_volume


@pytest.fixture()
def ramp_data():
    """Small float stack + matching grayscale rendering."""
    data = np.tile(np.linspace(0.0, 1.0, 32), (3, 16, 1))
    u8 = np.tile(np.arange(0, 256, 8, dtype=np.uint8), (16, 1))
    return data, u8


META = {"pixel_size": [0.05, 0.05], "slice_spacing": 0.05}


# ------------------------------------------------------------ library --

def test_export_png_roundtrip_and_dpi(tmp_path, ramp_data):
    data, u8 = ramp_data
    p = export_slice(data, u8, tmp_path / "s.png", meta=META,
                     slice_index=0, colormap="Gray")
    from PIL import Image

    im = Image.open(p)
    assert im.format == "PNG"
    assert im.size == (32, 16)
    assert im.mode == "L"
    dpi = im.info.get("dpi")
    assert dpi is not None
    assert dpi[0] == pytest.approx(25.4 / 0.05, rel=0.01)  # 508 dpi
    np.testing.assert_array_equal(np.asarray(im), u8)


def test_export_png_rgb(tmp_path):
    rgb = np.zeros((6, 6, 3), np.uint8)
    rgb[..., 1] = 90
    p = export_slice(None, rgb, tmp_path / "c.png")
    from PIL import Image

    im = Image.open(p)
    assert im.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(im), rgb)


def test_export_pdf_document(tmp_path, ramp_data):
    data, u8 = ramp_data
    p = export_slice(data, u8, tmp_path / "s.pdf", meta=META,
                     slice_index=0)
    raw = p.read_bytes()
    assert raw[:4] == b"%PDF"
    assert b"/Filter" in raw  # embedded image stream
    # physical size honoring the DPI metadata
    page_w_mm = 32 * 0.05
    assert raw.count(b"MediaBox") == 1
    # Pillow writes points (841.9 if A4 default was used) -> check our
    # exact pixel-derived size: 32 px at 508 dpi = 1.6 mm = 4.535 pt
    assert b"4.54" in raw or b"4.535" in raw


def test_export_svg_vector_runs(tmp_path, ramp_data):
    data, u8 = ramp_data
    p = export_slice(data, u8, tmp_path / "s.svg", meta=META,
                     slice_index=1, colormap="Gray")
    root = ET.parse(p).getroot()
    assert root.tag.endswith("svg")
    assert root.get("width") == "1.6mm"   # 32 px * 0.05 mm
    assert root.get("height") == "0.8mm"  # 16 px * 0.05 mm
    assert root.get("viewBox") == "0 0 32 16"
    paths = [e for e in root.iter() if e.tag.endswith("path")]
    # ramp rows merge to runs -> exactly 32 gray levels, 32 paths
    assert len(paths) == 32
    fills = {e.get("fill") for e in paths}
    assert all(f and f.startswith("#") and len(f) == 7 for f in fills)
    assert len(fills) == 32  # distinct gray values


def test_export_svg_color_rects(tmp_path):
    rgb = np.zeros((6, 8, 3), np.uint8)
    rgb[..., 0] = 200
    rgb[:, 4:] = (10, 100, 200)
    p = export_slice(None, rgb, tmp_path / "c.svg", colormap="Rock")
    text = p.read_text()
    root = ET.parse(p).getroot()
    assert root.get("width") == "8"  # no meta -> unitless pixels
    rects = [e for e in root.iter() if e.tag.endswith("rect")]
    assert rects  # color images use <rect> elements
    assert 'fill="#c80000"' in text
    assert 'fill="#0a64c8"' in text


def test_export_svg_records_raw_range(tmp_path, ramp_data):
    data, u8 = ramp_data
    p = export_slice(data, u8, tmp_path / "m.svg", slice_index=0)
    assert "raw values: min 0, max 1" in p.read_text()


def test_export_csv_roundtrip(tmp_path, ramp_data):
    data, _u8 = ramp_data
    p = export_slice(data, None, tmp_path / "s.csv", slice_index=2)
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["row", "col", "value"]
    body = rows[1:]
    assert len(body) == 16 * 32
    vals = np.array([float(r[2]) for r in body]).reshape(16, 32)
    np.testing.assert_array_equal(vals, data[2])
    assert body[0][0] == "0" and body[0][1] == "0"
    # last value of the ramp row
    assert float(body[31][2]) == pytest.approx(1.0)


def test_export_csv_nan_cells(tmp_path):
    data = np.ones((1, 4, 4))
    data[0, 1, 2] = np.nan
    p = export_slice(data, None, tmp_path / "n.csv", slice_index=0)
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[1 + 1 * 4 + 2][2] == ""  # NaN -> empty cell
    assert float(rows[1][2]) == 1.0      # others survive


def test_export_rejects_unknown_and_csv_without_data(tmp_path, ramp_data):
    data, u8 = ramp_data
    with pytest.raises(ValueError, match="unsupported export format"):
        export_slice(data, u8, tmp_path / "s.bmp")
    with pytest.raises(ValueError, match="needs the raw data"):
        export_slice(None, u8, tmp_path / "s.csv")


def test_physical_pixel_size_variants():
    assert physical_pixel_size(META) == 0.05
    assert physical_pixel_size({"pixel_size": 0.3}) == 0.3
    assert physical_pixel_size(None) is None
    assert physical_pixel_size({}) is None
    assert physical_pixel_size({"pixel_size": [0.1, 0.2]}) is None
    assert physical_pixel_size({"pixel_size": [0, 0]}) is None


def test_export_formats_table():
    exts = [e for _, e in EXPORT_FORMATS]
    assert exts == [".png", ".svg", ".pdf", ".csv"]


# ------------------------------------------------------------- GUI flows --

def _pump(root, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def exp_app(tk_root, tmp_path):
    from rockstone.demo import make_rock_stack
    from rockstone.gui import RockstoneApp

    vol, _ = make_rock_stack(n_slices=4, size=40, seed=0, cupping=0.3)
    src = tmp_path / "vol.tif"
    save_volume(vol, src, meta={"pixel_size": [0.2, 0.2],
                                "slice_spacing": 0.2})
    return RockstoneApp(tk_root), src


def test_export_button_states(exp_app):
    app, src = exp_app
    assert str(app.export_btn.cget("state")) == "disabled"
    app.load_path(src)
    assert str(app.export_btn.cget("state")) == "normal"


def _find(widget, cls):
    """All widgets of type ``cls`` under ``widget``, at any depth."""
    out = []
    for c in widget.winfo_children():
        if isinstance(c, cls):
            out.append(c)
        out.extend(_find(c, cls))
    return out


def _find_buttons(w):
    from tkinter import ttk

    return _find(w, ttk.Button)


def test_gui_export_png_flow(exp_app, tmp_path, monkeypatch):
    import tkinter as tk

    from rockstone import gui as gui_module

    app, src = exp_app
    app.load_path(src)
    target = tmp_path / "out.png"
    monkeypatch.setattr(
        gui_module.filedialog, "asksaveasfilename",
        staticmethod(lambda **k: str(target)))
    app.export_slice_as()
    dlg = next(w for w in app.root.winfo_children()
               if isinstance(w, tk.Toplevel)
               and "export" in w.title().lower())
    next(b for b in _find_buttons(dlg)
         if str(b.cget("text")) == "Export").invoke()
    assert _pump(app.root, lambda: "exported slice 1"
                 in app.status.get()), app.status.get()
    assert target.exists()
    from PIL import Image

    im = Image.open(target)
    assert im.size == (app.data.shape[2], app.data.shape[1])
    dpi = im.info.get("dpi")
    assert dpi[0] == pytest.approx(25.4 / 0.2, rel=0.01)  # 127 dpi


def test_gui_export_csv_flow(exp_app, tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import ttk

    from rockstone import gui as gui_module

    app, src = exp_app
    app.load_path(src)
    app.slice_var.set(2)
    app._on_slice_move()
    target = tmp_path / "out.csv"
    picked = {}

    def fake_dialog(**k):
        picked.update(k)
        return str(target)

    monkeypatch.setattr(gui_module.filedialog, "asksaveasfilename",
                        staticmethod(fake_dialog))
    app.export_slice_as()
    dlg = next(w for w in app.root.winfo_children()
               if isinstance(w, tk.Toplevel)
               and "export" in w.title().lower())

    # pick the CSV format in the dialog, then export
    from tkinter import ttk

    combo = _find(dlg, ttk.Combobox)[0]
    combo.set("CSV values (*.csv)")
    next(b for b in _find_buttons(dlg)
         if str(b.cget("text")) == "Export").invoke()
    assert picked["initialfile"] == "vol_slice3.csv"
    assert _pump(app.root, lambda: target.exists()), app.status.get()
    with open(target, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["row", "col", "value"]
    assert len(rows) - 1 == app.data.shape[1] * app.data.shape[2]


def test_gui_export_cancelled_dialog_keeps_open(exp_app, tmp_path,
                                                monkeypatch):
    import tkinter as tk
    from tkinter import ttk

    from rockstone import gui as gui_module

    app, src = exp_app
    app.load_path(src)
    monkeypatch.setattr(gui_module.filedialog, "asksaveasfilename",
                        staticmethod(lambda **k: ""))  # user cancelled
    app.export_slice_as()
    dlg = next(w for w in app.root.winfo_children()
               if isinstance(w, tk.Toplevel)
               and "export" in w.title().lower())
    assert bool(dlg.winfo_exists())  # still open for another try
