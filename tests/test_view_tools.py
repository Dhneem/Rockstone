"""Tests for view tools: rotate, brightness/contrast, histogram panel,
and physical-dimension metadata (DICOM / TIFF / NPZ)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rockstone.io import load_volume, read_metadata, save_volume
from rockstone.gui import histogram_counts, window_u8


# --------------------------------------------------------------- metadata --

def test_metadata_roundtrip_tiff(tmp_path):
    vol = np.random.default_rng(1).random((4, 8, 8)).astype(np.float32)
    p = tmp_path / "v.tif"
    save_volume(vol, p, meta={"pixel_size": [0.05, 0.05],
                              "slice_spacing": 0.03})
    np.testing.assert_array_equal(load_volume(p), vol)  # pixels lossless
    assert read_metadata(p) == {"pixel_size": [0.05, 0.05],
                                "slice_spacing": 0.03}


def test_metadata_roundtrip_npz(tmp_path):
    vol = np.arange(2 * 6 * 6, dtype=np.float64).reshape(2, 6, 6)
    p = tmp_path / "v.npz"
    save_volume(vol, p, meta={"pixel_size": [1.0, 2.0],
                              "slice_spacing": 3.0})
    np.testing.assert_array_equal(load_volume(p), vol)
    assert read_metadata(p)["slice_spacing"] == 3.0


def test_metadata_absent_returns_none(tmp_path):
    vol = np.zeros((2, 4, 4))
    p_tif = tmp_path / "plain.tif"
    save_volume(vol, p_tif)  # no meta
    assert read_metadata(p_tif) is None
    p_npy = tmp_path / "plain.npy"
    save_volume(vol, p_npy, meta={"pixel_size": [0.1, 0.1],
                                  "slice_spacing": 0.1})  # .npy cannot hold it
    assert read_metadata(p_npy) is None
    assert read_metadata(tmp_path / "missing.tif") is None


def test_metadata_from_dicom(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.77.1"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = "1.2.826.0.1.77"
    ds.StudyInstanceUID = "1.2.826.0.1.42"
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = 6, 8
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    # DICOM PixelSpacing is [row, col]; read_metadata reports [x, y]
    ds.PixelSpacing = [0.4, 0.5]
    ds.SliceThickness = 1.25
    ds.PixelData = np.full((6, 8), 500, dtype=np.int16).tobytes()
    p = tmp_path / "slice.dcm"
    ds.save_as(p, enforce_file_format=True)
    del pydicom

    out = read_metadata(p)
    assert out == {"pixel_size": [0.5, 0.4], "slice_spacing": 1.25}
    # a folder of DICOM files is scanned for the first one
    assert read_metadata(tmp_path) == out


def test_read_metadata_survives_garbage(tmp_path):
    p = tmp_path / "x.tif"
    p.write_bytes(b"not a tiff")
    assert read_metadata(p) is None  # best-effort: never raises


# ------------------------------------------------- display window / hist --

def test_window_u8_stretches_range():
    ramp = np.tile(np.linspace(0.0, 1.0, 100), (10, 1))
    out = window_u8(ramp, 0.25, 0.75)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255
    assert out[0, 0] == 0        # below window -> clipped to black
    assert out[0, -1] == 255     # above window -> clipped to white
    mid = int(round((0.5 - 0.25) / 0.5 * 255))
    assert abs(int(out[0, 50]) - mid) <= 2  # uint8 rounding


def test_window_u8_degenerate_and_nan():
    plane = np.array([[np.nan, 0.5], [1.0, np.inf]])
    out = window_u8(plane, 0.75, 0.75)  # hi <= lo must not divide by zero
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()


def test_histogram_counts_bins_and_sum():
    rng = np.random.default_rng(2)
    u8 = rng.integers(0, 256, size=(50, 40), dtype=np.uint8)
    counts = histogram_counts(u8)
    assert counts.size == 32
    assert counts.sum() == 50 * 40
    assert counts.min() >= 0
    # deterministic: matches numpy directly
    ref, _ = np.histogram(u8.ravel(), bins=32, range=(0, 255))
    np.testing.assert_array_equal(counts, ref)


# ------------------------------------------------------------- GUI flows --

@pytest.fixture()
def view_app(tk_root, tmp_path):
    from rockstone.demo import make_rock_stack
    from rockstone.gui import RockstoneApp

    vol, _ = make_rock_stack(n_slices=5, size=32, seed=0, cupping=0.3)
    src = tmp_path / "vol.tif"
    save_volume(vol, src)
    app = RockstoneApp(tk_root)
    return app, src


def _pump(root, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


class TestRotate:
    def test_clockwise_rotates_in_plane(self, view_app):
        app, src = view_app
        app.load_path(src)
        before = app.data
        app.rotate_volume(1)
        assert app.data.shape == (before.shape[0], before.shape[2],
                                  before.shape[1])
        np.testing.assert_array_equal(app.data[0], np.rot90(before[0], k=1))
        assert "rotated 90° clockwise" in app.status.get()
        assert app._processed is True
        assert app._dirty is True
        assert "*" in app.root.title()

    def test_ccw_is_inverse(self, view_app):
        app, src = view_app
        app.load_path(src)
        before = app.data
        app.rotate_volume(-1)
        app.rotate_volume(1)
        np.testing.assert_array_equal(app.data, before)

    def test_undo_redo(self, view_app):
        app, src = view_app
        app.load_path(src)
        before = app.data
        app.rotate_volume(1)
        app.undo()
        assert app.data is before
        assert "undid rotate" in app.status.get()
        app.redo()
        assert "re-applied rotate" in app.status.get()
        np.testing.assert_array_equal(app.data, np.rot90(before, k=1,
                                                         axes=(1, 2)))

    def test_noop_without_data(self, view_app):
        app, _src = view_app
        app.rotate_volume(1)  # must not raise
        assert app.data is None


class TestBrightnessContrast:
    def test_slider_narrows_window_and_rerenders(self, view_app):
        app, src = view_app
        app.load_path(src)
        plane = app.displayed
        base_u8 = app._ensure_u8(plane).copy()
        base_lo, base_hi = app._display_range(plane)

        app.contrast_var.set(50.0)
        app._on_brightness()
        assert app._window is not None
        lo, hi = app._window
        assert (hi - lo) < (base_hi - base_lo)  # contrast narrows
        # cache key changed -> a different rendering is produced
        assert not np.array_equal(app._ensure_u8(plane), base_u8)

    def test_brightness_shifts_window(self, view_app):
        app, src = view_app
        app.load_path(src)
        plane = app.displayed
        lo0, hi0 = app._display_range(plane)
        app.bright_var.set(50.0)
        app._on_brightness()
        lo1, hi1 = app._window
        width0, width1 = hi0 - lo0, hi1 - lo1
        assert width1 == pytest.approx(width0)
        assert lo1 > lo0 and hi1 > hi0  # window slid upward

    def test_reset_display_restores_default(self, view_app):
        app, src = view_app
        app.load_path(src)
        plane = app.displayed
        base_u8 = app._ensure_u8(plane).copy()
        app.bright_var.set(80.0)
        app.contrast_var.set(-60.0)
        app._on_brightness()
        app._reset_display()
        assert app._window is None
        assert app.bright_var.get() == 0.0
        assert app.contrast_var.get() == 0.0
        np.testing.assert_array_equal(app._ensure_u8(plane), base_u8)

    def test_widening_negative_contrast(self, view_app):
        app, src = view_app
        app.load_path(src)
        plane = app.displayed
        lo0, hi0 = app._display_range(plane)
        app.contrast_var.set(-50.0)
        app._on_brightness()
        lo1, hi1 = app._window
        assert (hi1 - lo1) > (hi0 - lo0)  # negative contrast widens


class TestHistogramPanel:
    def test_draws_bars_for_loaded_data(self, view_app):
        app, src = view_app
        assert not app.hist_canvas.find_all()  # empty at startup
        app.load_path(src)
        assert app.hist_canvas.find_all()  # bars + label drawn
        # counts behind the drawing match the helper
        u8 = app._norm_cache[2]
        counts = histogram_counts(u8[..., 0] if u8.ndim == 3 else u8)
        assert counts.sum() == u8.shape[0] * u8.shape[1]

    def test_redraws_on_slice_change(self, view_app):
        app, src = view_app
        app.load_path(src)
        app.slice_var.set(3)
        app._on_slice_move()
        assert app._shown_index == 3
        assert app.hist_canvas.find_all()

    def test_empty_without_data(self, view_app):
        app, _src = view_app
        app._draw_histogram()
        assert not app.hist_canvas.find_all()


class TestDimensions:
    def test_meta_prefilled_from_file(self, view_app, tmp_path):
        app, src = view_app
        save_volume(load_volume(src), src, meta={"pixel_size": [0.2, 0.2],
                                                 "slice_spacing": 0.1})
        app.load_path(src)
        assert app._meta == {"pixel_size": [0.2, 0.2], "slice_spacing": 0.1}
        assert "voxel: 0.2 x 0.2 x 0.1 mm" in app.info.get()

    def test_dialog_sets_and_clears_meta(self, view_app):
        import tkinter as tk
        from tkinter import ttk

        app, src = view_app
        app.load_path(src)
        app.edit_dimensions()
        dlg = next(w for w in app.root.winfo_children()
                   if isinstance(w, tk.Toplevel) and "dimensions" in
                   w.title().lower())
        frm = next(c for c in dlg.winfo_children())
        entries = [w for w in frm.winfo_children() if isinstance(w, ttk.Entry)]
        assert len(entries) == 3
        for e in entries:
            e.delete(0, "end")
        entries[0].insert(0, "0.1")
        entries[1].insert(0, "0.1")
        entries[2].insert(0, "0.2")
        frames = [w for w in frm.winfo_children() if isinstance(w, ttk.Frame)]
        buttons = [b for f in frames for b in f.winfo_children()
                   if isinstance(b, ttk.Button)]
        ok_btn = next(b for b in buttons if str(b.cget("text")) == "OK")
        ok_btn.invoke()
        assert app._meta == {"pixel_size": [0.1, 0.1], "slice_spacing": 0.2}
        assert "voxel: 0.1 x 0.1 x 0.2 mm" in app.info.get()

        # reopening prefills; clearing all fields removes the metadata
        app.edit_dimensions()
        dlg2 = next(w for w in app.root.winfo_children()
                    if isinstance(w, tk.Toplevel) and "dimensions" in
                    w.title().lower())
        frm2 = next(c for c in dlg2.winfo_children())
        entries2 = [w for w in frm2.winfo_children()
                    if isinstance(w, ttk.Entry)]
        assert entries2[0].get() == "0.1"  # prefilled from app._meta
        for e in entries2:
            e.delete(0, "end")
        frames2 = [w for w in frm2.winfo_children()
                   if isinstance(w, ttk.Frame)]
        ok2 = next(b for f in frames2 for b in f.winfo_children()
                   if isinstance(b, ttk.Button)
                   and str(b.cget("text")) == "OK")
        ok2.invoke()
        assert app._meta is None

    def test_dialog_rejects_non_numeric(self, view_app, monkeypatch):
        import tkinter as tk

        from rockstone import gui as gui_module

        app, src = view_app
        app.load_path(src)
        errors = []
        monkeypatch.setattr(gui_module.messagebox, "showerror",
                            staticmethod(lambda *a, **k: errors.append(a)))
        app.edit_dimensions()
        dlg = next(w for w in app.root.winfo_children()
                   if isinstance(w, tk.Toplevel) and "dimensions" in
                   w.title().lower())
        frm = next(c for c in dlg.winfo_children())
        entries = [w for w in frm.winfo_children()
                   if type(w).__name__ == "Entry"]
        entries[0].insert(0, "abc")
        frames = [w for w in frm.winfo_children()
                  if type(w).__name__ == "Frame"]
        ok_btn = next(b for f in frames for b in f.winfo_children()
                      if type(b).__name__ == "TButton" or
                      (hasattr(b, "cget") and str(b.cget("text")) == "OK"))
        ok_btn.invoke()
        assert errors  # error dialog shown, dialog stays open
        assert app._meta is None

    def test_meta_embedded_on_save(self, view_app, tmp_path, monkeypatch):
        from rockstone import gui as gui_module

        app, src = view_app
        app.load_path(src)
        app._meta = {"pixel_size": [0.05, 0.05], "slice_spacing": 0.05}
        target = tmp_path / "out.tif"
        monkeypatch.setattr(
            gui_module.filedialog, "asksaveasfilename",
            staticmethod(lambda **k: str(target)))
        app.save_as()
        assert _pump(app.root, lambda: target.exists())
        assert target.exists()
        assert read_metadata(target) == {"pixel_size": [0.05, 0.05],
                                         "slice_spacing": 0.05}
        np.testing.assert_array_equal(load_volume(target), app.data)

    def test_save_without_meta_stays_clean(self, view_app, tmp_path,
                                           monkeypatch):
        from rockstone import gui as gui_module

        app, src = view_app
        app.load_path(src)
        assert app._meta is None
        target = tmp_path / "plain.tif"
        monkeypatch.setattr(
            gui_module.filedialog, "asksaveasfilename",
            staticmethod(lambda **k: str(target)))
        app.save_as()
        assert _pump(app.root, lambda: target.exists())
        assert read_metadata(target) is None
