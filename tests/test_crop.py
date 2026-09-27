"""Tests for the interactive crop tool (drag a rectangle, apply to
every slice, undoable)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rockstone.gui import paint_crop_overlay
from rockstone.io import read_metadata, save_volume


# ----------------------------------------------------- overlay painting --

def test_paint_overlay_identity_and_pixels():
    img = np.zeros((10, 10), dtype=np.uint8)
    out = paint_crop_overlay(img, (2, 3, 7, 8), (10, 10))
    assert out is not img
    assert img.sum() == 0  # input untouched
    # rect (2,3)-(7,8) is inclusive: x 2..7, y 3..8
    assert (out[3, 2:8] == 255).all()   # top edge
    assert (out[8, 2:8] == 255).all()   # bottom edge
    assert (out[3:9, 2] == 255).all()   # left edge
    assert (out[3:9, 7] == 255).all()   # right edge
    assert out[3, 8] == 0 and out[2, 5] == 0  # outside corners
    assert out[5, 5] == 0               # interior untouched


def test_paint_overlay_rgb_is_red():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = paint_crop_overlay(img, (0, 0, 7, 7), (8, 8))
    assert out[0, 4].tolist() == [255, 70, 70]
    assert out[7, 4].tolist() == [255, 70, 70]


def test_paint_overlay_maps_zoomed_image():
    img = np.zeros((20, 20), dtype=np.uint8)
    # rect (5,5)-(9,9) in a 10x10 source, shown 2x
    out = paint_crop_overlay(img, (5, 5, 9, 9), (10, 10))
    assert out[10, 15] == 255 and out[19, 15] == 255
    assert out[5, 5] == 0  # outside the scaled rectangle


# ------------------------------------------------------------ GUI flows --

@pytest.fixture()
def crop_app(tk_root, tmp_path):
    from rockstone.demo import make_rock_stack
    from rockstone.gui import RockstoneApp

    vol, _ = make_rock_stack(n_slices=6, size=40, seed=0, cupping=0.3)
    src = tmp_path / "vol.tif"
    save_volume(vol, src)
    return RockstoneApp(tk_root), src


def _pump(root, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


class _Ev:
    """Fake Tk event with widget coordinates."""

    def __init__(self, x, y):
        self.x, self.y = x, y


def _drag(app, x0, y0, x1, y1):
    app._on_drag_start(_Ev(x0, y0))
    app._on_drag_move(_Ev(x1, y1))


class TestCropFlow:
    def test_button_states(self, crop_app):
        app, src = crop_app
        assert str(app.crop_btn.cget("state")) == "disabled"
        app.load_path(src)
        assert str(app.crop_btn.cget("state")) == "normal"

    def test_drag_enables_apply_and_paints(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        app.start_crop()
        assert app._crop_win is not None
        assert str(app._crop_apply_btn.cget("state")) == "disabled"
        _drag(app, 7, 7, 21, 21)  # widget coords -> data (5,5)..(19,19)
        assert app._crop_sel == (5, 5, 19, 19)
        assert str(app._crop_apply_btn.cget("state")) == "normal"
        assert "15 x 15 px" in app._crop_lbl_var.get()
        # the red outline is painted into the displayed photo
        assert app._photo.get(5, 5) == (255, 70, 70)

    def test_apply_crops_every_slice(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        before = app.data.copy()
        app.start_crop()
        _drag(app, 7, 7, 21, 21)
        app.apply_crop()
        assert app.data.shape == (6, 15, 15)
        np.testing.assert_array_equal(app.data, before[:, 5:20, 5:20])
        assert app._processed is True
        assert app._dirty is True
        assert "*" in app.root.title()
        assert "cropped: 15 x 15 px" in app.status.get()
        assert "shape: 6 x 15 x 15" in app.info.get()

    def test_undo_redo(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        before = app.data
        app.start_crop()
        _drag(app, 7, 7, 21, 21)
        app.apply_crop()
        app.undo()
        assert app.data is before
        assert "undid crop" in app.status.get()
        app.redo()
        assert app.data.shape == (6, 15, 15)
        assert "re-applied crop" in app.status.get()

    def test_reverse_drag_gives_same_rect(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        app.start_crop()
        _drag(app, 21, 21, 7, 7)
        assert app._crop_sel == (5, 5, 19, 19)

    def test_escape_cancels(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        app.start_crop()
        _drag(app, 7, 7, 21, 21)
        app._on_escape()
        assert app._crop_win is None
        assert app._crop_sel is None
        assert "crop cancelled" in app.status.get()
        assert app.data.shape == (6, 40, 40)  # untouched

    def test_cancel_button(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        app.start_crop()
        app._close_crop_dialog()
        assert app._crop_win is None
        assert app._crop_sel is None

    def test_tiny_selection_rejected(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        app.start_crop()
        _drag(app, 7, 7, 7, 7)  # single pixel: degenerate
        assert app._crop_sel is None
        assert str(app._crop_apply_btn.cget("state")) == "disabled"
        app.apply_crop()  # must be a no-op
        assert app.data.shape == (6, 40, 40)
        assert app._crop_win is not None  # dialog stays open
        # 2 x 2 px is the smallest accepted crop
        _drag(app, 7, 7, 8, 8)
        assert app._crop_sel == (5, 5, 6, 6)

    def test_events_clamp_to_image(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        app.start_crop()
        _drag(app, 0, 0, 5000, 5000)
        assert app._crop_sel == (0, 0, 39, 39)

    def test_busy_closes_dialog_and_disables(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        app.start_crop()
        assert app._crop_win is not None
        app._set_busy(True, "working")
        assert app._crop_win is None  # dialog dropped
        assert str(app.crop_btn.cget("state")) == "disabled"
        app._set_busy(False)
        assert str(app.crop_btn.cget("state")) == "normal"

    def test_crop_keeps_voxel_meta_on_save(self, crop_app, tmp_path,
                                           monkeypatch):
        from rockstone import gui as gui_module

        app, src = crop_app
        app.load_path(src)
        app._meta = {"pixel_size": [0.2, 0.2], "slice_spacing": 0.2}
        app.start_crop()
        _drag(app, 7, 7, 21, 21)
        app.apply_crop()
        target = tmp_path / "out.tif"
        monkeypatch.setattr(
            gui_module.filedialog, "asksaveasfilename",
            staticmethod(lambda **k: str(target)))
        app.save_as()
        assert _pump(app.root, lambda: target.exists())
        assert read_metadata(target) == {"pixel_size": [0.2, 0.2],
                                         "slice_spacing": 0.2}
        assert load_shape(target) == (6, 15, 15)

    def test_projections_kind_preserved(self, crop_app):
        app, src = crop_app
        app.load_path(src)
        assert app.kind == "volume"
        # a detector stack (nz >> ny, nx) is "projections"
        app.data = np.arange(24 * 12 * 12, dtype=np.float64) \
            .reshape(24, 12, 12)
        app.kind = "projections"
        app.start_crop()
        _drag(app, 7, 7, 13, 13)  # (5,5)..(11,11)
        app.apply_crop()
        assert app.data.shape == (24, 7, 7)
        assert app.kind == "projections"  # untouched by cropping


def load_shape(path):
    from rockstone.io import load_volume

    return load_volume(path).shape
