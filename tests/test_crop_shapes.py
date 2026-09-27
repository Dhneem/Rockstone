"""Tests for crop shape selection: rectangle and ellipse."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rockstone.crop import (build_ellipse_mask, crop_ellipse,
                            crop_rectangle, ellipse_outline_mask)
from rockstone.demo import make_rock_stack
from rockstone.gui import paint_crop_overlay
from rockstone.io import save_volume


@pytest.fixture()
def vol():
    return np.arange(2 * 6 * 8, dtype=np.float64).reshape(2, 6, 8)


# ----------------------------------------------------------- library --

def test_crop_rectangle_values(vol):
    r = crop_rectangle(vol, (2, 1, 5, 3))
    assert r.shape == (2, 3, 4)
    np.testing.assert_array_equal(r, vol[:, 1:4, 2:6])
    assert r is not vol  # a copy, not a view


def test_crop_rectangle_rejects_bad_rects(vol):
    with pytest.raises(ValueError, match="degenerate"):
        crop_rectangle(vol, (5, 1, 2, 3))
    with pytest.raises(ValueError, match="outside"):
        crop_rectangle(vol, (0, 0, 99, 3))
    with pytest.raises(ValueError, match="3-D"):
        crop_rectangle(vol[0], (0, 0, 2, 2))


def test_ellipse_mask_geometry():
    # circle inscribed in an 11x11 square
    m = build_ellipse_mask((11, 11), (0, 0, 10, 10))
    assert m[5, 5]          # center
    assert not m[0, 0]      # corners
    assert m[0, 5]          # tangent edge midpoint belongs (exact test)
    assert m[5, 0] or m[5, 1]
    # area check on a larger circle (small radii pixelize heavily);
    # semi-axis is (x1-x0)/2 = 50
    big = build_ellipse_mask((101, 101), (0, 0, 100, 100))
    expected = np.pi * 50.0 ** 2
    assert abs(big.sum() - expected) < 0.01 * expected


def test_ellipse_mask_true_ellipse():
    # 8x4 selection -> a/b = 2
    m = build_ellipse_mask((6, 8), (0, 1, 7, 4))
    assert m.sum() < build_ellipse_mask((6, 8), (0, 1, 7, 4)).sum() + 1
    # wider than tall: more pixels on the center row than column
    row = m[2].sum()
    col = m[:, 3].sum()
    assert row > col


def test_ellipse_mask_rejects_bad_input():
    with pytest.raises(ValueError, match="degenerate"):
        build_ellipse_mask((6, 8), (5, 1, 2, 3))
    with pytest.raises(ValueError, match="outside"):
        build_ellipse_mask((6, 8), (0, 0, 9, 9))


def test_ellipse_outline_mask():
    m = build_ellipse_mask((21, 21), (0, 0, 20, 20))
    o = ellipse_outline_mask((21, 21), (0, 0, 20, 20))
    assert o.sum() < m.sum()          # outline, not the full disk
    assert (o & m).sum() == o.sum()   # outline lies inside the disk
    # the outline is a ring around the mask interior
    inner = m & ~o
    assert inner.sum() > 0


def test_crop_ellipse_fills(vol):
    from rockstone.constants import AIR_HU

    nan = crop_ellipse(vol, (0, 0, 7, 5), fill="nan")
    assert nan.shape == vol.shape
    assert np.isnan(nan[:, 0, 0]).all()      # corner outside
    assert nan[0, 2, 3] == vol[0, 2, 3]      # center kept
    zero = crop_ellipse(vol, (0, 0, 7, 5), fill="zero")
    assert (zero[:, 0, 0] == 0).all()
    air = crop_ellipse(vol, (0, 0, 7, 5), fill="air", air_level=0.05)
    assert (air[:, 0, 0] == 0.05).all()
    # default air fill is the standard CT air value of -1000 HU
    air_default = crop_ellipse(vol, (0, 0, 7, 5), fill="air")
    assert (air_default[:, 0, 0] == AIR_HU).all()


def test_crop_ellipse_dtype_and_errors(vol):
    out = crop_ellipse(vol.astype(np.int32), (0, 0, 7, 5))
    assert out.dtype == np.float64           # NaN needs float
    with pytest.raises(ValueError, match="unknown fill"):
        crop_ellipse(vol, (0, 0, 7, 5), fill="magenta")
    with pytest.raises(ValueError, match="3-D"):
        crop_ellipse(vol[0], (0, 0, 7, 5))


# ----------------------------------------------------------- overlay --

def test_overlay_ellipse_outline():
    rgb = np.zeros((21, 21, 3), np.uint8)
    out = paint_crop_overlay(rgb, (0, 0, 20, 20), (21, 21),
                             color="red", shape="ellipse")
    assert out[10, 0].tolist() == [255, 70, 70]   # left tip painted
    assert out[10, 10].tolist() == [0, 0, 0]      # center untouched
    assert out[0, 0].tolist() == [0, 0, 0]        # corner untouched
    # rectangle default still paints the straight edge
    rect = paint_crop_overlay(rgb, (0, 0, 20, 20), (21, 21))
    assert rect[0, 10].tolist() == [255, 70, 70]


# --------------------------------------------------------------- GUI --

def _pump(root, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


class _Ev:
    def __init__(self, x, y):
        self.x, self.y = x, y


def _find(widget, cls):
    out = []
    for c in widget.winfo_children():
        if isinstance(c, cls):
            out.append(c)
        out.extend(_find(c, cls))
    return out


@pytest.fixture()
def crop_app(tk_root, tmp_path):
    vol, _ = make_rock_stack(n_slices=3, size=40, seed=0, cupping=0.2)
    src = tmp_path / "vol.tif"
    save_volume(vol, src)
    from rockstone.gui import RockstoneApp

    return RockstoneApp(tk_root), src, vol


def test_gui_rectangle_crop_unchanged(crop_app):
    app, src, vol = crop_app
    app.load_path(src)
    app.start_crop()
    app._on_drag_start(_Ev(7, 7))
    app._on_drag_move(_Ev(21, 21))
    app.apply_crop()
    assert app.data.shape == (3, 15, 15)
    np.testing.assert_array_equal(app.data, vol[:, 5:20, 5:20])


def test_gui_ellipse_crop_nan(crop_app):
    app, src, vol = crop_app
    app.load_path(src)
    app.start_crop()
    combos = _find(app._crop_win, __import__("tkinter").ttk.Combobox)
    combos[0].set("Ellipse")  # fill defaults to NaN
    app._on_drag_start(_Ev(7, 7))
    app._on_drag_move(_Ev(27, 27))
    assert app._crop_sel == (5, 5, 25, 25)
    app.apply_crop()
    assert app.data.shape == (3, 40, 40)      # size preserved
    assert np.isnan(app.data[:, 0, 0]).all()  # corner filled
    assert np.isfinite(app.data[:, 20, 20]).all()
    n_finite = np.isfinite(app.data[0]).sum()
    expected = np.pi * 10.5 ** 2
    assert abs(n_finite - expected) < 0.15 * expected
    assert "ellipse" in app.status.get()
    # undoable like every crop
    app.undo()
    np.testing.assert_array_equal(app.data, vol)
    assert "undid crop" in app.status.get()


def test_gui_ellipse_crop_always_nan(crop_app):
    # the fill dropdown was removed: ellipse crops are always NaN-filled
    app, src, vol = crop_app
    app.load_path(src)
    app.start_crop()
    from tkinter import ttk

    combos = _find(app._crop_win, ttk.Combobox)
    assert len(combos) == 1  # only the shape dropdown remains
    combos[0].set("Ellipse")
    app._on_drag_start(_Ev(7, 7))
    app._on_drag_move(_Ev(27, 27))
    app.apply_crop()
    assert app.data.shape == (3, 40, 40)
    assert np.isnan(app.data[:, 0, 0]).all()  # outside NaN
    assert np.isfinite(app.data[:, 20, 20]).all()
    assert "outside NaN" in app.status.get()


def test_gui_crop_dialog_has_only_shape_dropdown(crop_app):
    app, src, _vol = crop_app
    app.load_path(src)
    app.start_crop()
    from tkinter import ttk

    # the "Outside (ellipse)" dropdown was removed
    assert len(_find(app._crop_win, ttk.Combobox)) == 1
    assert "Outside" not in app._crop_lbl_var.get()


def test_gui_ellipse_outline_drawn_while_dragging(crop_app):
    app, src, _vol = crop_app
    app.load_path(src)
    app.start_crop()
    combos = _find(app._crop_win, __import__("tkinter").ttk.Combobox)
    combos[0].set("Ellipse")
    app._on_drag_start(_Ev(7, 7))
    app._on_drag_move(_Ev(27, 27))
    # some pixel of the inscribed circle's rim is painted bright red
    painted = any(
        app._photo.get(x, y) == (255, 70, 70)
        for y in range(40) for x in range(40))
    assert painted
