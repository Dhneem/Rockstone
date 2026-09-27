"""Tests for the ROI (region of interest) feature: library, overlay,
GUI and CLI."""

from __future__ import annotations

import csv
import time

import numpy as np
import pytest

from rockstone.demo import make_rock_stack
from rockstone.gui import paint_crop_overlay
from rockstone.io import save_volume
from rockstone.roi import (roi_stats, roi_stats_series, write_roi_csv)


@pytest.fixture()
def block_vol():
    vol = np.zeros((3, 40, 40))
    vol[:, 10:20, 10:20] = 0.5  # 10x10 block at 0.5
    vol[1] += 0.1               # slice 1: 0.6
    return vol


META = {"pixel_size": [0.05, 0.05], "slice_spacing": 0.05}


# ----------------------------------------------------------- library --

def test_roi_stats_values(block_vol):
    s = roi_stats(block_vol, (10, 10, 19, 19), 0)
    assert s.n_pixels == 100
    assert s.mean == 0.5 and s.std == 0.0
    assert s.minimum == 0.5 and s.maximum == 0.5
    assert s.median == 0.5
    assert s.slice_index == 0
    assert s.area_mm2 is None  # no meta -> no physical area


def test_roi_stats_per_slice_differs(block_vol):
    s0 = roi_stats(block_vol, (10, 10, 19, 19), 0)
    s1 = roi_stats(block_vol, (10, 10, 19, 19), 1)
    assert s1.mean == pytest.approx(0.6)
    assert s1.mean != s0.mean


def test_roi_stats_area_from_meta(block_vol):
    s = roi_stats(block_vol, (10, 10, 19, 19), 0, meta=META)
    assert s.area_mm2 == pytest.approx(0.05 * 0.05 * 100)


def test_roi_stats_air_reference(block_vol):
    s = roi_stats(block_vol, (10, 10, 19, 19), 0, air_reference=0.05)
    assert s.mean_above_air == pytest.approx(0.45)
    assert roi_stats(block_vol, (10, 10, 19, 19), 0).mean_above_air is None


def test_roi_stats_nan_ignored():
    vol = np.zeros((1, 10, 10))
    vol[0, 2, 2] = np.nan
    s = roi_stats(vol, (0, 0, 9, 9), 0)
    assert s.n_pixels == 99
    assert np.isfinite(s.mean)


def test_roi_stats_rejects_bad_input(block_vol):
    with pytest.raises(ValueError, match="degenerate"):
        roi_stats(block_vol, (5, 5, 3, 8), 0)
    with pytest.raises(ValueError, match="outside the slice"):
        roi_stats(block_vol, (35, 35, 45, 45), 0)
    with pytest.raises(IndexError):
        roi_stats(block_vol, (0, 0, 3, 3), 9)
    with pytest.raises(ValueError, match="3-D"):
        roi_stats(block_vol[0], (0, 0, 3, 3), 0)


def test_roi_series_and_csv(block_vol, tmp_path):
    from rockstone.constants import AIR_HU

    series = roi_stats_series(block_vol, (10, 10, 19, 19), meta=META,
                              with_air=True)
    assert len(series) == 3
    assert series[1].mean == pytest.approx(0.6)
    assert series[0].air_reference == AIR_HU  # -1000 HU standard
    p = write_roi_csv(series, tmp_path / "r.csv")
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    assert len(rows) == 3
    assert rows[0]["slice"] == "1" and rows[2]["slice"] == "3"
    assert float(rows[1]["mean"]) == pytest.approx(0.6)
    assert "mean_above_air" in rows[0]
    with pytest.raises(ValueError, match="no ROI measurements"):
        write_roi_csv([], tmp_path / "x.csv")


# ----------------------------------------------------------- overlay --

def test_overlay_colors():
    rgb = np.zeros((10, 10, 3), np.uint8)
    red = paint_crop_overlay(rgb, (2, 2, 7, 7), (10, 10), color="red")
    cyan = paint_crop_overlay(rgb, (2, 2, 7, 7), (10, 10), color="cyan")
    yellow = paint_crop_overlay(rgb, (2, 2, 7, 7), (10, 10),
                                color="yellow")
    assert red[2, 4].tolist() == [255, 70, 70]
    assert cyan[2, 4].tolist() == [60, 220, 255]
    assert yellow[2, 4].tolist() == [255, 220, 60]
    assert red[2, 4].tolist() != cyan[2, 4].tolist()


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


def _all_buttons(widget):
    from tkinter import ttk

    out = []
    for c in widget.winfo_children():
        if isinstance(c, ttk.Button):
            out.append(c)
        out.extend(_all_buttons(c))
    return out


@pytest.fixture()
def roi_app(tk_root, tmp_path):
    vol, _ = make_rock_stack(n_slices=5, size=48, seed=0, cupping=0.3)
    src = tmp_path / "vol.tif"
    save_volume(vol, src, meta={"pixel_size": [0.1, 0.1],
                                "slice_spacing": 0.1})
    from rockstone.gui import RockstoneApp

    return RockstoneApp(tk_root), src


def test_roi_button_states(roi_app):
    app, src = roi_app
    assert str(app.roi_btn.cget("state")) == "disabled"
    app.load_path(src)
    assert str(app.roi_btn.cget("state")) == "normal"


def test_gui_roi_drag_stats_and_overlay(roi_app):
    app, src = roi_app
    app.load_path(src)
    app.start_roi()
    assert app._roi_win is not None
    app._on_drag_start(_Ev(12, 12))
    app._on_drag_move(_Ev(32, 32))
    assert app._roi_sel == (10, 10, 30, 30)
    text = app._roi_stats_var.get()
    assert "mean" in text and "std" in text
    assert "21 x 21 px" in text and "mm²" in text
    # cyan outline painted into the displayed photo
    assert app._photo.get(15, 10) == (60, 220, 255)   # top edge
    assert app._photo.get(10, 15) == (60, 220, 255)   # left edge
    # interior not painted
    assert app._photo.get(20, 20) != (60, 220, 255)


def test_gui_roi_stats_follow_slice(roi_app):
    app, src = roi_app
    app.load_path(src)
    app.start_roi()
    app._on_drag_start(_Ev(12, 12))
    app._on_drag_move(_Ev(32, 32))
    assert "slice 1" in app._roi_stats_var.get()
    app.slice_var.set(2)
    app._on_slice_move()
    assert "slice 3" in app._roi_stats_var.get()


def test_gui_roi_escape_closes(roi_app):
    app, src = roi_app
    app.load_path(src)
    app.start_roi()
    app._on_drag_start(_Ev(12, 12))
    app._on_drag_move(_Ev(32, 32))
    app._on_escape()
    assert app._roi_win is None and app._roi_sel is None
    assert app.data.shape == (5, 48, 48)  # nothing modified


def test_gui_roi_crop_to_roi(roi_app):
    app, src = roi_app
    app.load_path(src)
    app.start_roi()
    app._on_drag_start(_Ev(12, 12))
    app._on_drag_move(_Ev(32, 32))
    before = app.data.copy()
    crop_btn = next(b for b in _all_buttons(app._roi_win)
                    if str(b.cget("text")) == "Crop to ROI…")
    crop_btn.invoke()
    assert app.data.shape == (5, 21, 21)
    np.testing.assert_array_equal(app.data, before[:, 10:31, 10:31])
    assert "cropped to ROI" in app.status.get()
    assert app._dirty is True and "*" in app.root.title()
    app.undo()
    np.testing.assert_array_equal(app.data, before)
    assert "undid crop to ROI" in app.status.get()


def test_gui_roi_save_csv(roi_app, tmp_path, monkeypatch):
    from rockstone import gui as gui_module

    app, src = roi_app
    app.load_path(src)
    app.start_roi()
    app._on_drag_start(_Ev(12, 12))
    app._on_drag_move(_Ev(32, 32))
    target = tmp_path / "out.csv"
    monkeypatch.setattr(
        gui_module.filedialog, "asksaveasfilename",
        staticmethod(lambda **k: str(target)))
    csv_btn = next(b for b in _all_buttons(app._roi_win)
                   if str(b.cget("text")) == "Save all slices (CSV)…")
    csv_btn.invoke()
    assert _pump(app.root, lambda: "ROI measurements saved"
                 in app.status.get()), app.status.get()
    rows = list(csv.DictReader(open(target, newline="",
                                    encoding="utf-8")))
    assert len(rows) == 5
    assert all(int(r["n_pixels"]) == 441 for r in rows)


def test_gui_roi_busy_closes_dialog(roi_app):
    app, src = roi_app
    app.load_path(src)
    app.start_roi()
    assert app._roi_win is not None
    app._set_busy(True, "working")
    assert app._roi_win is None  # dialog auto-closed
    app._set_busy(False)


# --------------------------------------------------------------- CLI --

def test_cli_roi_single_slice(tmp_path, capsys):
    from rockstone.cli import main

    src = tmp_path / "in.tif"
    save_volume(make_rock_stack(n_slices=4, size=32, seed=0,
                                cupping=0.3)[0], src, meta=META)
    rc = main(["roi", str(src), "--rect", "5", "5", "15", "15",
               "--slice", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "slice 2" in out and "mean" in out and "area" in out


def test_cli_roi_all_slices_and_csv(tmp_path, capsys):
    from rockstone.cli import main

    src = tmp_path / "in.tif"
    save_volume(make_rock_stack(n_slices=4, size=32, seed=0,
                                cupping=0.3)[0], src, meta=META)
    rc = main(["roi", str(src), "--rect", "5", "5", "15", "15",
               "--all-slices"])
    assert rc == 0
    assert "slice   3" in capsys.readouterr().out
    rc = main(["roi", str(src), "--rect", "5", "5", "15", "15",
               "--csv"])
    assert rc == 0
    out_csv = tmp_path / "in_roi.csv"
    assert out_csv.exists()
    rows = list(csv.DictReader(open(out_csv, newline="",
                                    encoding="utf-8")))
    assert len(rows) == 4
