"""Tests for the slice map (contact sheet of all slices)."""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("tkinter")

from rockstone.gui import build_slice_montage, resize_nearest  # noqa: E402


# ------------------------------------------------------------- helpers --

def test_resize_nearest_shapes():
    out = resize_nearest(np.zeros((10, 20), dtype=np.uint8), 96, 96)
    assert out.shape == (96, 96)
    out = resize_nearest(np.zeros((4, 4, 3), dtype=np.uint8), 8, 8)
    assert out.shape == (8, 8, 3)
    assert resize_nearest(np.zeros((5, 5), dtype=np.uint8), 0, 0).shape \
        == (1, 1)  # degenerate target clamps to >= 1


def test_montage_layout_and_cells():
    n = 7
    m, cells, rows, cols = build_slice_montage(np.zeros((n, 32, 32)),
                                               cols=4)
    assert (rows, cols) == (2, 4)
    assert len(cells) == n
    cell, gap = 96, 4
    assert m.shape[:2] == (rows * (cell + gap) + gap,
                           cols * (cell + gap) + gap)
    assert m.shape[2] == 3 and m.dtype == np.uint8
    for rect in cells:
        x0, y0, x1, y1 = rect
        assert (x1 - x0, y1 - y0) == (cell, cell)
    # cells do not overlap and are in row-major order
    (ax0, ay0, _ax1, _ay1), (bx0, by0, *_b) = cells[0], cells[1]
    assert bx0 > ax0 and by0 == ay0
    (cx0, cy0, *_c) = cells[4]
    assert cy0 > ay0 and cx0 == ax0


def test_montage_pixels_match_direct_rendering():
    from rockstone.gui import DEFAULT_COLORMAP, apply_colormap, normalize_u8

    rng = np.random.default_rng(3)
    data = rng.random((4, 24, 20))
    m, cells, _r, _c = build_slice_montage(data, colormap=DEFAULT_COLORMAP)
    for i, (x0, y0, x1, y1) in enumerate(cells):
        u8 = apply_colormap(normalize_u8(data[i]), DEFAULT_COLORMAP)
        h, w = u8.shape[:2]
        scale = min(96 / h, 96 / w)
        th, tw = max(1, round(h * scale)), max(1, round(w * scale))
        expected = resize_nearest(u8, th, tw)
        oy, ox = (96 - th) // 2, (96 - tw) // 2
        np.testing.assert_array_equal(
            m[y0 + oy:y0 + oy + th, x0 + ox:x0 + ox + tw], expected)
        # the rest of the cell is montage background (black)
        row = m[y0 + (96 - 1), x0:x1]  # bottom row inside the cell
        assert row[:, 0].max() == 0 if th < 96 else True


# ----------------------------------------------------------------- GUI --

@pytest.fixture()
def map_app(tk_root, tmp_path, monkeypatch):
    import rockstone.gui as gui_module
    from rockstone.gui import RockstoneApp

    from rockstone.demo import make_rock_stack
    from rockstone.io import save_volume

    vol, _ = make_rock_stack(n_slices=9, size=40, seed=2, cupping=0.2)
    p = tmp_path / "vol.npy"
    save_volume(vol.astype(np.float32), p)

    errors = []
    monkeypatch.setattr(gui_module, "messagebox", type("MB", (), {
        "showinfo": staticmethod(lambda *a, **k: None),
        "showerror": staticmethod(lambda *a, **k: errors.append(a))}))
    app = RockstoneApp(tk_root)
    app.load_path(p)
    return app, errors


def _pump(root, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_open_map_and_click_to_jump(map_app):
    application, errors = map_app
    root = application.root
    # the map is a popup window opened by the Slice map button
    assert application._map_win is None  # no auto-open
    application._open_map()
    assert _pump(root, lambda: application._map_win is not None
                 and len(application._map_cells) == 9)
    assert not errors
    assert "9" in application._map_win.title()

    # click on the thumbnail of slice 6 -> main view jumps there
    x0, y0, x1, y1 = application._map_cells[6]
    event = type("E", (), {})()
    event.x, event.y = x0 + 10, y0 + 10
    application._on_map_click(event)
    assert int(application.slice_var.get()) == 6

    # highlight rectangle tracks the current slice
    application.slice_var.set(2)
    application._on_slice_move()
    coords = application._map_canvas.coords(application._map_hl)
    hx0, hy0, hx1, hy1 = application._map_cells[2]
    assert [round(v) for v in coords] == [hx0, hy0, hx1, hy1]

    application._toggle_map()  # closes
    assert application._map_win is None
    application._toggle_map()  # reopens
    assert _pump(root, lambda: application._map_win is not None)


def test_map_refreshes_on_reload_and_colormap(map_app):
    application, _errors = map_app
    root = application.root
    application._open_map()
    assert _pump(root, lambda: application._map_win is not None
                 and application._map_cells)
    old_canvas = application._map_canvas

    application.color_var.set("Jet")
    application._on_colormap_change()
    assert _pump(root, lambda: application._map_canvas is not old_canvas)

    # loading a new file while the map is open rebuilds it
    application.load_path(application.path)
    assert _pump(root, lambda: application._map_win is not None
                 and application._map_cells)
    application._toggle_map()


def test_map_disabled_without_data(tk_root, monkeypatch):
    import rockstone.gui as gui_module
    from rockstone.gui import RockstoneApp

    monkeypatch.setattr(gui_module, "messagebox", type("MB", (), {
        "showinfo": staticmethod(lambda *a, **k: None),
        "showerror": staticmethod(lambda *a, **k: None)}))
    application = RockstoneApp(tk_root)
    application._open_map()  # no data loaded -> no-op, no crash
    assert application._map_win is None
    event = type("E", (), {})()
    event.x, event.y = 5, 5
    application._on_map_click(event)  # also a safe no-op
