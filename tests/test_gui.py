"""Tests for the Tkinter GUI (window is created but kept withdrawn).

The suite skips gracefully when no display / Tk is available (e.g. a
bare CI runner); the pure-numpy helpers are still tested in that case.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("tkinter")

import tkinter as tk  # noqa: E402

from rockstone.gui import (  # noqa: E402
    RockstoneApp,
    detect_kind,
    normalize_u8,
    normalize_u8_plane3,
    to_pgm,
)


@pytest.fixture(scope="session")
def root(tk_root):
    # shared session root from conftest: Tk on Windows cannot reliably
    # re-initialize after all roots are destroyed
    return tk_root


def _pump(root, app: RockstoneApp, needle: str, timeout: float = 120.0) -> bool:
    """Process Tk events until ``needle`` appears in the status line."""
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if needle in app.status.get():
            return True
        time.sleep(0.02)
    return False


# ------------------------------------------------------------- helpers --

def test_detect_kind():
    assert detect_kind(np.zeros((24, 128, 128))) == "volume"
    assert detect_kind(np.zeros((200, 32, 64))) == "projections"


def test_normalize_u8_range_and_nan():
    plane = np.linspace(0, 1, 100).reshape(10, 10)
    plane[0, 0] = np.nan
    u8 = normalize_u8(plane)
    assert u8.dtype == np.uint8
    assert u8.max() <= 255 and u8.min() >= 0
    assert u8[0, 0] == 0  # NaN clips to the low end


def test_to_pgm_ppm_headers():
    gray = np.zeros((4, 5), dtype=np.uint8)
    assert to_pgm(gray).startswith(b"P5 5 4 255\n")
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    ppm = to_pgm(rgb)
    assert ppm.startswith(b"P6 5 4 255\n")
    assert len(ppm) == len(b"P6 5 4 255\n") + 4 * 5 * 3


def test_photoimage_roundtrip(root):
    u8 = normalize_u8(np.random.default_rng(0).random((32, 24)))
    ph = tk.PhotoImage(data=to_pgm(u8))
    assert (ph.width(), ph.height()) == (24, 32)


# ----------------------------------------------------------------- app --

@pytest.fixture()
def app(root, tmp_path):
    from rockstone.demo import make_rock_stack

    vol, _truth = make_rock_stack(n_slices=10, size=48, seed=3, cupping=0.3)
    path = tmp_path / "mini_stack.npy"
    np.save(path, vol.astype(np.float32))
    application = RockstoneApp(root)
    yield application, path
    # nothing to tear down: root is module-scoped and destroyed there


def test_load_and_browse(app):
    application, path = app
    application.load_path(path)
    assert application.kind == "volume"
    assert application.data.shape == (10, 48, 48)
    assert str(application.align_btn.cget("state")) == "normal"
    assert "cupping" in application.info.get()

    application.slice_var.set(4)
    application._on_slice_move()
    assert application.slice_lbl.cget("text") == "5 / 10"
    np.testing.assert_array_equal(application.displayed, application.data[4])


def test_load_missing_file_reports_error(app, monkeypatch):
    import rockstone.gui as gui_module

    application, _path = app
    shown = []
    monkeypatch.setattr(gui_module, "messagebox",
                        type("MB", (), {"showerror": staticmethod(
                            lambda *a, **k: shown.append(a))}))
    application.load_path(_path.parent / "does_not_exist.npy")
    assert len(shown) == 1
    assert application.status.get() == "Ready."


def test_align_and_deharden_tasks(app):
    application, path = app
    application.load_path(path)
    before = application.data.copy()

    application.run_align()
    assert _pump(application.root, application, "alignment finished"), \
        application.status.get()
    assert application.data.shape == before.shape
    assert not application._busy

    application.run_deharden()
    assert _pump(application.root, application,
                 "beam-hardening correction finished"), \
        application.status.get()
    assert np.all(np.isfinite(application.data))

    from rockstone.metrics import cupping_index

    cb = cupping_index(before[before.shape[0] // 2])
    ca = cupping_index(application.data[application.data.shape[0] // 2])
    assert ca < cb  # the cupping correction actually reduced cupping


def test_show_multichannel_plane(app):
    application, _path = app
    rng = np.random.default_rng(0)
    color = rng.random((20, 20, 3))
    application._show_plane(color)
    assert application.displayed is color
    assert application._photo.width() == 20


def test_deharden_button_label_for_projections(root, tmp_path):
    rng = np.random.default_rng(7)
    # (96, 16, 48): many projections vs detector columns -> the loader
    # heuristic classifies this as raw projections
    p = tmp_path / "proj.npy"
    np.save(p, rng.random((96, 16, 48)).astype(np.float32))
    application = RockstoneApp(root)
    application.load_path(p)
    assert application.kind == "projections"
    assert "Linearize" in application.deharden_btn.cget("text")


# ------------------------------------------------------------ color systems --

def test_colormap_luts_are_rgb():
    from rockstone.gui import COLORMAPS

    for name, lut in COLORMAPS.items():
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8
        assert list(COLORMAPS)[0] == "Gray"  # default listed first


def test_apply_colormap_endpoints_and_fallback():
    from rockstone.gui import apply_colormap

    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    vir = apply_colormap(ramp, "Viridis")
    assert vir.shape == (1, 256, 3)
    assert tuple(vir[0, 0]) == (68, 1, 84)      # viridis low anchor
    assert tuple(vir[0, -1]) == (253, 231, 37)  # viridis high anchor
    gray = apply_colormap(ramp, "Gray")
    np.testing.assert_array_equal(gray[0, :, 0], ramp[0])
    # unknown names fall back to plain gray
    np.testing.assert_array_equal(apply_colormap(ramp, "??")[0, :, 0],
                                  ramp[0])


def test_gui_colormap_switch_and_recolor(app):
    application, path = app
    application.load_path(path)
    assert application.colormap == "Gray"

    application.color_var.set("Jet")
    application._on_colormap_change()
    assert application.colormap == "Jet"
    # slider still works after a color change and keeps the new palette
    application.slice_var.set(2)
    application._on_slice_move()
    np.testing.assert_array_equal(application.displayed,
                                  application.data[2])


def test_new_colormaps_present_and_sane():
    from rockstone.gui import COLORMAPS

    for name in ("Asphalt", "Ice", "Copper", "Thermal", "Bone"):
        assert name in COLORMAPS
    asphalt = COLORMAPS["Asphalt"]
    # pavement: neutral grays (channels track each other), dark -> light
    r, g, b = asphalt[:, 0].astype(int), asphalt[:, 1].astype(int), \
        asphalt[:, 2].astype(int)
    assert np.all(np.abs(r - g) <= 8) and np.all(np.abs(g - b) <= 8)
    assert asphalt[0].max() < 15 and asphalt[-1].min() > 200
    assert np.all(np.diff(asphalt[:, 0].astype(int)) >= 0)  # monotonic


def test_gui_colormap_switch_without_data(root):
    application = RockstoneApp(root)
    application.color_var.set("Hot metal")
    application._on_colormap_change()  # must be a safe no-op on the view
    assert application.colormap == "Hot metal"


# ----------------------------------------------------------------- zoom --

def test_zoom_buttons_and_clamps(app):
    import rockstone.gui as gui_module

    application, path = app
    application.load_path(path)
    w0, h0 = application._photo.width(), application._photo.height()

    application._apply_zoom(2.0)
    assert (application._photo.width(), application._photo.height()) \
        == (2 * w0, 2 * h0)
    assert application.zoom_var.get() == "200%"

    application._apply_zoom(0.5)
    assert application._photo.width() == w0

    application._apply_zoom(1e9)
    assert application.zoom == gui_module.ZOOM_MAX  # clamped
    application._apply_zoom(1 / 1e9)
    assert application.zoom == gui_module.ZOOM_MIN

    application._zoom_fit()
    assert application._photo.width() <= application.canvas.winfo_width()
    assert application._photo.height() <= application.canvas.winfo_height()


def test_zoom_wheel_and_pan(app):
    import rockstone.gui as gui_module

    application, path = app
    application.load_path(path)

    class E:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    application._on_wheel(E(delta=120))
    z_up = application.zoom
    assert z_up > 1.0
    application._on_wheel(E(delta=-120))
    assert application.zoom == 1.0
    application._on_wheel_linux(E(num=4))
    assert application.zoom > 1.0
    application._on_wheel_linux(E(num=5))
    assert application.zoom == 1.0

    application._on_drag_start(E(x=10, y=5))
    application._on_drag_move(E(x=40, y=25))
    assert application._pan == (30, 20)
    assert int(application.canvas.place_info()["x"]) == 30
    application._on_drag_move(E(x=0, y=0))
    assert application._pan == (-10, -5)  # cumulative drag deltas
    # no data -> wheel is a no-op
    application2 = RockstoneApp(application.root)
    application2._on_wheel(E(delta=120))
    assert application2.zoom == 1.0


def test_zoom_survives_slice_and_colormap_changes(app):
    application, path = app
    application.load_path(path)
    w0 = application._photo.width()
    application._apply_zoom(2.0)

    application.slice_var.set(4)
    application._on_slice_move()
    assert application.zoom == 2.0
    assert application._photo.width() == 2 * w0

    cache = application._norm_cache
    application._redisplay()  # same plane+colormap -> cache reused
    assert application._norm_cache is cache
    application.color_var.set("Viridis")
    application._on_colormap_change()
    assert application._norm_cache is not cache  # recomputed
    assert application.zoom == 2.0               # but view kept


def test_zoom_resets_on_new_load(app):
    application, path = app
    application.load_path(path)
    application._apply_zoom(4.0)
    application.load_path(path)
    assert application.zoom == 1.0
    assert application.zoom_var.get() == "100%"


def test_normalize_u8_plane3_is_uint8():
    rng = np.random.default_rng(0)
    rgb = rng.random((8, 8, 3))
    out = normalize_u8_plane3(rgb)
    assert out.dtype == np.uint8  # PPM encoder requires uint8
    assert out.shape == (8, 8, 3)
