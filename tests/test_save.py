"""Tests for Save / Save As in the GUI and the extended save_volume."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from rockstone.io import load_volume, save_volume


def test_save_volume_npz(tmp_path):
    vol = np.random.default_rng(0).random((4, 8, 8)).astype(np.float32)
    p = tmp_path / "v.npz"
    save_volume(vol, p)
    np.testing.assert_array_equal(load_volume(p), vol)  # lossless


def test_save_volume_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError, match="unsupported output format"):
        save_volume(np.zeros((2, 4, 4)), tmp_path / "v.png")


@pytest.fixture()
def save_app(tk_root, tmp_path, monkeypatch):
    import rockstone.gui as gui_module
    from rockstone.demo import make_rock_stack
    from rockstone.gui import RockstoneApp

    vol, _ = make_rock_stack(n_slices=5, size=32, seed=0, cupping=0.3)
    src = tmp_path / "vol.tif"
    save_volume(vol, src)

    dialogs = {"askyesnocancel": [], "asksaveas": []}
    monkeypatch.setattr(gui_module, "messagebox", type("MB", (), {
        "showinfo": staticmethod(lambda *a, **k: None),
        "showerror": staticmethod(lambda *a, **k: None),
        "askyesnocancel": staticmethod(
            lambda *a, **k: dialogs["askyesnocancel"].append(a) or True)}))
    monkeypatch.setattr(
        gui_module.filedialog, "asksaveasfilename",
        staticmethod(lambda **k: dialogs["asksaveas"].append(k) or ""))

    app = RockstoneApp(tk_root)
    return app, src, dialogs


def _pump(root, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_save_button_states_and_title(save_app):
    app, src, _dialogs = save_app
    assert str(app.save_btn.cget("state")) == "disabled"
    assert str(app.save_as_btn.cget("state")) == "disabled"
    assert "*" not in app.root.title()
    app.load_path(src)
    assert str(app.save_btn.cget("state")) == "normal"
    assert str(app.save_as_btn.cget("state")) == "normal"
    assert app._dirty is False
    assert "*" not in app.root.title()


def test_full_save_cycle(save_app):
    app, src, dialogs = save_app
    app.load_path(src)
    assert str(app.save_btn.cget("state")) == "normal"
    assert app._dirty is False
    assert "*" not in app.root.title()
    from rockstone.metrics import cupping_index

    original = cupping_index(app.data[2])

    app.run_deharden()
    assert _pump(app.root, lambda: "finished" in app.status.get())
    assert app._dirty is True
    assert "*" in app.root.title()

    # plain Save overwrites the loaded file
    app.save()
    assert _pump(app.root, lambda: "saved vol.tif" in app.status.get())
    assert app._dirty is False
    assert "*" not in app.root.title()

    saved = load_volume(src)
    assert cupping_index(saved[2]) < original  # the processed data landed


def test_save_as_new_file_and_retarget(save_app):
    app, src, dialogs = save_app
    app.load_path(src)
    app.run_deharden()
    assert _pump(app.root, lambda: "finished" in app.status.get())

    dest = src.parent / "out" / "processed.npz"
    dialogs["asksaveas"].clear()
    import rockstone.gui as gui_module

    gui_module.filedialog.asksaveasfilename = staticmethod(
        lambda **k: dialogs["asksaveas"].append(k) or str(dest))
    app.save_as()
    assert _pump(app.root, lambda: "saved processed.npz" in app.status.get())
    assert dest.exists()
    assert app.path == dest and app.multi_paths is None
    np.testing.assert_allclose(load_volume(dest), app.data)

    # a later plain Save now retargets the Save As file
    app.run_align()
    assert _pump(app.root, lambda: "alignment finished" in app.status.get())
    app.save()
    assert _pump(app.root, lambda: "saved processed.npz" in app.status.get())


def test_save_as_fallback_for_multi_open(save_app, tmp_path, monkeypatch):
    import tifffile

    import rockstone.gui as gui_module

    app, _src, dialogs = save_app
    a = tmp_path / "s1.tif"
    b = tmp_path / "s2.tif"
    tifffile.imwrite(a, np.zeros((8, 8), dtype=np.uint8))
    tifffile.imwrite(b, np.full((8, 8), 5, dtype=np.uint8))
    app.load_paths([a, b])
    assert app.multi_paths is not None and app.path is None

    dest = tmp_path / "stack.tif"
    gui_module.filedialog.asksaveasfilename = staticmethod(
        lambda **k: str(dest))
    app.save()  # no single target -> must fall back to Save As
    assert _pump(app.root, lambda: "saved stack.tif" in app.status.get())
    np.testing.assert_array_equal(load_volume(dest), app.data)


def test_close_guard_confirm_discard_cancel(save_app):
    import rockstone.gui as gui_module

    app, src, dialogs = save_app
    app.load_path(src)

    # clean data: no dialog, proceeds
    assert app._confirm_discard() is True
    assert dialogs["askyesnocancel"] == []

    # modified data + user chooses "yes": dialog shown, save triggered
    app.run_deharden()
    assert _pump(app.root, lambda: "finished" in app.status.get())
    assert app._confirm_discard() is True
    assert len(dialogs["askyesnocancel"]) == 1
    assert _pump(app.root, lambda: "saved vol.tif" in app.status.get())

    # modified data + user cancels: blocked
    app.run_deharden()
    assert _pump(app.root, lambda: "finished" in app.status.get())
    gui_module.messagebox.askyesnocancel = staticmethod(
        lambda *a, **k: None)
    assert app._confirm_discard() is False


def test_ctrl_s_bindings_registered(save_app):
    app, _src, _dialogs = save_app
    assert app.root.bind("<Control-s>")
    assert app.root.bind("<Control-S>")
