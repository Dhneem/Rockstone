"""Tests for Undo/Redo (snapshot history with identity-based dirty flag)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from rockstone.io import save_volume


@pytest.fixture()
def hist_app(tk_root, tmp_path, monkeypatch):
    import rockstone.gui as gui_module
    from rockstone.demo import make_rock_stack
    from rockstone.gui import RockstoneApp

    vol, _ = make_rock_stack(n_slices=5, size=32, seed=0, cupping=0.35)
    src = tmp_path / "vol.tif"
    save_volume(vol, src)

    monkeypatch.setattr(gui_module, "messagebox", type("MB", (), {
        "showinfo": staticmethod(lambda *a, **k: None),
        "showerror": staticmethod(lambda *a, **k: None),
        "askyesnocancel": staticmethod(lambda *a, **k: True)}))
    app = RockstoneApp(tk_root)
    app.load_path(src)
    return app


def _pump(root, cond, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_undo_redo_full_cycle(hist_app):
    app = hist_app
    v0 = app.data
    assert str(app.undo_btn.cget("state")) == "disabled"
    assert str(app.redo_btn.cget("state")) == "disabled"

    app.run_deharden()
    assert _pump(app.root, lambda: "correction finished" in app.status.get())
    v1 = app.data
    assert v1 is not v0

    app.run_align()
    assert _pump(app.root, lambda: "alignment finished" in app.status.get())
    v2 = app.data

    # undo align
    app.undo()
    assert _pump(app.root, lambda: "undid align slices" in app.status.get())
    assert app.data is v1
    assert str(app.redo_btn.cget("state")) == "normal"

    # undo deharden -> back to the original; dirty clears automatically
    app.undo()
    assert _pump(app.root, lambda: "undid deharden" in app.status.get())
    assert app.data is v0
    assert app._dirty is False
    assert "*" not in app.root.title()
    assert str(app.undo_btn.cget("state")) == "disabled"

    # redo both
    app.redo()
    assert app.data is v1
    app.redo()
    assert app.data is v2
    assert app._dirty is True
    assert str(app.redo_btn.cget("state")) == "disabled"


def test_new_modification_clears_redo(hist_app):
    app = hist_app
    v0 = app.data
    app.run_deharden()
    assert _pump(app.root, lambda: "correction finished" in app.status.get())
    app.undo()
    assert _pump(app.root, lambda: "undid deharden" in app.status.get())
    assert app._redo_stack
    app.run_align()
    assert _pump(app.root, lambda: "alignment finished" in app.status.get())
    assert not app._redo_stack  # branching: redo fork is dropped
    assert str(app.redo_btn.cget("state")) == "disabled"


def test_undo_after_save_clears_dirty(hist_app):
    app = hist_app
    app.run_deharden()
    assert _pump(app.root, lambda: "correction finished" in app.status.get())
    app.save()
    assert _pump(app.root, lambda: "saved vol.tif" in app.status.get())
    saved_state = app.data
    app.run_deharden()
    assert _pump(app.root, lambda: "correction finished" in app.status.get())
    assert app._dirty is True
    app.undo()
    assert _pump(app.root, lambda: "undid" in app.status.get())
    assert app.data is saved_state  # identity match -> not dirty
    assert app._dirty is False
    assert "*" not in app.root.title()


def test_history_cap(hist_app):
    from rockstone.gui import HISTORY_MAX

    app = hist_app
    for _ in range(HISTORY_MAX + 4):
        app.run_deharden()
        assert _pump(app.root, lambda: "correction finished"
                     in app.status.get())
    assert len(app._undo_stack) == HISTORY_MAX
    # undo everything the cap allows without error
    for _ in range(HISTORY_MAX):
        app.undo()
    assert not app._undo_stack
    assert str(app.undo_btn.cget("state")) == "disabled"


def test_history_cleared_on_load(hist_app):
    app = hist_app
    app.run_deharden()
    assert _pump(app.root, lambda: "correction finished" in app.status.get())
    assert app._undo_stack
    app.load_path(app.path)
    assert not app._undo_stack
    assert not app._redo_stack
    assert app._dirty is False


def test_undo_redo_noops_without_data(tk_root, monkeypatch):
    import rockstone.gui as gui_module
    from rockstone.gui import RockstoneApp

    monkeypatch.setattr(gui_module, "messagebox", type("MB", (), {
        "showinfo": staticmethod(lambda *a, **k: None),
        "showerror": staticmethod(lambda *a, **k: None)}))
    app = RockstoneApp(tk_root)
    app.undo()  # must be silent no-ops
    app.redo()
    assert app.data is None
