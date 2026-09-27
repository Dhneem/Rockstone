"""Tests for the menu bar (File / Edit / View / Process / Help), its
state syncing and the keyboard accelerators."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest

from rockstone.io import save_volume


def _command_labels(menu: tk.Menu) -> list[str]:
    n = menu.index("end")
    out = []
    for i in range(n + 1):
        if menu.type(i) == "command":
            out.append(menu.entrycget(i, "label"))
    return out


def _menu_state(menu: tk.Menu, label: str) -> str:
    return str(menu.entrycget(label, "state"))


@pytest.fixture()
def app(tk_root, tmp_path):
    from rockstone.demo import make_rock_stack
    from rockstone.gui import RockstoneApp

    vol, _ = make_rock_stack(n_slices=4, size=32, seed=0, cupping=0.3)
    src = tmp_path / "vol.tif"
    save_volume(vol, src)
    return RockstoneApp(tk_root), src


class TestMenuStructure:
    def test_five_menus(self, app):
        a, _ = app
        assert len(a._menus) == 5
        file_m, edit_m, view_m, proc_m, help_m = a._menus
        assert "Open file…" in _command_labels(file_m)
        assert "Open folder…" in _command_labels(file_m)
        assert "Save" in _command_labels(file_m)
        assert "Exit" in _command_labels(file_m)
        assert "Undo" in _command_labels(edit_m)
        assert "Redo" in _command_labels(edit_m)
        assert "Rotate 90° clockwise" in _command_labels(edit_m)
        assert "Crop…" in _command_labels(edit_m)
        assert "Physical dimensions…" in _command_labels(edit_m)
        assert "Zoom in" in _command_labels(view_m)
        assert "Slice map" in _command_labels(view_m)  # toggle entry
        assert "Compare…" in _command_labels(view_m)
        assert "Show difference" in _command_labels(view_m)
        assert "Align slices" in _command_labels(proc_m)
        assert "Remove beam hardening" in _command_labels(proc_m)
        assert "Background correction…" in _command_labels(proc_m)
        assert "Export current slice…" in _command_labels(proc_m)
        assert "About Rockstone…" in _command_labels(help_m)

    def test_accelerators(self, app):
        a, _ = app
        file_m, edit_m, view_m, proc_m, _ = a._menus
        assert file_m.entrycget("Open file…", "accelerator") == "Ctrl+O"
        assert file_m.entrycget("Save", "accelerator") == "Ctrl+S"
        assert edit_m.entrycget("Undo", "accelerator") == "Ctrl+Z"
        assert view_m.entrycget("Fit to window", "accelerator") == "Ctrl+F"
        assert proc_m.entrycget("Export current slice…",
                                "accelerator") == "Ctrl+E"

    def test_root_menu_is_set(self, app):
        a, _ = app
        assert str(a.root.cget("menu")) == str(a._menu_bar)
        assert isinstance(a._menu_bar, tk.Menu)


class TestMenuStates:
    def test_initially_disabled(self, app):
        a, _ = app
        file_m, edit_m, view_m, proc_m, _ = a._menus
        assert _menu_state(file_m, "Save") == "disabled"
        assert _menu_state(edit_m, "Undo") == "disabled"
        assert _menu_state(edit_m, "Crop…") == "disabled"
        assert _menu_state(proc_m, "Align slices") == "disabled"
        assert _menu_state(view_m, "Compare…") == "disabled"

    def test_enabled_after_load(self, app):
        a, src = app
        a.load_path(src)
        file_m, edit_m, view_m, proc_m, _ = a._menus
        assert _menu_state(file_m, "Save") == "normal"
        assert _menu_state(edit_m, "Crop…") == "normal"
        assert _menu_state(edit_m, "Rotate 90° clockwise") == "normal"
        assert _menu_state(proc_m, "Align slices") == "normal"
        assert _menu_state(view_m, "Slice map") == "normal"
        # undo still empty
        assert _menu_state(edit_m, "Undo") == "disabled"

    def test_undo_redo_entries_follow_history(self, app):
        a, src = app
        a.load_path(src)
        edit_m = a._menus[1]
        a.rotate_volume(1)
        assert _menu_state(edit_m, "Undo") == "normal"
        assert _menu_state(edit_m, "Redo") == "disabled"
        a.undo()
        assert _menu_state(edit_m, "Undo") == "disabled"
        assert _menu_state(edit_m, "Redo") == "normal"
        a.redo()
        assert _menu_state(edit_m, "Undo") == "normal"
        assert _menu_state(edit_m, "Redo") == "disabled"

    def test_show_difference_entry_follows_compare(self, app):
        a, src = app
        a.load_path(src)
        view_m = a._menus[2]
        assert _menu_state(view_m, "Show difference") == "disabled"
        # simulate entering compare mode
        a._compare_mode = "side"
        a.diff_btn.configure(state=tk.NORMAL)
        a._sync_menu_states()
        assert _menu_state(view_m, "Show difference") == "normal"
        a._exit_compare()
        assert _menu_state(view_m, "Show difference") == "disabled"

    def test_busy_disables_menus(self, app):
        a, src = app
        a.load_path(src)
        proc_m = a._menus[3]
        assert _menu_state(proc_m, "Align slices") == "normal"
        a._set_busy(True, "working")
        assert _menu_state(proc_m, "Align slices") == "disabled"
        a._set_busy(False)
        assert _menu_state(proc_m, "Align slices") == "normal"


class TestMenuCommands:
    def test_about_dialog(self, app, monkeypatch):
        from rockstone import gui as gui_module

        a, _ = app
        shown = []
        monkeypatch.setattr(gui_module.messagebox, "showinfo",
                            staticmethod(lambda *args, **k:
                                         shown.append(args)))
        help_m = a._menus[4]
        help_m.invoke("About Rockstone…")
        assert shown
        assert "Rockstone" in shown[0][1]

    def test_exit_entry_is_wired(self, app):
        # never *invoke* Exit in tests: it would destroy the shared
        # test root; just verify the command is attached
        a, _ = app
        cmd = a._menus[0].entrycget("Exit", "command")
        assert cmd and "_on_close" in cmd

    def test_menu_invoke_rotates(self, app):
        a, src = app
        a.load_path(src)
        before = a.data.copy()
        a._menus[1].invoke("Rotate 90° clockwise")
        assert a.data.shape == (4, 32, 32)  # square data: same shape
        np.testing.assert_array_equal(a.data, np.rot90(before, k=1,
                                                       axes=(1, 2)))


class TestKeyboardAccelerators:
    def test_all_accelerator_bindings_registered(self, app):
        a, _ = app
        for seq in ("<Control-o>", "<Control-O>", "<Control-s>",
                    "<Control-S>", "<Control-z>", "<Control-y>",
                    "<Control-e>", "<Control-m>", "<Control-f>",
                    "<Control-plus>", "<Control-minus>",
                    "<Control-equal>", "<Escape>"):
            assert a.root.bind(seq), f"no binding for {seq}"

    def test_open_dialog_shows_file_dialog(self, app, monkeypatch):
        from rockstone import gui as gui_module

        a, _src = app
        picked = {}
        monkeypatch.setattr(
            gui_module.filedialog, "askopenfilenames",
            staticmethod(lambda **k: picked.update(k) or ()))
        a.open_dialog()  # what the Ctrl+O accelerator invokes
        assert "filetypes" in picked  # dialog was opened

    def test_export_shortcut_action_opens_dialog(self, app):
        # what the Ctrl+E accelerator invokes (event dispatch of
        # synthetic key events is focus-dependent under the withdrawn
        # shared test root, so call the handler directly)
        a, src = app
        a.load_path(src)
        a.export_slice_as()
        dialogs = [w for w in a.root.winfo_children()
                   if isinstance(w, tk.Toplevel)
                   and "export" in w.title().lower()]
        assert dialogs
        dialogs[0].destroy()

    def test_ctrl_m_toggles_slice_map(self, app):
        # what the Ctrl+M accelerator invokes (synthetic key events are
        # focus-dependent under the withdrawn test root)
        a, src = app
        a.load_path(src)
        a._toggle_map()
        end = __import__("time").time() + 10
        while __import__("time").time() < end:
            a.root.update()
            if a._map_win is not None and a._map_cells:
                break
            __import__("time").sleep(0.02)
        assert a._map_win is not None and a._map_cells
        a._toggle_map()
        assert a._map_win is None
        a._close_map()

    def test_ctrl_plus_zooms_in(self, app):
        a, src = app
        a.load_path(src)
        before = a.zoom
        a.root.event_generate("<Control-plus>")
        a.root.update()
        assert a.zoom > before


class TestToolbarLayout:
    def test_separators_present(self, app):
        a, _ = app
        seps = [w for w in a.root.winfo_children()
                if isinstance(w, ttk.Frame)
                for c in w.winfo_children()
                if isinstance(c, ttk.Separator)]
        assert len(seps) >= 5  # open | process | compare | zoom | edit | save

    def test_side_panel_has_labeled_groups(self, app):
        a, _ = app
        texts = []
        stack = list(a.root.winfo_children())
        while stack:
            w = stack.pop()
            stack.extend(w.winfo_children())
            if isinstance(w, ttk.LabelFrame):
                texts.append(str(w.cget("text")))
        assert "Appearance" in texts
        assert "Export & compare" in texts
        assert "Slice" in texts
