"""Tests for background correction: library, CLI, pipeline and GUI."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rockstone.background import BackgroundResult, correct_background
from rockstone.io import load_volume, save_volume


def _disk_with_ramp(n=64, nz=3, pedestal=0.08, lo=0.05, hi=0.20,
                    value=0.5):
    """Flat sample disk on a linearly increasing background ramp."""
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    disk = (r < n * 0.35).astype(np.float64) * value
    ramp = np.linspace(lo, hi, n)[None, :] + pedestal
    return np.stack([disk + ramp] * nz)


# ------------------------------------------------------------- library --

def test_offset_flattens_background_and_keeps_sample():
    vol = _disk_with_ramp()
    corr, res = correct_background(vol)
    assert isinstance(res, BackgroundResult)
    assert res.mode == "offset" and res.per_slice is True
    # background: the ramp spread collapses (block-surface fit leaves a
    # small bilinear-interpolation residual, ~90% reduction)
    assert res.background_std_after < 0.25 * res.background_std_before
    # air level ends near zero
    assert abs(res.air_after) < 0.02
    # the sample interior keeps its value (ramp removed everywhere)
    n = vol.shape[-1]
    assert corr[0, n // 2, n // 2] == pytest.approx(0.5, abs=0.05)
    # extreme corners: background removed
    assert abs(corr[:, :4, :4].mean()) < 0.02


def test_input_not_modified():
    vol = _disk_with_ramp()
    ref = vol.copy()
    correct_background(vol)
    np.testing.assert_array_equal(vol, ref)


def test_divide_mode_flattens_multiplicative_shading():
    n, nz = 64, 2
    shade = np.linspace(0.8, 1.2, n)[None, :]
    disk_value = 1.0  # sample and background differ only by shading
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    sample = (r < n * 0.35).astype(np.float64) * disk_value + 0.2
    vol = np.stack([(sample + shade) ] * nz)
    corr, res = correct_background(vol, mode="divide", dilate=0)
    assert res.mode == "divide"
    assert res.background_std_after < res.background_std_before


def test_shared_surface_option():
    vol = _disk_with_ramp()
    corr, res = correct_background(vol, per_slice=False)
    assert res.per_slice is False
    assert res.background_std_after < 0.25 * res.background_std_before


def test_air_levels_reported_per_slice():
    vol = _disk_with_ramp(nz=4)
    _, res = correct_background(vol)
    assert res.air_levels.shape == (4,)
    assert res.air_before == pytest.approx(np.median(res.air_levels))
    # the pedestal is 0.08 on the ramp [0.05, 0.20] -> air ~0.13
    assert res.air_before == pytest.approx(0.13, abs=0.03)


def test_nan_preserved():
    vol = _disk_with_ramp(nz=2, n=48)
    vol[0, 0, 0] = np.nan
    corr, _ = correct_background(vol)
    assert np.isnan(corr[0, 0, 0])
    assert np.isfinite(corr[0, 1, 1])


def test_integer_input_promoted():
    vol = _disk_with_ramp(n=48).astype(np.uint8)
    corr, _ = correct_background(vol)
    assert np.issubdtype(corr.dtype, np.floating)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="unknown mode"):
        correct_background(_disk_with_ramp(n=48), mode="bogus")
    with pytest.raises(ValueError, match="3-D"):
        correct_background(np.zeros((4, 4)))


def test_flattish_image_is_stable():
    # no sample contrast: mask falls back to everything, correction ~ no-op
    vol = np.full((2, 32, 32), 0.3)
    corr, res = correct_background(vol)
    np.testing.assert_allclose(corr, 0.0, atol=1e-9)
    assert abs(res.air_after) < 1e-6


# ----------------------------------------------------------------- CLI --

def test_cli_background_subcommand(tmp_path, capsys):
    from rockstone.cli import main

    vol = _disk_with_ramp(nz=3, n=64)
    src = tmp_path / "in.tif"
    save_volume(vol, src)
    out = tmp_path / "out.tif"
    rc = main(["background", str(src), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    text = capsys.readouterr().out
    assert "air:" in text and "background spread:" in text
    corr = load_volume(out)
    assert abs(corr[:, :4, :4].mean()) < 0.05


def test_cli_background_divide_mode(tmp_path):
    from rockstone.cli import main

    n, nz = 48, 2
    vol = np.full((nz, n, n), 0.4)
    src = tmp_path / "in.tif"
    save_volume(vol, src)
    rc = main(["background", str(src), "-o", str(tmp_path / "d.tif"),
               "--mode", "divide", "--shared"])
    assert rc == 0
    # constant input, divide mode: stays ~constant (surface ~ 0.4)
    corr = load_volume(tmp_path / "d.tif")
    np.testing.assert_allclose(corr, 1.0, atol=0.05)


def test_cli_run_includes_background_stage(tmp_path, capsys):
    from rockstone.cli import main

    vol = _disk_with_ramp(nz=3, n=64)
    src = tmp_path / "in.tif"
    save_volume(vol, src)
    out = tmp_path / "pipe.tif"
    rc = main(["run", str(src), "-o", str(out), "--no-align",
               "--no-deharden", "--with-background"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "background_offset" in text
    assert "air level" in text
    assert out.exists()


def test_cli_run_background_opt_in(tmp_path, capsys):
    from rockstone.cli import main

    vol = _disk_with_ramp(nz=3, n=64)
    src = tmp_path / "in.tif"
    save_volume(vol, src)

    # default: no background stage (pre-existing workflows unchanged)
    rc = main(["run", str(src), "-o", str(tmp_path / "nb.tif"),
               "--no-align", "--no-deharden"])
    assert rc == 2
    assert "nothing to do" in capsys.readouterr().err

    rc = main(["run", str(src), "-o", str(tmp_path / "nb2.tif"),
               "--no-align", "--no-deharden", "--with-background"])
    assert rc == 0
    steps_line = next(line for line in capsys.readouterr().out.splitlines()
                      if line.startswith("steps"))
    assert "background" in steps_line

    # and deharden-only runs keep working without the stage
    rc = main(["run", str(src), "-o", str(tmp_path / "nb3.tif"),
               "--no-align"])
    assert rc == 0
    steps_line = next(line for line in capsys.readouterr().out.splitlines()
                      if line.startswith("steps"))
    assert "background" not in steps_line


# ------------------------------------------------------------ pipeline --

def test_pipeline_background_stage_and_report(tmp_path):
    from rockstone.pipeline import process

    vol = _disk_with_ramp(nz=3, n=64)
    src = tmp_path / "in.tif"
    save_volume(vol, src)
    report = process(src, tmp_path / "p.tif", align=False, deharden=False,
                     background=True)
    assert report.steps_run == ["background_offset"]
    assert report.background is not None
    assert report.background.air_before == pytest.approx(0.13, abs=0.05)
    assert "air level" in report.summary()


def test_pipeline_background_divide_options(tmp_path):
    from rockstone.pipeline import process

    n = 48
    vol = np.full((2, n, n), 0.4)
    src = tmp_path / "in.tif"
    save_volume(vol, src)
    report = process(src, tmp_path / "p2.tif", align=False,
                     deharden=False, background=True,
                     background_options={"mode": "divide"})
    assert report.steps_run == ["background_divide"]


def test_pipeline_all_stages_disabled_raises(tmp_path):
    from rockstone.pipeline import process

    src = tmp_path / "in.tif"
    save_volume(_disk_with_ramp(nz=2, n=48), src)
    with pytest.raises(ValueError, match="nothing to do"):
        process(src, tmp_path / "x.tif", align=False, deharden=False,
                background=False)


def test_pipeline_background_is_opt_in(tmp_path):
    from rockstone.pipeline import process

    src = tmp_path / "in.tif"
    save_volume(_disk_with_ramp(nz=2, n=48), src)
    # background stays off unless explicitly requested (pre-existing
    # workflows keep their exact results)
    report = process(src, tmp_path / "off.tif", align=False,
                     deharden=False, background=True)
    assert "background_offset" in report.steps_run
    report2 = process(src, tmp_path / "off2.tif", deharden=False)
    assert report2.steps_run == ["align_slices"]
    assert report2.background is None


# ----------------------------------------------------------------- GUI --

def _pump(root, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def bg_app(tk_root, tmp_path):
    from rockstone.demo import make_rock_stack
    from rockstone.gui import RockstoneApp

    vol, _ = make_rock_stack(n_slices=5, size=48, seed=0, cupping=0.3)
    src = tmp_path / "vol.tif"
    save_volume(vol, src)
    return RockstoneApp(tk_root), src


def test_gui_background_button_states(bg_app):
    app, src = bg_app
    assert str(app.bg_btn.cget("state")) == "disabled"
    app.load_path(src)
    assert str(app.bg_btn.cget("state")) == "normal"


def _find_buttons(widget):
    """All ttk.Buttons under ``widget``, at any depth."""
    from tkinter import ttk

    out = []
    for w in widget.winfo_children():
        if isinstance(w, ttk.Button):
            out.append(w)
        out.extend(_find_buttons(w))
    return out


def test_gui_background_flow(bg_app, monkeypatch):
    from rockstone import gui as gui_module
    from rockstone.metrics import cupping_index

    app, src = bg_app
    app.load_path(src)
    errors = []
    monkeypatch.setattr(gui_module.messagebox, "showerror",
                        staticmethod(lambda *a, **k: errors.append(a)))
    before = app.data.copy()
    before_obj = app.data  # identity: undo must restore this object
    app.run_background()  # opens the dialog
    # find the Run button in the dialog and click it
    import tkinter as tk
    from tkinter import ttk

    dlg = next(w for w in app.root.winfo_children()
               if isinstance(w, tk.Toplevel)
               and "background" in w.title().lower())
    run_btn = next(b for b in _find_buttons(dlg)
                   if str(b.cget("text")) == "Run")
    run_btn.invoke()
    assert _pump(app.root, lambda: "background correction finished"
                 in app.status.get()), app.status.get()
    assert not errors
    assert app.data.shape == before.shape
    assert not np.array_equal(app.data, before)
    assert app._processed is True and app._dirty is True
    assert "*" in app.root.title()
    # undo works like every other modification
    app.undo()
    assert app.data is before_obj
    np.testing.assert_array_equal(app.data, before)
    assert "undid background correction" in app.status.get()
    # dialog is gone
    assert not bool(dlg.winfo_exists())


def test_gui_background_busy_guard(bg_app):
    app, src = bg_app
    app.load_path(src)
    app._set_busy(True, "working")
    app.run_background()  # must be a no-op
    assert app.status.get() == "working"
