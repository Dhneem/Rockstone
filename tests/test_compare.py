"""Tests for dataset comparison (library, CLI and GUI flow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rockstone.compare import compare_volumes, crop_to_common


@pytest.fixture(scope="module")
def stack_pair(tmp_path_factory):
    from rockstone.demo import make_rock_stack
    from rockstone.io import save_volume

    vol, _ = make_rock_stack(n_slices=6, size=48, seed=0, cupping=0.3)
    rng = np.random.default_rng(1)
    noisy = vol + rng.normal(0, 0.01, vol.shape)
    d = tmp_path_factory.mktemp("cmp")
    pa, pb = d / "a.tif", d / "b.tif"
    save_volume(vol, pa)
    save_volume(noisy, pb)
    return vol, noisy, pa, pb


def test_identical_detection(stack_pair):
    vol, _noisy, _pa, _pb = stack_pair
    res = compare_volumes(vol, vol.copy())
    assert res.identical and res.shapes_match
    assert res.psnr == float("inf")
    assert "identical" in res.summary()


def test_noise_metrics(stack_pair):
    vol, noisy, _pa, _pb = stack_pair
    res = compare_volumes(vol, noisy)
    assert not res.identical
    assert res.correlation > 0.99
    assert res.psnr > 20.0  # 0.01 sigma noise on a ~[0, 1] signal
    assert 0 < res.mean_abs_diff < 0.05
    assert res.max_abs_diff >= res.mean_abs_diff
    assert len(res.worst_slices) == 3


def test_shape_mismatch_crops_to_overlap(stack_pair):
    vol, _noisy, _pa, _pb = stack_pair
    cut = vol[:, 5:-5, 3:-3]
    a, b = crop_to_common(vol, cut)
    assert a.shape == b.shape == (6, 38, 42)
    res = compare_volumes(vol, cut)
    assert res.shapes_match is False
    assert res.n_compared == 6
    assert res.identical is False  # different shapes -> not "identical"
    report = res.as_report()
    assert "overlapping region" in report


def test_report_contents(stack_pair):
    _vol, _noisy, pa, pb = stack_pair
    from rockstone.io import load_volume

    res = compare_volumes(load_volume(pa), load_volume(pb))
    report = res.as_report()
    for key in ("shapes match", "slices compared", "max |diff|",
                "rmse", "psnr", "correlation"):
        assert key in report


def test_cli_compare(stack_pair, capsys):
    from rockstone.cli import main

    _vol, _noisy, pa, pb = stack_pair
    rc = main(["compare", str(pa), str(pb)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "psnr" in out and str(pa) in out


def test_gui_compare_flow(stack_pair, tmp_path, monkeypatch, tk_root):
    pytest.importorskip("tkinter")

    import rockstone.gui as gui_module
    from rockstone.gui import RockstoneApp

    vol, noisy, _pa, _pb = stack_pair
    errors = []
    monkeypatch.setattr(gui_module, "messagebox", type("MB", (), {
        "showinfo": staticmethod(lambda *a, **k: None),
        "showerror": staticmethod(lambda *a, **k: errors.append(a))}))

    root = tk_root  # shared session root (see tests/conftest.py)
    from rockstone.io import save_volume

    src = tmp_path / "main.tif"
    other = tmp_path / "other.tif"
    save_volume(vol, src)
    save_volume(noisy, other)

    app = RockstoneApp(root)
    app.load_path(src)

    class V:
        def __init__(self, v):
            self._v = v

        def get(self):
            return self._v

    app._do_compare(None, V(str(other)))
    import time

    end = time.time() + 30
    while time.time() < end:
        root.update()
        if app._compare_result is not None:
            break
        time.sleep(0.02)
    assert not errors
    assert app._compare_mode == "side"
    assert app._compare_b is not None
    assert not app._compare_result.identical

    # side-by-side canvas: twice the width plus a gap
    assert app._photo.width() == 2 * vol.shape[2] + 4

    # toggle to difference and back
    app._toggle_diff()
    assert app._compare_mode == "diff"
    assert "side-by-side" in app.diff_btn.cget("text")
    app._toggle_diff()
    assert app._compare_mode == "side"

    # slider follows in compare mode
    app.slice_var.set(3)
    app._on_slice_move()
    app._exit_compare()
    assert app._compare_mode is None and app._compare_b is None
    assert str(app.diff_btn.cget("state")) == "disabled"
