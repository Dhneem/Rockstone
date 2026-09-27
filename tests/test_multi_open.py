"""Tests for opening many image files at once (multi-selection stacking)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from rockstone.io import load_stack_from_files, load_volume


@pytest.fixture()
def slice_files(tmp_path):
    import imageio.v2 as iio

    values = [("slice_1.tif", 1), ("slice_2.png", 2),
              ("slice_10.tif", 10), ("slice_20.jpg", 20)]
    for name, v in values:
        arr = np.full((6, 8), v, dtype=np.uint8)
        if name.endswith(".tif"):
            tifffile.imwrite(tmp_path / name, arr)
        else:
            iio.imwrite(tmp_path / name, arr)
    return {name: tmp_path / name for name, _ in values}


def test_natural_order_regardless_of_selection_order(slice_files):
    vol = load_stack_from_files([
        slice_files["slice_10.tif"], slice_files["slice_1.tif"],
        slice_files["slice_20.jpg"], slice_files["slice_2.png"]])
    assert vol.shape == (4, 6, 8)
    assert [int(s[0, 0]) for s in vol] == [1, 2, 10, 20]


def test_mixed_extensions_and_color(slice_files, tmp_path):
    import imageio.v2 as iio

    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 7
    iio.imwrite(tmp_path / "slice_3_color.png", rgb)
    vol = load_stack_from_files([slice_files["slice_2.png"],
                                 tmp_path / "slice_3_color.png",
                                 slice_files["slice_1.tif"]])
    # loaders return grayscale volumes: color input is reduced like the
    # RGBA-TIFF path does (gray slices are promoted to match, then the
    # stack is averaged back to one channel)
    assert vol.shape == (3, 6, 8)
    assert int(vol[0, 0, 0]) == 1
    assert int(vol[1, 0, 0]) == 7 // 3  # red 7 -> gray mean
    assert int(vol[2, 0, 0]) == 2


def test_multipage_tiff_expands_to_pages(slice_files, tmp_path):
    pages = np.stack([np.full((6, 8), 50, dtype=np.uint8),
                      np.full((6, 8), 60, dtype=np.uint8)])
    tifffile.imwrite(tmp_path / "page_a.tif", pages)
    vol = load_stack_from_files([tmp_path / "page_a.tif",
                                 slice_files["slice_1.tif"]])
    assert [int(s[0, 0]) for s in vol] == [50, 60, 1]


def test_shape_mismatch_names_offender(slice_files, tmp_path):
    tifffile.imwrite(tmp_path / "big.tif", np.zeros((10, 12), dtype=np.uint8))
    with pytest.raises(ValueError, match="big.tif"):
        load_stack_from_files([slice_files["slice_1.tif"],
                               tmp_path / "big.tif"])


def test_empty_selection_and_missing_file(slice_files):
    with pytest.raises(ValueError):
        load_stack_from_files([])
    with pytest.raises(FileNotFoundError):
        load_stack_from_files([slice_files["slice_1.tif"].parent
                               / "nope.tif"])


def test_gui_load_paths_multi_selection(slice_files, monkeypatch, tk_root):
    import rockstone.gui as gui_module
    from rockstone.gui import RockstoneApp

    errors = []
    monkeypatch.setattr(gui_module, "messagebox", type("MB", (), {
        "showinfo": staticmethod(lambda *a, **k: None),
        "showerror": staticmethod(lambda *a, **k: errors.append(a))}))
    app = RockstoneApp(tk_root)
    app.load_paths([slice_files["slice_10.tif"], slice_files["slice_1.tif"],
                    slice_files["slice_2.png"]])
    assert not errors
    assert app.data.shape == (3, 6, 8)
    assert [int(s[0, 0]) for s in app.data] == [1, 2, 10]
    assert app.multi_paths is not None and len(app.multi_paths) == 3
    assert app.path is None  # no single source file
    assert "3 files (stacked)" in app.info.get()
    assert "3 slices" in app.status.get()

    # single-path call behaves like the classic load_path
    app.load_path(slice_files["slice_1.tif"])
    assert app.path is not None and app.multi_paths is None
    assert app.data.shape == (1, 6, 8)


def test_cli_info_on_stacked_volume(tmp_path):
    from rockstone.cli import main

    p = tmp_path / "v.npy"
    np.save(p, np.zeros((5, 8, 8), dtype=np.float32))
    import io as _io
    import contextlib

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["info", str(p)])
    assert rc == 0
    assert "(5, 8, 8)" in buf.getvalue()
