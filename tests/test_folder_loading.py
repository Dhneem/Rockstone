"""Tests for opening folders of slice images (recursion + natural order)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from rockstone.io import _collect_image_files, load_volume


def _write_tiffs(folder: Path, names_values: list[tuple[str, int]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in names_values:
        tifffile.imwrite(folder / f"{name}.tif",
                         np.full((6, 8), value, dtype=np.uint8))


def _values(volume: np.ndarray) -> list[int]:
    return [int(s[0, 0]) for s in volume]


def test_natural_order_numeric_names(tmp_path):
    # lexicographic would give slice_1, slice_10, slice_2 ...
    _write_tiffs(tmp_path, [("slice_1", 1), ("slice_2", 2),
                            ("slice_10", 10), ("slice_20", 20)])
    vol = load_volume(tmp_path)
    assert _values(vol) == [1, 2, 10, 20]


def test_recursive_folder_gathers_subfolders(tmp_path):
    _write_tiffs(tmp_path / "run_a", [("slice_1", 1), ("slice_2", 2)])
    _write_tiffs(tmp_path / "run_b" / "deep", [("slice_3", 3)])
    _write_tiffs(tmp_path, [("slice_0", 0)])
    vol = load_volume(tmp_path)  # recursive by default
    # full-path natural order: subfolder names group first ("run_..."
    # sorts before "slice_..."), numeric order inside each folder
    assert _values(vol) == [1, 2, 3, 0]
    files = _collect_image_files(tmp_path, recursive=True)
    assert files[0] == tmp_path / "run_a" / "slice_1.tif"
    assert files[-1] == tmp_path / "slice_0.tif"


def test_non_recursive_top_level_only(tmp_path):
    _write_tiffs(tmp_path / "sub", [("slice_5", 5)])
    _write_tiffs(tmp_path, [("slice_1", 1)])
    vol = load_volume(tmp_path, recursive=False)
    assert vol.shape[0] == 1


def test_unreadable_file_skipped_with_warning(tmp_path):
    import warnings as _warnings

    _write_tiffs(tmp_path, [("slice_1", 1), ("slice_2", 2)])
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "slice_3.tif").write_bytes(b"corrupt garbage")
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        vol = load_volume(tmp_path)
    assert vol.shape[0] == 2  # corrupt slice skipped, good ones kept
    assert _values(vol) == [1, 2]
    assert any("unreadable" in str(w.message) for w in caught)


def test_empty_folder_raises(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError):
        load_volume(tmp_path)


def test_mixed_extensions_natural_order(tmp_path):
    from rockstone.io import _read_generic_image  # ensure importable

    _write_tiffs(tmp_path, [("slice_1", 1)])
    import imageio.v2 as iio

    iio.imwrite(tmp_path / "slice_2.png",
                np.full((6, 8), 2, dtype=np.uint8))
    iio.imwrite(tmp_path / "slice_10.jpg",
                np.full((6, 8), 10, dtype=np.uint8))
    vol = load_volume(tmp_path)
    assert _values(vol) == [1, 2, 10]


def test_gui_loads_folder_and_counts_slices(tmp_path, monkeypatch, tk_root):
    import rockstone.gui as gui_module
    from rockstone.gui import RockstoneApp

    _write_tiffs(tmp_path, [("slice_1", 10), ("slice_2", 20),
                            ("slice_10", 30)])
    errors = []
    monkeypatch.setattr(gui_module, "messagebox", type("MB", (), {
        "showinfo": staticmethod(lambda *a, **k: None),
        "showerror": staticmethod(lambda *a, **k: errors.append(a))}))
    app = RockstoneApp(tk_root)
    app.load_path(tmp_path)
    assert not errors
    assert app.data.shape == (3, 6, 8)
    assert app.slice_scale.cget("to") == 2
    assert "3 slices" in app.status.get()
    np.testing.assert_array_equal(app.data[1],
                                  np.full((6, 8), 20, dtype=np.uint8))
