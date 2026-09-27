"""Tests for I/O, the pipeline, the CLI and the synthetic demo."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from rockstone.demo import make_rock_stack
from rockstone.io import load_volume, read_data, save_volume
from rockstone.pipeline import process


@pytest.fixture()
def tmp_tiff_stack(tmp_path):
    vol, true = make_rock_stack(n_slices=6, size=96, seed=0, cupping=0.3)
    p = tmp_path / "stack.tif"
    tifffile.imwrite(p, vol.astype(np.float32))
    return p, vol, true


class TestIO:
    def test_tiff_roundtrip(self, tmp_tiff_stack, tmp_path):
        p, vol, _ = tmp_tiff_stack
        loaded = load_volume(p)
        assert loaded.shape == vol.shape
        out = tmp_path / "out.tif"
        save_volume(loaded, out)
        np.testing.assert_allclose(load_volume(out), loaded)

    def test_npy_roundtrip(self, tmp_tiff_stack, tmp_path):
        p, vol, _ = tmp_tiff_stack
        out = tmp_path / "out.npy"
        save_volume(vol, out)
        np.testing.assert_allclose(load_volume(out), vol, atol=1e-6)

    def test_folder_of_slices(self, tmp_tiff_stack, tmp_path):
        _, vol, _ = tmp_tiff_stack
        folder = tmp_path / "slices"
        folder.mkdir()
        for i in range(vol.shape[0]):
            tifffile.imwrite(folder / f"s{i:03d}.tif", vol[i])
        loaded = load_volume(folder)
        assert loaded.shape == vol.shape
        np.testing.assert_allclose(loaded, vol, atol=1e-6)

    def test_kind_detection(self, tmp_tiff_stack):
        p, _, _ = tmp_tiff_stack
        data, kind = read_data(p)
        assert kind == "volume"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_volume(tmp_path / "nope.tif")


class TestPipeline:
    def test_process_volume(self, tmp_tiff_stack, tmp_path):
        p, _, _ = tmp_tiff_stack
        out = tmp_path / "processed.tif"
        report = process(p, output_path=out, align=True, deharden=True)
        assert out.exists()
        assert "align_slices" in report.steps_run
        assert "deharden_radial" in report.steps_run
        assert report.cupping_after < report.cupping_before
        text = report.summary()
        assert "cupping" in text

    def test_align_only(self, tmp_tiff_stack, tmp_path):
        p, _, _ = tmp_tiff_stack
        out = tmp_path / "aligned.tif"
        report = process(p, output_path=out, align=True, deharden=False)
        assert report.steps_run == ["align_slices"]
        assert np.isnan(report.cupping_after)

    def test_nothing_to_do(self, tmp_tiff_stack, tmp_path):
        p, _, _ = tmp_tiff_stack
        with pytest.raises(ValueError):
            process(p, align=False, deharden=False)


class TestCLI:
    def test_info(self, tmp_tiff_stack, capsys):
        from rockstone.cli import main

        p, _, _ = tmp_tiff_stack
        rc = main(["info", str(p)])
        assert rc == 0
        assert "shape" in capsys.readouterr().out

    def test_align_and_deharden(self, tmp_tiff_stack, tmp_path, capsys):
        from rockstone.cli import main

        p, _, _ = tmp_tiff_stack
        out = tmp_path / "cli_out.tif"
        rc = main(["align", str(p), "-o", str(out), "--no-rotation"])
        assert rc == 0 and out.exists()
        rc = main(["deharden", str(p), "-o", str(out), "--method", "radial"])
        assert rc == 0 and out.exists()

    def test_error_handling(self, tmp_path, capsys):
        from rockstone.cli import main

        rc = main(["info", str(tmp_path / "missing.tif")])
        assert rc == 2


class TestDemo:
    def test_demo_end_to_end(self, tmp_path, capsys):
        from rockstone.demo import run_demo

        rc = run_demo(out_dir=str(tmp_path / "demo"), n_slices=6, size=96)
        assert rc == 0
        text = capsys.readouterr().out
        assert "ground truth" in text
        assert "cupping" in text
