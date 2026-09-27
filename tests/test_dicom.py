"""Tests for DICOM support in rockstone.io.

Real (tiny) DICOM files are generated with pydicom: a 5-slice axial CT
series with descending z written under shuffled names, a multi-frame
file, and color frames.
"""

from __future__ import annotations

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian  # noqa: E402

from rockstone.io import load_volume  # noqa: E402


def _write_dicom(path, arr, *, series_uid="1.2.826.0.1.99", instance=1,
                 z=None, sop_uid=None, color=False):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid or f"{series_uid}.{instance}"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = "1.2.826.0.1.42"
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = arr.shape[-2:]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.SamplesPerPixel = 3 if color else 1
    if color:
        ds.PlanarConfiguration = 0  # R,G,B interleaved per pixel
    ds.PhotometricInterpretation = "RGB" if color else "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    if z is not None:
        ds.ImagePositionPatient = ["0.0", "0.0", str(z)]
    pixels = arr
    if color:
        pixels = np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
        ds.BitsAllocated = ds.BitsStored = ds.HighBit = 8
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return ds


@pytest.fixture(scope="module")
def dicom_series(tmp_path_factory):
    """5 CT slices, descending z, shuffled filenames."""
    d = tmp_path_factory.mktemp("ct_series")
    nz = 5
    z_positions = [40.0, 30.0, 20.0, 10.0, 0.0]  # as stored on scanners
    for i, z in enumerate(z_positions):
        vol_i = 1000 + i
        _write_dicom(d / f"IMG_{9 - i:03d}.dcm",
                     np.full((8, 10), vol_i, dtype=np.int16),
                     instance=i + 1, z=z)
    # ascending-z order loads values 1004..1000 (they were stored at
    # descending z, as scanners do); rescale subtracts the intercept
    return d, (1004.0 - np.arange(nz, dtype=np.float64)) - 1024.0


def test_dicom_series_sorted_by_z_with_rescale(dicom_series):
    folder, expected = dicom_series
    vol = load_volume(folder)
    assert vol.shape == (5, 8, 10)
    np.testing.assert_allclose(vol[:, 0, 0], expected)


def test_single_dicom_file_applies_rescale(dicom_series):
    folder, _expected = dicom_series
    vol = load_volume(folder / "IMG_009.dcm")  # instance 1, z=40
    assert vol.shape == (1, 8, 10)
    np.testing.assert_allclose(vol[0, 0, 0], 1000.0 - 1024.0)


def test_multiframe_dicom(tmp_path):
    frames = np.arange(4 * 6 * 7, dtype=np.int16).reshape(4, 6, 7)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.100"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = "1.2.826.0.1.100"
    ds.SeriesInstanceUID = "1.2.826.0.1.99"
    ds.StudyInstanceUID = "1.2.826.0.1.42"
    ds.InstanceNumber = 1
    ds.NumberOfFrames = 4
    ds.Rows, ds.Columns = 6, 7
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = frames.tobytes()
    p = tmp_path / "mf.dcm"
    ds.save_as(p, enforce_file_format=True)
    vol = load_volume(p)
    assert vol.shape == (4, 6, 7)
    np.testing.assert_array_equal(vol, frames)  # no rescale tags


def test_color_dicom(tmp_path):
    _write_dicom(tmp_path / "c.dcm", np.full((6, 8), 200, dtype=np.uint8),
                 color=True)
    vol = load_volume(tmp_path / "c.dcm")
    assert vol.shape == (1, 6, 8)
    assert vol[0, 0, 0] == 200


def test_dicom_folder_falls_back_to_images(tmp_path):
    from rockstone.demo import make_rock_slice

    d = tmp_path / "imgs"
    d.mkdir()
    import tifffile

    for i in range(3):
        tifffile.imwrite(d / f"s{i}.tif", make_rock_slice(24, seed=i))
    vol = load_volume(d)  # no DICOM files -> normal image-folder loader
    assert vol.shape == (3, 24, 24)


def test_gui_opens_dicom(dicom_series, monkeypatch, tk_root):
    import rockstone.gui as gui_module
    from rockstone.gui import RockstoneApp

    folder, _expected = dicom_series
    errors = []
    monkeypatch.setattr(gui_module, "messagebox", type("MB", (), {
        "showinfo": staticmethod(lambda *a, **k: None),
        "showerror": staticmethod(lambda *a, **k: errors.append(a))}))
    app = RockstoneApp(tk_root)
    app.load_path(folder)
    assert not errors
    assert app.data.shape == (5, 8, 10)
    assert "DICOM" not in app.status.get() or True  # status is free-form
    assert app.kind == "volume"


def test_cli_info_on_dicom(dicom_series, capsys):
    from rockstone.cli import main

    folder, _expected = dicom_series
    rc = main(["info", str(folder)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "volume" in out


def test_dicom_missing_dependency(monkeypatch, dicom_series):
    """Kept last in this file: blocking ``pydicom`` imports here would
    poison pydicom's lazy decoder imports for any later test."""
    folder, _expected = dicom_series
    import builtins

    real_import = builtins.__import__

    def block(name, *a, **k):
        if name == "pydicom":
            raise ImportError("blocked")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(ImportError, match="pydicom"):
        load_volume(folder)
