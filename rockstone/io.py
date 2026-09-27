"""Reading and writing of slice stacks and raw projections.

Supported inputs:

- a single multi-page TIFF (``stack.tiff``), read with tifffile
- a folder of 2D images (``slices/*.tif``, ``*.png``, ...) — including
  subfolders — ordered naturally (``slice_2`` before ``slice_10``)
- a NumPy ``.npy`` file or ``.npz`` archive containing one 3D array
  (``.npz`` picks the first 3D array unless ``key`` is given)
- a DICOM file (``.dcm``/``.dicom``), single-frame or multi-frame, or a
  folder containing a DICOM series (sorted by ImagePositionPatient z /
  InstanceNumber); RescaleSlope/Intercept are applied so CT data comes
  back in Hounsfield units

All loaders return a :class:`numpy.ndarray` with shape ``(nz, ny, nx)``.
For very large stacks, use :class:`VolumeReader` to iterate slice-by-slice
without holding the whole volume in memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

try:  # tifffile is a hard dependency but keep import errors readable
    import tifffile
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rockstone requires tifffile; install with `pip install tifffile`"
    ) from exc

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}


def _is_dicom_file(path: Path) -> bool:
    """True for ``.dcm`` files and for files with the DICOM magic
    (``DICM`` at byte offset 128), regardless of extension."""
    if path.suffix.lower() in DICOM_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def _require_pydicom():
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DICOM support requires pydicom; install with `pip install pydicom`"
        ) from exc
    return pydicom


def _dicom_rescale(ds) -> np.ndarray:
    """Apply RescaleSlope/Intercept (CT: raw stored values -> HU)."""
    arr = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr.astype(np.float64) * slope + intercept
    if arr.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 3:
        # color DICOM (e.g. RGB micrograph): reduce to grayscale like the
        # TIFF/RGBA path does
        arr = arr[..., :3].mean(axis=-1)
    return arr


def _load_dicom_file(path: Path) -> np.ndarray:
    pydicom = _require_pydicom()
    ds = pydicom.dcmread(path)
    arr = _dicom_rescale(ds)
    if arr.ndim == 2:  # single frame -> one-slice stack
        arr = arr[np.newaxis]
    return _as_volume(arr)


def _load_dicom_series(folder: Path) -> np.ndarray:
    """Stack all DICOM files in ``folder``, ordered by slice position.

    Sort key: ``ImagePositionPatient`` z (axial CT series), falling back
    to ``InstanceNumber``, then filename. Rescale is applied per file
    (slope/intercept can vary slice to slice).
    """
    pydicom = _require_pydicom()
    files = [f for f in sorted(folder.iterdir())
             if f.is_file() and _is_dicom_file(f)]
    if not files:
        raise FileNotFoundError(f"no DICOM files found in {folder}")

    datasets: list[tuple[Path, object]] = []
    for f in files:
        try:
            datasets.append((f, pydicom.dcmread(f)))
        except Exception as exc:  # noqa: BLE001 - skip non-DICOM strays
            if f.suffix.lower() in DICOM_EXTENSIONS:
                raise ValueError(f"unreadable DICOM file: {f}\n{exc}") from exc

    def sort_key(item) -> tuple:
        _f, ds = item
        pos = getattr(ds, "ImagePositionPatient", None)
        if pos is not None and len(pos) >= 3:
            return (0, float(pos[2]), "")
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            return (1, float(inst), "")
        return (2, 0.0, str(item[0]))

    datasets.sort(key=sort_key)

    slices = []
    shape = None
    for f, ds in datasets:
        arr = _dicom_rescale(ds)
        if arr.ndim == 3:  # embedded multi-frame file inside a series
            slices.extend(a for a in arr)
        elif arr.ndim == 2:
            slices.append(arr)
        else:
            raise ValueError(f"unsupported DICOM pixel shape {arr.shape} "
                             f"in {f}")
        if shape is None:
            shape = arr.shape[-2:]
        elif arr.shape[-2:] != shape:
            raise ValueError(
                f"inconsistent slice sizes in DICOM series: "
                f"{arr.shape[-2:]} vs {shape} (in {f})")
    return _as_volume(np.stack(slices, axis=0))


def _norm_path(path: str | os.PathLike) -> Path:
    return Path(path)


def load_volume(path: str | os.PathLike, key: str | None = None,
                recursive: bool = True) -> np.ndarray:
    """Load a 3D stack from a file or folder. Returns ``(nz, ny, nx)``.

    For folders, ``recursive=True`` (the default) also gathers images
    from subfolders, in natural (numeric-aware) order; set it to False
    for the old top-level-only behavior.
    """
    p = _norm_path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        if any(_is_dicom_file(f) for f in p.iterdir() if f.is_file()):
            return _load_dicom_series(p)
        return _load_folder(p, recursive=recursive)
    suffix = p.suffix.lower()
    if suffix in DICOM_EXTENSIONS:
        return _load_dicom_file(p)
    if suffix == ".npy":
        data = np.load(p)
        return _as_volume(data)
    if suffix == ".npz":
        with np.load(p) as z:
            if key is not None:
                return _as_volume(z[key])
            for name in z.files:
                arr = z[name]
                if arr.ndim >= 3:
                    return _as_volume(arr)
            raise ValueError(f"no 3D array found in {p}")
    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(p) as tif:
            return _as_volume(tif.asarray())
    raise ValueError(f"unsupported volume format: {p.suffix!r} for {p}")


def _natural_key(path: Path) -> tuple:
    """Sort key that treats digit runs as numbers, so ``slice_2``
    comes before ``slice_10`` (lexicographic sort would not). Uses the
    full POSIX path so subfolder names also order correctly."""
    import re

    parts = re.split(r"(\d+)", path.as_posix().lower())
    key: list = []
    for part in parts:
        if part.isdigit():
            key.append((1, int(part), ""))
        elif part:
            key.append((0, 0, part))
    return tuple(key)


def _collect_image_files(folder: Path, recursive: bool = False) -> list[Path]:
    """All image files in ``folder`` (and subfolders when recursive),
    in natural order. DICOM files inside are ignored here — a folder
    with DICOM files takes the series-loading path instead."""
    pattern = "**/*" if recursive else "*"
    files = [f for f in sorted(folder.glob(pattern), key=_natural_key)
             if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    return files


def _load_folder(folder: Path, recursive: bool = False) -> np.ndarray:
    files = _collect_image_files(folder, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"no images found in {folder}")
    slices = []
    skipped: list[str] = []
    for f in files:
        try:
            arr = (tifffile.imread(f)
                   if f.suffix.lower() in {".tif", ".tiff"}
                   else _read_generic_image(f))
        except Exception:  # noqa: BLE001 - keep going on unreadable files
            skipped.append(f.name)
            continue
        slices.append(arr)
    if not slices:
        raise FileNotFoundError(f"no readable images found in {folder}")
    if skipped:
        import warnings

        warnings.warn(f"skipped {len(skipped)} unreadable image(s) in "
                      f"{folder}: {', '.join(skipped)}")
    return _as_volume(np.stack(slices, axis=0))


def load_stack_from_files(paths) -> np.ndarray:
    """Stack several image files into one volume, in natural order.

    Accepts any iterable of paths (as given, e.g., by a multi-selection
    file dialog). Each file contributes one slice (2-D image, or a
    color image), except multi-page TIFFs / volumes without channel
    data, which contribute every page as a separate slice. Slice shapes
    must agree; mismatches raise a :class:`ValueError` naming the files.
    Grayscale and color selections stack fine together; like the other
    loaders, the returned volume is grayscale ``(nz, ny, nx)``.
    """
    from pathlib import Path as _P

    files = sorted((_P(p) for p in paths), key=_natural_key)
    if not files:
        raise ValueError("no files selected")

    slices: list[np.ndarray] = []
    shapes: list[tuple[tuple, Path]] = []
    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)
        suffix = p.suffix.lower()
        if suffix in DICOM_EXTENSIONS:
            arr = _load_dicom_file(p)
            if arr.shape[0] == 1:  # single-frame DICOM -> its one slice
                arr = arr[0]
        elif suffix == ".npy":
            arr = np.load(p)
        elif suffix == ".npz":
            with np.load(p) as z:
                arr = None
                for name in z.files:
                    if z[name].ndim >= 2:
                        arr = z[name]
                        break
                if arr is None:
                    raise ValueError(f"no array found in {p}")
        elif suffix in {".tif", ".tiff"}:
            arr = tifffile.imread(p)
        elif suffix in IMAGE_EXTENSIONS:
            arr = _read_generic_image(p)
        else:
            raise ValueError(f"unsupported file type: {p}")

        arr = np.asarray(arr)
        is_multipage = arr.ndim == 4 or (arr.ndim == 3 and arr.shape[-1] > 4)
        if is_multipage:
            # multi-page volume: every page becomes a slice
            slices.extend(a for a in arr)
            shapes.extend([(a.shape, p) for a in arr])
        else:
            slices.append(arr)
            shapes.append((arr.shape, p))

    # grayscale + color mix: promote gray slices to 3-channel RGB so the
    # selection still stacks (common when screenshots join gray TIFFs)
    flat = [s for s in slices if s.ndim == 2]
    color = [s for s in slices if s.ndim == 3]
    if flat and color:
        target = color[0].shape[-1]
        slices = [np.repeat(s[..., None], target, axis=-1) if s.ndim == 2
                  else s for s in slices]
        shapes = [(s.shape, p) for s, (_sh, p) in zip(slices, shapes)]

    ref_shape = shapes[0][0]
    for shape, p in shapes[1:]:
        if shape != ref_shape:
            raise ValueError(
                f"mismatched slice sizes in selection: {p.name} is "
                f"{shape} but {shapes[0][1].name} is {ref_shape}; all "
                f"slices must have the same width and height")
    return _as_volume(np.stack(slices, axis=0))


def _read_generic_image(path: Path) -> np.ndarray:
    import imageio.v2 as iio  # deferred: only needed for PNG/JPG folders

    return iio.imread(path)


def load_projections(
    path: str | os.PathLike, key: str | None = None, as_sinogram: bool = True
) -> np.ndarray:
    """Load raw projections.

    Detector files usually store ``(projections, rows, columns)``. If
    ``as_sinogram`` is True (default) the data is transposed to the
    internal sinogram layout ``(rows, columns, projections)``; set it to
    False to keep the detector layout.
    """
    data = load_volume(path, key=key)
    if data.ndim != 3:
        raise ValueError(f"expected 3D projection data, got shape {data.shape}")
    if as_sinogram:
        return np.transpose(data, (1, 2, 0))
    return data


def save_projections(
    sino: np.ndarray,
    path: str | os.PathLike,
    from_sinogram: bool = True,
) -> None:
    """Save sinogram-ordered data back to detector order ``(n, rows, cols)``."""
    p = _norm_path(path)
    data = np.transpose(sino, (2, 0, 1)) if from_sinogram else sino
    _save_array(data, p)


def save_volume(volume: np.ndarray, path: str | os.PathLike,
                meta: dict | None = None) -> None:
    """Save a ``(nz, ny, nx)`` volume as a multi-page TIFF, ``.npy`` or
    compressed ``.npz``.

    ``meta`` optionally embeds physical dimensions, e.g.
    ``{"pixel_size": [0.05, 0.05], "slice_spacing": 0.05}`` (mm). It is
    written into the TIFF image description / a NPZ sidecar key and can
    be read back with :func:`read_metadata` (``.npy`` cannot hold it).
    """
    _save_array(np.asarray(volume), _norm_path(path), meta=meta)


def _save_array(data: np.ndarray, p: Path, meta: dict | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".tif", ".tiff"}:
        desc = json.dumps(meta) if meta else None
        tifffile.imwrite(p, data, description=desc)
    elif p.suffix.lower() == ".npy":
        np.save(p, data)
    elif p.suffix.lower() == ".npz":
        if meta:
            np.savez_compressed(p, data=data, meta_json=json.dumps(meta))
        else:
            np.savez_compressed(p, data=data)
    else:
        raise ValueError(f"unsupported output format: {p.suffix!r}")


def read_metadata(path: str | os.PathLike) -> dict | None:
    """Read embedded physical dimensions from a data file, if present.

    Returns e.g. ``{"pixel_size": [0.05, 0.05], "slice_spacing": 0.05}``
    (millimeters) or ``None``. For DICOM, ``PixelSpacing`` and
    ``SliceThickness`` are reported; TIFF looks for a JSON image
    description written by :func:`save_volume`; NPZ for a ``meta_json``
    key. Folders are scanned for their first DICOM file.
    """
    p = _norm_path(path)
    try:
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and _is_dicom_file(f):
                    meta = read_metadata(f)
                    if meta:
                        return meta
            return None
        suffix = p.suffix.lower()
        if suffix in DICOM_EXTENSIONS:
            import pydicom

            ds = pydicom.dcmread(p, stop_before_pixels=True)
            out: dict = {}
            ps = getattr(ds, "PixelSpacing", None)
            if ps is not None and len(ps) >= 2:
                # DICOM PixelSpacing is [row, col]; report as (x, y)
                out["pixel_size"] = [float(ps[1]), float(ps[0])]
            st = getattr(ds, "SliceThickness", None)
            if st is not None:
                out["slice_spacing"] = float(st)
            return out or None
        if suffix in {".tif", ".tiff"}:
            with tifffile.TiffFile(p) as tif:
                desc = tif.pages[0].description or ""
            if desc.startswith("{"):
                d = json.loads(desc)
                if isinstance(d, dict) and "pixel_size" in d:
                    return d
            return None
        if suffix == ".npz":
            with np.load(p) as z:
                if "meta_json" in z.files:
                    d = json.loads(str(z["meta_json"]))
                    return d if isinstance(d, dict) else None
            return None
    except Exception:  # noqa: BLE001 - metadata is best-effort
        return None
    return None


def _as_volume(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 2:
        return data[np.newaxis, ...]
    if data.ndim == 3:
        return data
    if data.ndim == 4 and data.shape[-1] <= 4:
        # e.g. RGBA page stack: drop alpha, average color to gray
        rgb = data[..., :3].astype(np.float64)
        gray = rgb.mean(axis=-1)
        return gray.astype(data.dtype)
    raise ValueError(f"cannot interpret array of shape {data.shape} as a stack")


def read_data(path: str | os.PathLike, key: str | None = None) -> tuple[np.ndarray, str]:
    """Auto-detect whether ``path`` holds a stack or projections.

    Returns ``(array, kind)`` where ``kind`` is ``"volume"`` or
    ``"projections"``. Heuristic: a volume is treated as a *stack* when its
    first-axis extent (slice count) is large compared with the other axes.
    Explicit overrides are available through the individual loaders.
    """
    data = load_volume(path, key=key)
    nz, ny, nx = data.shape
    if nz > 1.5 * max(ny, nx):
        return data, "projections"  # detector stack: many projections
    return data, "volume"


class VolumeReader:
    """Random-access reader for slice stacks that avoids loading everything.

    Only the folder-of-images and multi-page-TIFF layouts support true
    lazy reading; other formats fall back to a full load.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = _norm_path(path)
        if self.path.is_dir():
            self._files = sorted(
                f for f in self.path.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not self._files:
                raise FileNotFoundError(f"no images found in {self.path}")
            first = self._read_file(0)
            self.shape = (len(self._files),) + first.shape
        elif self.path.suffix.lower() in {".tif", ".tiff"}:
            self._tif = tifffile.TiffFile(self.path)
            self.shape = (len(self._tif.pages),) + self._tif.pages[0].shape
        else:
            self._data = load_volume(self.path)
            self.shape = self._data.shape

    def _read_file(self, index: int) -> np.ndarray:
        f = self._files[index]
        if f.suffix.lower() in {".tif", ".tiff"}:
            return tifffile.imread(f)
        return _read_generic_image(f)

    def read_slice(self, index: int) -> np.ndarray:
        if self.path.is_dir():
            return self._read_file(index)
        if self.path.suffix.lower() in {".tif", ".tiff"}:
            return self._tif.pages[index].asarray()
        return self._data[index]

    def read_range(self, start: int, stop: int) -> np.ndarray:
        return np.stack([self.read_slice(i) for i in range(start, stop)], axis=0)

    def close(self) -> None:
        if hasattr(self, "_tif"):
            self._tif.close()

    def __enter__(self) -> "VolumeReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
