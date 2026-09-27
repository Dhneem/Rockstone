"""Tkinter GUI: click a file, it opens — then align and deharden.

Run with ``rockstone gui`` or ``python -m rockstone.gui``. The window
greets you with a "Open file…" panel; clicking a supported file (TIFF
stack, NPY/NPZ, or a folder of PNG/JPEG slices) loads it, shows the kind
(volume vs raw projections), dims and cupping severity, and lets you
browse slices and run the alignment / beam-hardening corrections.

The GUI only renders; all image math lives in the library modules, which
keeps it testable without a display.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np

from .align import align_slices
from .background import correct_background
from .beam_hardening import remove_beam_hardening
from .compare import compare_volumes
from .crop import crop_ellipse, ellipse_outline_mask
from .export import EXPORT_FORMATS, export_slice
from .report import collect_report_data, generate_report
from .roi import roi_stats, roi_stats_series, write_roi_csv
from .io import (load_stack_from_files, load_volume, read_metadata,
                 save_volume)
from .metrics import cupping_index, radial_profile

__all__ = ["RockstoneApp", "detect_kind", "launch", "COLORMAPS",
           "apply_colormap", "normalize_u8", "window_u8",
           "histogram_counts", "paint_crop_overlay"]

#: extensions offered in the open-file dialog (plus "folder of images")
FILE_PATTERNS = [
    ("TIFF stacks", "*.tif *.tiff"),
    ("DICOM files", "*.dcm *.dicom"),
    ("NumPy arrays", "*.npy *.npz"),
    ("All files", "*.*"),
]

LONG_TASK_TOKEN = object()  # sentinel for the work queue

#: zoom limits and per-step factor for buttons / mouse wheel
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.1, 20.0, 1.25

#: maximum number of modification steps kept for undo/redo
HISTORY_MAX = 10


# ------------------------------------------------------------ color systems --

def _lut_from_anchors(
    anchors: list[tuple[float, tuple[int, int, int]]],
) -> np.ndarray:
    """Build a 256-entry uint8 RGB LUT by interpolating anchor colors."""
    xs = np.array([a[0] for a in anchors], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        ys = np.array([a[1][c] for a in anchors], dtype=np.float64)
        lut[:, c] = np.clip(np.round(np.interp(t, xs, ys)), 0, 255)
    return lut


def _gray_lut(inverted: bool = False) -> np.ndarray:
    i = np.arange(256, dtype=np.uint8)
    if inverted:
        i = 255 - i
    return np.stack([i, i, i], axis=-1)


#: display color systems, in the order shown in the GUI dropdown
COLORMAPS: dict[str, np.ndarray] = {
    "Gray": _gray_lut(False),
    "Inverted": _gray_lut(True),
    "Hot metal": _lut_from_anchors([  # classic 'hot': black-red-yellow-white
        (0.0, (0, 0, 0)), (0.375, (255, 0, 0)),
        (0.75, (255, 255, 0)), (1.0, (255, 255, 255))]),
    "Viridis": _lut_from_anchors([  # perceptually uniform (matplotlib key points)
        (0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]),
    "Jet": _lut_from_anchors([  # dark blue -> cyan -> yellow -> dark red
        (0.0, (0, 0, 143)), (0.11, (0, 0, 255)), (0.34, (0, 255, 255)),
        (0.5, (0, 255, 0)), (0.66, (255, 255, 0)), (0.89, (255, 0, 0)),
        (1.0, (128, 0, 0))]),
    "Rock": _lut_from_anchors([  # warm sepia tuned for rock texture
        (0.0, (10, 5, 0)), (0.35, (94, 52, 18)),
        (0.7, (194, 143, 82)), (1.0, (252, 240, 214))]),
    "Asphalt": _lut_from_anchors([  # cool dark grays, like fresh pavement
        (0.0, (5, 6, 8)), (0.4, (45, 48, 54)),
        (0.75, (110, 115, 122)), (1.0, (208, 212, 218))]),
    "Ice": _lut_from_anchors([  # black -> deep blue -> pale ice white
        (0.0, (2, 8, 20)), (0.35, (10, 60, 120)),
        (0.7, (60, 150, 210)), (1.0, (225, 245, 255))]),
    "Copper": _lut_from_anchors([  # black -> bronze -> polished copper
        (0.0, (0, 0, 0)), (0.4, (120, 50, 20)),
        (0.75, (200, 120, 60)), (1.0, (255, 205, 150))]),
    "Thermal": _lut_from_anchors([  # iron/heat: black -> purple -> orange -> yellow
        (0.0, (0, 0, 0)), (0.25, (60, 0, 90)),
        (0.5, (200, 30, 60)), (0.75, (255, 140, 0)),
        (1.0, (255, 240, 120))]),
    "Bone": _lut_from_anchors([  # radiographic: cool mid-grays, warm white
        (0.0, (0, 0, 0)), (0.4, (70, 75, 85)),
        (0.75, (175, 180, 185)), (1.0, (255, 250, 240))]),
}

DEFAULT_COLORMAP = "Gray"


def apply_colormap(gray_u8: np.ndarray, name: str) -> np.ndarray:
    """Map a single-channel uint8 image through a named color system.

    Returns an ``(h, w, 3)`` uint8 RGB array. Unknown names fall back to
    :data:`DEFAULT_COLORMAP`.
    """
    lut = COLORMAPS.get(name, COLORMAPS[DEFAULT_COLORMAP])
    return lut[np.asarray(gray_u8, dtype=np.uint8)]


# ---------------------------------------------------------------- views --

def detect_kind(data: np.ndarray) -> str:
    """Same heuristic as :func:`rockstone.io.read_data`."""
    nz, ny, nx = data.shape
    return "projections" if nz > 1.5 * max(ny, nx) else "volume"


def normalize_u8(plane: np.ndarray) -> np.ndarray:
    """Percentile-stretch a float plane to uint8 for display."""
    plane = np.asarray(plane, dtype=np.float64)
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros(plane.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    out = (plane - lo) / (hi - lo)
    out = np.nan_to_num(out, nan=0.0)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def window_u8(plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Stretch a plane to uint8 over the display window ``[lo, hi]``.

    Like :func:`normalize_u8` but with user-chosen bounds — this backs
    the brightness/contrast sliders, which only affect the *display*:
    pixel values, histograms of the data and saved files stay untouched.
    """
    plane = np.asarray(plane, dtype=np.float64)
    if hi <= lo:
        hi = lo + 1.0
    out = (plane - lo) / (hi - lo)
    out = np.nan_to_num(out, nan=0.0)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def histogram_counts(u8: np.ndarray, bins: int = 32) -> np.ndarray:
    """Bin counts for the histogram panel, from a uint8 display image."""
    counts, _edges = np.histogram(np.asarray(u8).ravel(), bins=bins,
                                  range=(0, 255))
    return counts.astype(np.int64)


def normalize_u8_plane3(plane: np.ndarray) -> np.ndarray:
    """Display version of a multichannel plane: per-channel stretch to
    uint8 so colors stay balanced."""
    rgb = np.asarray(plane[..., :3], dtype=np.float64)
    out = np.empty_like(rgb, dtype=np.float64)
    for c in range(3):
        out[..., c] = normalize_u8(rgb[..., c])
    return out.astype(np.uint8)


def paint_crop_overlay(u8: np.ndarray,
                       rect: tuple[int, int, int, int],
                       src_hw: tuple[int, int],
                       color: str = "red",
                       shape: str = "rect") -> np.ndarray:
    """Return ``u8`` with the selection painted as an outline.

    ``rect`` is ``(x0, y0, x1, y1)`` in source-image pixels (inclusive),
    ``src_hw`` the (h, w) of the unrescaled source image; the selection
    is mapped onto ``u8``'s current (possibly zoomed) size. ``shape``
    draws the rectangle outline or the ellipse inscribed in it. Gray
    images get a bright outline with a dark inner line (rect only), RGB
    images a colored one (red for crop selections, cyan for ROI
    measurements, ...).
    """
    src_h, src_w = src_hw
    h, w = u8.shape[:2]
    x0, y0, x1, y1 = rect
    dx0 = max(0, min(int(round(x0 * w / src_w)), w - 1))
    dx1 = max(0, min(int(round((x1 + 1) * w / src_w)) - 1, w - 1), dx0)
    dy0 = max(0, min(int(round(y0 * h / src_h)), h - 1))
    dy1 = max(0, min(int(round((y1 + 1) * h / src_h)) - 1, h - 1), dy0)
    out = np.asarray(u8).copy()
    if out.ndim == 3:
        bright = {"red": (255, 70, 70), "cyan": (60, 220, 255),
                  "yellow": (255, 220, 60)}.get(color, (255, 70, 70))
        dark = (0, 0, 0)
    else:
        bright, dark = 255, 0
    if shape == "ellipse":
        outline = ellipse_outline_mask((h, w), (dx0, dy0, dx1, dy1),
                                       thickness=2)
        out[outline] = bright
        return out
    for yy in (dy0, dy0 + 1, dy1 - 1, dy1):
        if 0 <= yy < h:
            out[yy, dx0:dx1 + 1] = bright
    for xx in (dx0, dx0 + 1, dx1 - 1, dx1):
        if 0 <= xx < w:
            out[dy0:dy1 + 1, xx] = bright
    # dark inner lines keep the rectangle visible on bright content
    for yy in (dy0 + 2, dy1 - 2):
        if 0 <= yy < h:
            out[yy, dx0 + 2:max(dx0 + 2, dx1 - 1)] = dark
    for xx in (dx0 + 2, dx1 - 2):
        if 0 <= xx < w:
            out[dy0 + 2:max(dy0 + 2, dy1 - 1), xx] = dark
    return out


def to_pgm(plane_u8: np.ndarray) -> bytes:
    """Serialize a uint8 plane as binary PGM (gray) or PPM (color).

    Tk's ``PhotoImage`` reads Netpbm formats natively (binary variants
    only), so no Pillow dependency is needed to display images.
    """
    if plane_u8.ndim == 3:
        h, w, n = plane_u8.shape
        if n < 3:
            plane_u8 = np.repeat(plane_u8[..., :1], 3, axis=-1)
        body = np.ascontiguousarray(plane_u8[..., :3]).tobytes()
        return b"P6 %d %d 255\n" % (w, h) + body
    h, w = plane_u8.shape
    return b"P5 %d %d 255\n" % (w, h) + np.ascontiguousarray(plane_u8).tobytes()


def resize_nearest(u8: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Nearest-neighbor resize of a uint8 (h, w[, c]) image to (th, tw)."""
    src_h, src_w = u8.shape[:2]
    th, tw = max(1, int(th)), max(1, int(tw))
    ys = np.clip((np.arange(th) * src_h / th).astype(np.int64), 0, src_h - 1)
    xs = np.clip((np.arange(tw) * src_w / tw).astype(np.int64), 0, src_w - 1)
    return np.ascontiguousarray(u8[ys][:, xs])


def build_slice_montage(
    data: np.ndarray,
    colormap: str = DEFAULT_COLORMAP,
    cell: int = 96,
    gap: int = 4,
    cols: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], int, int]:
    """Build a contact-sheet image of every slice in ``data``.

    Returns ``(montage_u8, cells, rows, cols)`` where ``cells[i]`` is
    the ``(x0, y0, x1, y1)`` pixel rectangle of slice ``i`` inside the
    montage. Each slice is normalized exactly like the main viewer
    (percentile stretch + color system), scaled to fit ``cell`` and
    centered in its cell.
    """
    data = np.asarray(data)
    n = data.shape[0]
    if cols is None:
        cols = max(1, min(int(np.ceil(np.sqrt(n))), 12))
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))

    montage = np.zeros((rows * (cell + gap) + gap,
                        cols * (cell + gap) + gap, 3), dtype=np.uint8)
    cells: list[tuple[int, int, int, int]] = []
    for i in range(n):
        plane = data[i]
        if plane.ndim == 3 and plane.shape[-1] >= 3:
            u8 = normalize_u8_plane3(plane)
        else:
            u8 = apply_colormap(normalize_u8(plane), colormap)
        h, w = u8.shape[:2]
        scale = min(cell / h, cell / w)
        th, tw = max(1, round(h * scale)), max(1, round(w * scale))
        thumb = resize_nearest(u8, th, tw)
        r, c = divmod(i, cols)
        x0 = gap + c * (cell + gap)
        y0 = gap + r * (cell + gap)
        ox, oy = (cell - tw) // 2, (cell - th) // 2
        montage[y0 + oy:y0 + oy + th, x0 + ox:x0 + ox + tw] = thumb
        cells.append((x0, y0, x0 + cell, y0 + cell))
    return montage, cells, rows, cols


# ------------------------------------------------------------- the app ----

class RockstoneApp:
    """Main application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Rockstone — rock sample processing")
        root.minsize(760, 520)

        self.data: np.ndarray | None = None
        self.kind: str | None = None
        self.path: Path | None = None
        self.multi_paths: list[Path] | None = None  # multi-selection open
        self.displayed: np.ndarray | None = None  # currently shown array
        self._shown_index: int | None = None
        self.colormap: str = DEFAULT_COLORMAP
        self._busy = False
        self._processed = False  # True after align/deharden (data in RAM only)
        self._compare_mode: str | None = None  # "side" | "diff" | None
        self._compare_b: np.ndarray | None = None  # other dataset
        self._compare_result = None  # ComparisonResult of the last compare
        self.zoom: float = 1.0
        self._pan: tuple[int, int] = (0, 0)
        self._norm_cache: tuple | None = None  # (key, plane, u8)
        self._window: tuple[float, float] | None = None  # display lo/hi
        self._meta: dict | None = None  # physical dimensions (mm)
        self._building_ui = False  # suppress slider callbacks during setup
        self._map_win: tk.Toplevel | None = None
        self._map_generation = 0  # bumps cancel pending map builds
        self._map_canvas: tk.Canvas | None = None
        self._map_cells: list = []
        self._map_hl = None
        self._map_photo = None
        self._undo_stack: list[tuple[np.ndarray, str]] = []
        self._redo_stack: list[tuple[np.ndarray, str]] = []
        self._saved_array: np.ndarray | None = None  # last saved/loaded state
        self._original_array: np.ndarray | None = None  # as-loaded state
        self._drag_start: tuple | None = None
        self._crop_win: tk.Toplevel | None = None  # crop-mode dialog
        self._crop_sel: tuple[int, int, int, int] | None = None
        self._crop_apply_btn: ttk.Button | None = None
        self._crop_lbl_var: tk.StringVar | None = None
        self._crop_shape_var: tk.StringVar | None = None
        self._roi_win: tk.Toplevel | None = None  # ROI measurement dialog
        self._roi_sel: tuple[int, int, int, int] | None = None
        self._roi_stats_var: tk.StringVar | None = None

        self._menus: list[tk.Menu] = []
        self._menu_data_entries: list[tuple[tk.Menu, int]] = []
        self._menu_undo_entries: list[tuple[tk.Menu, int]] = []
        self._menu_redo_entries: list[tuple[tk.Menu, int]] = []
        self._menu_diff_entries: list[tuple[tk.Menu, int]] = []

        self._build_ui()
        self._show_welcome()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.bind("<Escape>", self._on_escape)
        root.bind("<Control-s>", lambda e: self.save())
        root.bind("<Control-S>", lambda e: self.save_as())
        root.bind("<Control-z>", lambda e: self.undo())
        root.bind("<Control-y>", lambda e: self.redo())
        root.bind("<Control-Z>", lambda e: self.redo())  # Ctrl+Shift+Z
        root.bind("<Control-o>", lambda e: self.open_dialog())
        root.bind("<Control-O>", lambda e: self.open_folder())  # Ctrl+Shift+O
        root.bind("<Control-e>", lambda e: self.export_slice_as())
        root.bind("<Control-r>", lambda e: self.generate_report_as())
        root.bind("<Control-m>", lambda e: self._toggle_map())
        root.bind("<Control-plus>",
                  lambda e: self._apply_zoom(ZOOM_STEP))
        root.bind("<Control-equal>",
                  lambda e: self._apply_zoom(ZOOM_STEP))
        root.bind("<Control-minus>",
                  lambda e: self._apply_zoom(1 / ZOOM_STEP))
        root.bind("<Control-f>", lambda e: self._zoom_fit())

    # -- menu bar ------------------------------------------------------------

    def _menu_add(self, menu: tk.Menu, entries) -> None:
        """Fill ``menu``; ``entries`` items are ``(label, command,
        accelerator)`` tuples or ``None`` for a separator."""
        for entry in entries:
            if entry is None:
                menu.add_separator()
                continue
            label, cmd, accel = entry
            menu.add_command(label=label, command=cmd,
                             accelerator=accel)

    def _track_menu_entry(self, menu: tk.Menu, label: str,
                          bucket: list) -> None:
        """Remember a menu entry index for state syncing."""
        idx = menu.index(label)
        if idx is not None:
            bucket.append((menu, idx))

    def _build_menus(self) -> None:
        """Create the main menu bar (File / Edit / View / Process / Help)."""
        bar = tk.Menu(self.root)
        self.root.config(menu=bar)
        self._menu_bar = bar

        file_menu = tk.Menu(bar, tearoff=0)
        self._menu_add(file_menu, [
            ("Open file…", self.open_dialog, "Ctrl+O"),
            ("Open folder…", self.open_folder, "Ctrl+Shift+O"),
            None,
            ("Save", self.save, "Ctrl+S"),
            ("Save As…", self.save_as, "Ctrl+Shift+S"),
            None,
            ("Exit", self._on_close, ""),
        ])
        for lab in ("Save", "Save As…"):
            self._track_menu_entry(file_menu, lab,
                                   self._menu_data_entries)
        bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(bar, tearoff=0)
        self._menu_add(edit_menu, [
            ("Undo", self.undo, "Ctrl+Z"),
            ("Redo", self.redo, "Ctrl+Y"),
            None,
            ("Rotate 90° clockwise",
             lambda: self.rotate_volume(1), ""),
            ("Rotate 90° counterclockwise",
             lambda: self.rotate_volume(-1), ""),
            ("Crop…", self.start_crop, ""),
            ("Measure ROI…", self.start_roi, ""),
            None,
            ("Physical dimensions…", self.edit_dimensions, ""),
        ])
        self._track_menu_entry(edit_menu, "Undo", self._menu_undo_entries)
        self._track_menu_entry(edit_menu, "Redo", self._menu_redo_entries)
        for lab in ("Rotate 90° clockwise",
                    "Rotate 90° counterclockwise", "Crop…",
                    "Measure ROI…", "Physical dimensions…"):
            self._track_menu_entry(edit_menu, lab,
                                   self._menu_data_entries)
        bar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(bar, tearoff=0)
        self._menu_add(view_menu, [
            ("Zoom in", lambda: self._apply_zoom(ZOOM_STEP), "Ctrl++"),
            ("Zoom out", lambda: self._apply_zoom(1 / ZOOM_STEP),
             "Ctrl+−"),
            ("Fit to window", self._zoom_fit, "Ctrl+F"),
            None,
            ("Slice map", self._toggle_map, "Ctrl+M"),
            ("Compare…", self.run_compare, ""),
            ("Show difference", self._toggle_diff, ""),
            None,
            ("Reset display", self._reset_display, ""),
        ])
        for lab in ("Slice map", "Compare…"):
            self._track_menu_entry(view_menu, lab,
                                   self._menu_data_entries)
        self._track_menu_entry(view_menu, "Show difference",
                               self._menu_diff_entries)
        self._track_menu_entry(view_menu, "Reset display",
                               self._menu_data_entries)
        bar.add_cascade(label="View", menu=view_menu)

        proc_menu = tk.Menu(bar, tearoff=0)
        self._menu_add(proc_menu, [
            ("Align slices", self.run_align, ""),
            ("Remove beam hardening", self.run_deharden, ""),
            ("Background correction…", self.run_background, ""),
            None,
            ("Export current slice…", self.export_slice_as, "Ctrl+E"),
            ("Generate report…", self.generate_report_as, "Ctrl+R"),
        ])
        for lab in ("Align slices", "Remove beam hardening",
                    "Background correction…", "Export current slice…",
                    "Generate report…"):
            self._track_menu_entry(proc_menu, lab,
                                   self._menu_data_entries)
        bar.add_cascade(label="Process", menu=proc_menu)

        help_menu = tk.Menu(bar, tearoff=0)
        help_menu.add_command(label="About Rockstone…",
                              command=self._show_about)
        bar.add_cascade(label="Help", menu=help_menu)

        self._menus = [file_menu, edit_menu, view_menu, proc_menu,
                       help_menu]

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About Rockstone",
            "Rockstone — rock sample image processing\n\n"
            "Slice alignment, beam-hardening and background "
            "correction,\nmeasurement tools and multi-format export "
            "for micro-CT data.")

    def _sync_menu_states(self) -> None:
        """Enable/disable menu entries to mirror the toolbar buttons."""
        if not self._menus:
            return
        has = self.data is not None and not self._busy
        state = tk.NORMAL if has else tk.DISABLED
        for menu, idx in self._menu_data_entries:
            menu.entryconfig(idx, state=state)
        for menu, idx in self._menu_undo_entries:
            menu.entryconfig(idx, state=tk.NORMAL
                             if has and self._undo_stack else tk.DISABLED)
        for menu, idx in self._menu_redo_entries:
            menu.entryconfig(idx, state=tk.NORMAL
                             if has and self._redo_stack else tk.DISABLED)
        diff_state = tk.NORMAL if self._compare_mode else tk.DISABLED
        for menu, idx in self._menu_diff_entries:
            menu.entryconfig(idx, state=diff_state)

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        self.status = tk.StringVar(value="Ready.")
        self.info = tk.StringVar(value="")

        self._build_menus()

        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Open file…", command=self.open_dialog)\
            .pack(side=tk.LEFT)
        ttk.Button(bar, text="Open folder…", command=self.open_folder)\
            .pack(side=tk.LEFT, padx=(6, 0))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT,
                                                    fill=tk.Y, padx=10)

        self.align_btn = ttk.Button(bar, text="Align slices",
                                    command=self.run_align, state=tk.DISABLED)
        self.align_btn.pack(side=tk.LEFT, padx=(18, 6))
        self.deharden_btn = ttk.Button(bar, text="Remove beam hardening",
                                       command=self.run_deharden,
                                       state=tk.DISABLED)
        self.deharden_btn.pack(side=tk.LEFT)
        self.bg_btn = ttk.Button(bar, text="Background…",
                                 command=self.run_background,
                                 state=tk.DISABLED)
        self.bg_btn.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT,
                                                    fill=tk.Y, padx=10)
        self.compare_btn = ttk.Button(bar, text="Compare…",
                                      command=self.run_compare,
                                      state=tk.DISABLED)
        self.compare_btn.pack(side=tk.LEFT)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT,
                                                    fill=tk.Y, padx=10)
        ttk.Label(bar, text="Zoom").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(bar, text="−", width=3,
                   command=lambda: self._apply_zoom(1 / ZOOM_STEP))\
            .pack(side=tk.LEFT)
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(bar, textvariable=self.zoom_var, width=5,
                  anchor=tk.CENTER).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="+", width=3,
                   command=lambda: self._apply_zoom(ZOOM_STEP))\
            .pack(side=tk.LEFT)
        ttk.Button(bar, text="Fit", width=4, command=self._zoom_fit)\
            .pack(side=tk.LEFT, padx=(4, 0))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT,
                                                    fill=tk.Y, padx=10)
        self.map_btn = ttk.Button(bar, text="Slice map",
                                  command=self._toggle_map,
                                  state=tk.DISABLED)
        self.map_btn.pack(side=tk.LEFT)
        self.rot_ccw_btn = ttk.Button(bar, text="⟲ 90°",
                                      command=lambda: self.rotate_volume(-1),
                                      state=tk.DISABLED)
        self.rot_ccw_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.rot_cw_btn = ttk.Button(bar, text="⟳ 90°",
                                     command=lambda: self.rotate_volume(1),
                                     state=tk.DISABLED)
        self.rot_cw_btn.pack(side=tk.LEFT)
        self.dim_btn = ttk.Button(bar, text="Dimensions…",
                                  command=self.edit_dimensions,
                                  state=tk.DISABLED)
        self.dim_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.crop_btn = ttk.Button(bar, text="Crop…",
                                   command=self.start_crop,
                                   state=tk.DISABLED)
        self.crop_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.roi_btn = ttk.Button(bar, text="ROI…",
                                  command=self.start_roi,
                                  state=tk.DISABLED)
        self.roi_btn.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT,
                                                    fill=tk.Y, padx=10)
        self.save_btn = ttk.Button(bar, text="Save",
                                   command=self.save, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT)
        self.save_as_btn = ttk.Button(bar, text="Save As…",
                                      command=self.save_as, state=tk.DISABLED)
        self.save_as_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.undo_btn = ttk.Button(bar, text="Undo", command=self.undo,
                                   state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.redo_btn = ttk.Button(bar, text="Redo", command=self.redo,
                                   state=tk.DISABLED)
        self.redo_btn.pack(side=tk.LEFT, padx=(6, 0))

        main = ttk.Frame(self.root, padding=(8, 0))
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Label(main, bg="#101010", anchor=tk.CENTER)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                         padx=(0, 8))
        # zoom with the mouse wheel, pan by click-dragging
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel_linux)
        self.canvas.bind("<Button-5>", self._on_wheel_linux)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)

        side = ttk.Frame(main)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(side, textvariable=self.info, justify=tk.LEFT,
                  font=("TkFixedFont", 9)).pack(anchor=tk.NW, pady=(0, 8))

        # -- appearance group -------------------------------------------------
        look = ttk.LabelFrame(side, text="Appearance", padding=(8, 4))
        look.pack(fill=tk.X, pady=(0, 6))
        color_row = ttk.Frame(look)
        color_row.pack(anchor=tk.NW)
        ttk.Label(color_row, text="Colors").pack(side=tk.LEFT)
        self.color_var = tk.StringVar(value=self.colormap)
        self.color_box = ttk.Combobox(
            color_row, textvariable=self.color_var, width=11,
            values=list(COLORMAPS.keys()), state="readonly")
        self.color_box.pack(side=tk.LEFT, padx=(6, 0))
        self.color_box.bind("<<ComboboxSelected>>", self._on_colormap_change)

        self._building_ui = True
        ttk.Label(look, text="Brightness").pack(anchor=tk.NW, pady=(6, 0))
        self.bright_var = tk.DoubleVar(value=0.0)
        self.bright_scale = ttk.Scale(look, from_=-100, to=100,
                                      orient=tk.HORIZONTAL, length=150,
                                      variable=self.bright_var,
                                      command=self._on_brightness)
        self.bright_scale.pack(anchor=tk.NW)
        ttk.Label(look, text="Contrast").pack(anchor=tk.NW)
        self.contrast_var = tk.DoubleVar(value=0.0)
        self.contrast_scale = ttk.Scale(look, from_=-100, to=100,
                                        orient=tk.HORIZONTAL, length=150,
                                        variable=self.contrast_var,
                                        command=self._on_brightness)
        self.contrast_scale.pack(anchor=tk.NW)
        ttk.Button(look, text="Reset display", width=13,
                   command=self._reset_display).pack(anchor=tk.NW,
                                                     pady=(4, 0))
        self._building_ui = False

        ttk.Label(look, text="Histogram").pack(anchor=tk.NW)
        self.hist_canvas = tk.Canvas(look, width=172, height=64,
                                     bg="#181818", highlightthickness=0)
        self.hist_canvas.pack(anchor=tk.NW, pady=(2, 0))
        self._hist_photo = None

        # -- export / compare group -------------------------------------------
        actions = ttk.LabelFrame(side, text="Export & compare",
                                 padding=(8, 4))
        actions.pack(fill=tk.X, pady=(0, 6))
        self.export_btn = ttk.Button(actions, text="Export…",
                                     command=self.export_slice_as,
                                     state=tk.DISABLED)
        self.export_btn.pack(anchor=tk.NW, pady=(2, 2))
        self.report_btn = ttk.Button(actions, text="Report…",
                                     command=self.generate_report_as,
                                     state=tk.DISABLED)
        self.report_btn.pack(anchor=tk.NW)
        self.diff_btn = ttk.Button(actions, text="Show difference",
                                   command=self._toggle_diff,
                                   state=tk.DISABLED)
        self.diff_btn.pack(anchor=tk.NW, pady=(2, 0))

        # -- slice navigation group --------------------------------------------
        nav = ttk.LabelFrame(side, text="Slice", padding=(8, 4))
        nav.pack(fill=tk.BOTH, expand=True)
        self.slice_var = tk.IntVar(value=0)
        self.slice_scale = ttk.Scale(nav, from_=0, to=0, orient=tk.VERTICAL,
                                     length=240,
                                     variable=self.slice_var,
                                     command=self._on_slice_move,
                                     state=tk.DISABLED)
        self.slice_scale.pack(fill=tk.Y, expand=True)
        self.slice_lbl = ttk.Label(nav, text="-")
        self.slice_lbl.pack(anchor=tk.NW)

        ttk.Label(self.root, textvariable=self.status, padding=(8, 4),
                  relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM,
                                                      fill=tk.X)
        self._sync_menu_states()

    # -- welcome / loading -------------------------------------------------

    def _show_welcome(self) -> None:
        self.canvas.configure(text="\n\nClick \"Open file…\" to load a\n"
                              "TIFF / NPY / DICOM file, or\n"
                              "\"Open folder…\" to load every slice\n"
                              "image inside a folder.\n\n"
                              "Then use \"Align slices\" and\n"
                              "\"Remove beam hardening\".\n",
                              fg="#9a9a9a",
                              font=("TkDefaultFont", 12))
        self.info.set("no data loaded")

    def _set_busy(self, busy: bool, msg: str = "") -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for b in (self.align_btn, self.deharden_btn, self.bg_btn,
                  self.slice_scale,
                  self.rot_ccw_btn, self.rot_cw_btn, self.dim_btn,
                  self.crop_btn, self.report_btn,
                  self.roi_btn, self.map_btn):
            if b is self.slice_scale and (self.data is None
                                          or self.data.shape[0] < 2):
                continue
            b.configure(state=state)
        if busy:
            self._close_crop_dialog()  # no editing a moving target
            self._close_roi_dialog()
        self._sync_menu_states()
        if msg:
            self.status.set(msg)
        self.root.update_idletasks()

    def open_dialog(self) -> None:
        """Open one file — or several image files at once, which are
        stacked into a single volume in natural order (Ctrl/Shift-click
        in the dialog to select many)."""
        if self._busy:
            return
        paths = filedialog.askopenfilenames(title="Open rock data "
                                            "(select one or more files)",
                                            filetypes=FILE_PATTERNS)
        if paths:
            self.load_paths([Path(p) for p in paths])

    def open_folder(self) -> None:
        if self._busy:
            return
        p = filedialog.askdirectory(
            title="Open folder of slices (images or a DICOM series)")
        if p:
            self.load_path(Path(p))

    def load_path(self, path: Path) -> None:
        """Load a single file/folder and refresh the view."""
        self.load_paths([path])

    def _confirm_discard(self) -> bool:
        """True when it is OK to proceed (nothing unsaved, or user agrees).

        The save (if chosen) runs *synchronously*: the caller is about to
        load another file, and an async write to the same path would race
        with the read.
        """
        if not (self._dirty and self.data is not None):
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved changes",
            "The data has been modified (aligned/dehardened) but not "
            "saved.\n\nSave before continuing?")
        if answer is None:
            return False  # cancel
        if answer:
            target = self._save_target()
            if target is None:
                p = filedialog.asksaveasfilename(
                    title="Save volume as", defaultextension=".tif",
                    filetypes=self.SAVE_FILETYPES)
                if not p:
                    return False  # save aborted -> keep the data loaded
                target = Path(p)
                self.path = target
                self.multi_paths = None
            try:
                save_volume(self.data, target, meta=self._meta)
            except Exception as exc:  # noqa: BLE001 - surface to the user
                messagebox.showerror("Save failed", f"{target}\n\n{exc}")
                return False
            self._saved_array = self.data
            self._save_btn_states()
            self.status.set(f"saved {target.name}")
        return True  # no -> discard; yes -> saved synchronously

    def load_paths(self, paths: list[Path]) -> None:
        """Load one file/folder (or several files) and refresh the view."""
        if not paths:
            return
        if not self._confirm_discard():
            return
        label = paths[0].name if len(paths) == 1 else f"{len(paths)} files"
        try:
            self.status.set(f"loading {label} …")
            self.root.update_idletasks()
            if len(paths) == 1:
                data = load_volume(paths[0])
            else:
                data = load_stack_from_files(paths)
        except Exception as exc:  # noqa: BLE001 - surface to the user
            messagebox.showerror("Open failed",
                                 f"{paths[0]}\n\n{exc}")
            self.status.set("Ready.")
            return

        self.data = data
        self.kind = detect_kind(data)
        self.path = paths[0] if len(paths) == 1 else None
        self.multi_paths = list(paths) if len(paths) > 1 else None
        self._reset_window()
        src_for_meta = self.path if self.path is not None \
            else (self.multi_paths[0] if self.multi_paths else None)
        self._meta = None
        if src_for_meta is not None:
            stored = read_metadata(src_for_meta)
            if stored:
                self._meta = stored
        self.slice_var.set(0)
        n = data.shape[0]
        self.slice_scale.configure(from_=0, to=max(0, n - 1),
                                   state=tk.NORMAL if n > 1 else tk.DISABLED)
        self.info.set(self._info_text())
        self.align_btn.configure(state=tk.NORMAL)
        self.compare_btn.configure(state=tk.NORMAL)
        self.map_btn.configure(state=tk.NORMAL)
        self.bg_btn.configure(state=tk.NORMAL)
        self.export_btn.configure(state=tk.NORMAL)
        self.report_btn.configure(state=tk.NORMAL)
        self._processed = False
        self._saved_array = data
        self._original_array = data
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._save_btn_states()
        self._compare_mode = None
        self._compare_b = None
        self._compare_result = None
        self._close_roi_dialog()  # a fresh file, a fresh ROI
        self._close_crop_dialog()
        self.diff_btn.configure(state=tk.DISABLED,
                                text="Show difference")
        has_raw = self.kind == "projections"
        self.deharden_btn.configure(
            state=tk.NORMAL,
            text="Remove beam hardening" if not has_raw
            else "Linearize projections (chord)")
        self._show_plane(data[0])
        self._refresh_map()
        self.status.set(f"loaded {label} — {n} "
                        f"{'slice' if n == 1 else 'slices'}")

    def _info_text(self, extra: str = "") -> str:
        d = self.data
        if self.multi_paths:
            src = f"{len(self.multi_paths)} files (stacked)"
        else:
            src = self.path.name if self.path else "-"
        lines = [
            f"file : {src}",
            f"kind : {self.kind}",
            f"shape: {d.shape[0]} x {d.shape[1]} x {d.shape[2]}",
            f"range: [{np.nanmin(d):.4g}, {np.nanmax(d):.4g}]",
            f"cupping: {cupping_index(d[len(d) // 2]):.3f}"
            if self.kind == "volume" else "cupping: n/a (raw)",
        ]
        dims = self._dims_line()
        if dims != "not set":
            lines.append(f"voxel: {dims}")
        if extra:
            lines.append(extra)
        return "\n".join(lines)

    # -- display -----------------------------------------------------------

    def _dims_line(self) -> str:
        """Physical dimensions as a short string, e.g. "0.05 x 0.05 x
        0.05 mm" ("not set" when unknown)."""
        px = (self._meta or {}).get("pixel_size")
        sp = (self._meta or {}).get("slice_spacing")
        if px and len(px) >= 2 and sp:
            return f"{px[0]:g} x {px[1]:g} x {sp:g} mm"
        if px and len(px) >= 2:
            return f"{px[0]:g} x {px[1]:g} mm"
        if sp:
            return f"{sp:g} mm / slice"
        return "not set"

    def _display_range(self, plane: np.ndarray) -> tuple[float, float]:
        """lo/hi window for ``plane``: the 1–99.5 percentile stretch,
        narrowed/widened when the brightness/contrast sliders are used."""
        if self._window is not None:
            return float(self._window[0]), float(self._window[1])
        finite = np.asarray(plane, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = np.percentile(finite, (1.0, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        return float(lo), float(hi)

    def _ensure_u8(self, plane: np.ndarray) -> np.ndarray:
        """Normalized (and colormap-applied) uint8 image for ``plane``.

        The result is cached per (plane, colormap, display window) so
        wheel-zooming and window resizes only rescale/encode, they do
        not redo the percentile stretch or LUT mapping.
        """
        key = (id(plane), self.colormap, self._window)
        if self._norm_cache is not None and self._norm_cache[0] == key \
                and self._norm_cache[1] is plane:
            return self._norm_cache[2]
        if plane.ndim == 3 and plane.shape[-1] >= 3:
            u8 = normalize_u8_plane3(plane)
        else:
            lo, hi = self._display_range(plane)
            u8 = apply_colormap(window_u8(plane, lo, hi), self.colormap)
        self._norm_cache = (key, plane, u8)
        return u8

    def _zoomed_size(self) -> tuple[int, int]:
        """Display size (h, w) of the cached image at the current zoom."""
        u8 = self._norm_cache[2]
        h, w = u8.shape[:2]
        return max(1, round(h * self.zoom)), max(1, round(w * self.zoom))

    def _encode_photo(self) -> None:
        """Re-encode the cached uint8 image at the current zoom/size."""
        u8 = self._norm_cache[2]
        src_h, src_w = u8.shape[:2]
        th, tw = self._zoomed_size()
        # keep the encoded image within a sane pixel budget
        budget = 32_000_000
        if th * tw > budget:
            s = (budget / (th * tw)) ** 0.5
            th, tw = max(1, int(th * s)), max(1, int(tw * s))
        if (th, tw) != (src_h, src_w):
            ys = np.clip((np.arange(th) * src_h / th).astype(np.int64),
                         0, src_h - 1)
            xs = np.clip((np.arange(tw) * src_w / tw).astype(np.int64),
                         0, src_w - 1)
            u8 = np.ascontiguousarray(u8[ys][:, xs])
        if self._crop_sel is not None and self._crop_win is not None:
            _shape = (self._crop_shape_var.get() == "Ellipse"
                      if self._crop_shape_var is not None else False)
            u8 = paint_crop_overlay(u8, self._crop_sel,
                                    (src_h, src_w),
                                    shape="ellipse" if _shape else "rect")
        if self._roi_sel is not None and self._roi_win is not None:
            u8 = paint_crop_overlay(u8, self._roi_sel,
                                    (src_h, src_w), color="cyan")
        img = tk.PhotoImage(data=to_pgm(u8))
        self._photo = img  # keep a reference or Tk garbage-collects it
        self.canvas.configure(image=img)

    def _show_plane(self, plane: np.ndarray, keep_view: bool = False) -> None:
        """Render one 2-D plane (grayscale or multichannel).

        ``keep_view=True`` preserves the current zoom and pan (used when
        the slice slider moves); otherwise the view resets to 100%.
        """
        self.displayed = plane
        self._shown_index = (int(self.slice_var.get())
                             if self.data is not None else None)
        if not keep_view:
            self.zoom = 1.0
            self._pan = (0, 0)
            self._update_zoom_label()
        self._ensure_u8(plane)
        self.canvas.configure(text="", image=None, bg="#101010")
        self._encode_photo()
        self._apply_pan()  # place the pan offset on the fresh label
        self._draw_histogram()
        idx = int(self.slice_var.get())
        self.slice_lbl.configure(
            text=f"{idx + 1} / {self.data.shape[0]}"
            if self.data is not None else "-")

    def _redisplay(self) -> None:
        """Re-encode at the current zoom (after zoom/pan/resize changes)."""
        if self._norm_cache is None:
            return
        self._encode_photo()
        self._apply_pan()

    def _update_zoom_label(self) -> None:
        self.zoom_var.set(f"{round(self.zoom * 100)}%")

    def _apply_zoom(self, factor: float) -> None:
        if self._norm_cache is None:
            return
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * factor))
        self._update_zoom_label()
        self._redisplay()

    def _zoom_fit(self) -> None:
        if self._norm_cache is None:
            return
        u8 = self._norm_cache[2]
        h, w = u8.shape[:2]
        self.canvas.update_idletasks()
        avail_h = max(1, self.canvas.winfo_height())
        avail_w = max(1, self.canvas.winfo_width())
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX,
                                      min(avail_w / w, avail_h / h)))
        self._pan = (0, 0)
        self._update_zoom_label()
        self._redisplay()

    def _apply_pan(self) -> None:
        dx, dy = self._pan
        self.canvas.place_configure(x=dx, y=dy)

    def _on_wheel(self, event) -> None:
        if self._norm_cache is None:
            return
        self._apply_zoom(ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP)

    def _on_wheel_linux(self, event) -> None:
        if self._norm_cache is None:
            return
        self._apply_zoom(ZOOM_STEP if event.num == 4 else 1 / ZOOM_STEP)

    # -- crop (drag a rectangle, applies to every slice) ---------------------

    def start_crop(self) -> None:
        """Enter crop mode: drag a rectangle on the image, then Apply."""
        if self.data is None or self._busy or self._norm_cache is None:
            return
        if self._compare_mode is not None:
            self._exit_compare()
        if self._crop_win is not None:
            self._close_crop_dialog()
        self._crop_sel = None
        self._redisplay()  # redraw the current view (clears stale marks)
        self.root.update_idletasks()  # settle geometry for coord mapping
        dlg = tk.Toplevel(self.root)
        dlg.title("Crop")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        shape_row = ttk.Frame(frm)
        shape_row.pack(anchor=tk.W)
        ttk.Label(shape_row, text="Shape:").pack(side=tk.LEFT)
        shape_var = tk.StringVar(value="Rectangle")
        shape_box = ttk.Combobox(shape_row, textvariable=shape_var,
                                 width=10, state="readonly",
                                 values=["Rectangle", "Ellipse"])
        shape_box.pack(side=tk.LEFT, padx=(4, 12))
        lbl_var = tk.StringVar(
            value="Drag a region on the image, then press Apply.\n"
                  "Rectangle: the stack shrinks to the selection.\n"
                  "Ellipse: the outside is filled with NaN (transparent).\n"
                  "Applies to the whole stack.\n\nNothing selected yet.")
        ttk.Label(frm, textvariable=lbl_var, justify=tk.LEFT).pack(
            anchor=tk.W, pady=(6, 0))
        btns = ttk.Frame(frm)
        btns.pack(anchor=tk.E, pady=(10, 0))
        apply_btn = ttk.Button(btns, text="Apply", command=self.apply_crop,
                               state=tk.DISABLED)
        apply_btn.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Cancel",
                   command=self._close_crop_dialog).pack(side=tk.LEFT)
        dlg.protocol("WM_DELETE_WINDOW", self._close_crop_dialog)
        self._crop_win = dlg
        self._crop_apply_btn = apply_btn
        self._crop_lbl_var = lbl_var
        self._crop_shape_var = shape_var
        self.status.set("crop mode: drag a region on the image "
                        "(Escape cancels)")

    def _close_crop_dialog(self) -> None:
        """Leave crop mode: drop the dialog and the painted selection."""
        win = getattr(self, "_crop_win", None)
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass
            self._crop_win = None
        self._crop_sel = None
        self._crop_apply_btn = None
        self._crop_lbl_var = None
        self._crop_shape_var = None
        if self._norm_cache is not None and not self._busy:
            self._redisplay()  # repaint without the rectangle

    def apply_crop(self) -> None:
        """Apply the dragged selection to every slice (undoable)."""
        if self.data is None or self._busy or self._crop_sel is None:
            return
        # read the dialog controls *before* closing it (closing clears
        # the vars); ellipse crops always fill the outside with NaN
        shape_var = getattr(self, "_crop_shape_var", None)
        shape = shape_var.get() if shape_var is not None else "Rectangle"
        x0, y0, x1, y1 = self._crop_sel
        x1i, y1i = x1 + 1, y1 + 1  # rect is inclusive -> python slicing
        self._close_crop_dialog()  # also repaints without the rectangle
        prev = self.data
        self._push_history(prev, "crop")
        if shape == "Ellipse":
            self.data = crop_ellipse(prev, (x0, y0, x1, y1), fill="nan")
        else:
            self.data = prev[:, y0:y1i, x0:x1i]
        self._processed = True
        idx = max(0, min(int(self.slice_var.get()),
                         self.data.shape[0] - 1))
        self.slice_var.set(idx)
        self._save_btn_states()
        h, w = self.data.shape[1:3]
        self.info.set(self._info_text())
        self._show_plane(self.data[idx])
        self._refresh_map()
        what = "ellipse (outside NaN)" if shape == "Ellipse" \
            else f"{w} x {h} px"
        self.status.set(f"cropped: {what} — "
                        f"{self.data.shape[0]} "
                        f"{'slice' if self.data.shape[0] == 1 else 'slices'}"
                        " kept")

    def _on_escape(self, _event=None) -> None:
        """Escape: leave crop/ROI mode if active, else compare mode."""
        if self._crop_win is not None:
            self._close_crop_dialog()
            self.status.set("crop cancelled.")
            return
        if self._roi_win is not None:
            self._close_roi_dialog()
            self.status.set("ROI mode closed.")
            return
        self._exit_compare()

    def _crop_rect_from_event(self, event) -> tuple | None:
        """Map a widget event to a source-image pixel rectangle.

        The canvas is a Label that shrink-wraps the displayed photo with
        ~2 px of padding, so the photo origin is roughly at (2, 2); the
        small residual offset is harmless for interactive cropping.
        Returns ``(x0, y0, x1, y1)`` in source coordinates (inclusive,
        corners normalized regardless of drag direction).
        """
        if self._norm_cache is None or self.data is None:
            return None
        src_h, src_w = self._norm_cache[2].shape[:2]
        pad = 2  # label border/padding around the photo
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:  # window not laid out yet
            cw, ch = src_w + 2 * pad, src_h + 2 * pad
        # display->source: reverse of _encode_photo's nearest sampling
        x0 = min(max((event.x - pad) * src_w / max(1, cw - 2 * pad), 0),
                 src_w - 1)
        y0 = min(max((event.y - pad) * src_h / max(1, ch - 2 * pad), 0),
                 src_h - 1)
        return x0, y0

    def _on_drag_start(self, event) -> None:
        if self._crop_win is not None:
            self._drag_start = self._crop_rect_from_event(event)
            return
        if self._roi_win is not None:
            self._drag_start = self._crop_rect_from_event(event)
            return
        self._drag_start = (event.x, event.y)

    def _on_drag_move(self, event) -> None:
        if self._drag_start is None:
            return
        if self._crop_win is not None:
            self._update_crop_rect(event)
            return
        if self._roi_win is not None:
            self._update_roi_rect(event)
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._pan = (self._pan[0] + dx, self._pan[1] + dy)
        self._drag_start = (event.x, event.y)
        self._apply_pan()

    # -- long tasks (worker thread + scheduled result) ---------------------

    def _run_task(self, label: str, work, on_done) -> None:
        """Run ``work`` in a background thread, apply result on the UI.

        Results travel through a ``queue.Queue``; a timer scheduled on
        the *UI* thread polls for completion. (Calling ``root.after``
        from the worker thread directly is not allowed without a
        running mainloop.)
        """
        if self._busy:
            return
        self._set_busy(True, f"{label} … (working)")
        q: queue.Queue = queue.Queue()

        def runner():
            try:
                q.put(("ok", work()))
            except Exception as exc:  # noqa: BLE001 - surface to the user
                q.put(("err", exc))

        threading.Thread(target=runner, daemon=True).start()
        self._poll_task(label, q, on_done)

    def _poll_task(self, label, q, on_done) -> None:
        try:
            state, payload = q.get_nowait()
        except queue.Empty:
            self.root.after(50, lambda: self._poll_task(label, q, on_done))
            return
        self._set_busy(False)
        if state == "err":
            messagebox.showerror(label, str(payload))
            self.status.set(f"{label} failed.")
            return
        on_done(payload)

    # -- rotate (volume-level, undoable) ------------------------------------

    def rotate_volume(self, k: int) -> None:
        """Rotate the whole stack by ``k * 90`` degrees (k=+1 clockwise).

        Rotation happens in-plane (the axes rotated are rows/columns of
        every slice); slice count and order are unchanged. Undoable via
        the usual history mechanism.
        """
        if self.data is None or self._busy:
            return
        if self._compare_mode is not None:
            self._exit_compare()
        prev = self.data
        self._push_history(prev, "rotate 90°")
        self.data = np.rot90(prev, k=k, axes=(1, 2))
        self._processed = True
        idx = max(0, min(int(self.slice_var.get()),
                         self.data.shape[0] - 1))
        self.slice_var.set(idx)
        self._save_btn_states()
        self.info.set(self._info_text())
        self._show_plane(self.data[idx])
        self._refresh_map()
        self.status.set("rotated 90° clockwise." if k > 0
                        else "rotated 90° counterclockwise.")

    # -- brightness / contrast (display-only window) -------------------------

    def _compute_window(self) -> tuple[float, float] | None:
        """Display window implied by the brightness/contrast sliders.

        The base is the 1–99.5 percentile range of the current plane;
        brightness shifts both bounds, contrast narrows (positive) or
        widens (negative) the window. Returns ``None`` when nothing is
        displayed.
        """
        if self._norm_cache is None:
            return None
        plane = self._norm_cache[1]
        finite = np.asarray(plane, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        lo, hi = np.percentile(finite, (1.0, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        bright = float(self.bright_var.get())
        contrast = float(self.contrast_var.get())
        half = (hi - lo) / 2.0
        if contrast >= 0:
            half *= 1.0 - 0.9 * contrast / 100.0
        else:
            half *= 1.0 - contrast / 100.0
        mid = (lo + hi) / 2.0 + bright / 100.0 * (hi - lo)
        return mid - half, mid + half

    def _on_brightness(self, _value=None) -> None:
        """Slider callback: re-render with the new display window."""
        if getattr(self, "_building_ui", False) or self._busy \
                or self.data is None:
            return
        if self._compare_mode is not None:
            self._show_compare_view()
            return
        self._window = self._compute_window()
        self._show_plane(self.displayed, keep_view=True)
        self._draw_histogram()

    def _reset_window(self) -> None:
        """Clear the manual display window and the brightness sliders."""
        self._window = None
        self._building_ui = True
        try:
            self.bright_var.set(0.0)
            self.contrast_var.set(0.0)
        finally:
            self._building_ui = False

    def _reset_display(self) -> None:
        """Back to 100% brightness/contrast and the plain stretch."""
        self._reset_window()
        if self.data is None or self.displayed is None:
            return
        self._show_plane(self.displayed, keep_view=True)
        self._draw_histogram()
        self.status.set("display reset (100% brightness/contrast).")

    # -- histogram panel -----------------------------------------------------

    def _draw_histogram(self) -> None:
        """Redraw the tiny histogram of the currently displayed image."""
        cv = self.hist_canvas
        cv.delete("all")
        self._hist_photo = None
        if self._norm_cache is None:
            return
        u8 = self._norm_cache[2]
        counts = histogram_counts(u8[..., 0] if u8.ndim == 3 else u8)
        peak = int(counts.max()) if counts.size else 0
        if peak == 0:
            return
        try:
            w = max(40, int(cv.winfo_width()))
            h = max(20, int(cv.winfo_height()))
        except tk.TclError:
            w, h = 172, 64
        bw = w / counts.size
        color = "#40c070" if self.colormap == DEFAULT_COLORMAP \
            else "#e0a040"
        for i, c in enumerate(counts):
            bh = max(1, int(round(c / peak * (h - 4))))
            x0 = int(round(i * bw)) + 1
            x1 = max(x0 + 1, int(round((i + 1) * bw)) - 1)
            cv.create_rectangle(x0, h - bh, x1, h, fill=color, width=0)
        label = "histogram"
        if self._shown_index is not None:
            label += f" — slice {self._shown_index + 1}"
        cv.create_text(3, 3, anchor=tk.NW, fill="#808080", text=label,
                       font=("TkDefaultFont", 7))

    # -- crop rectangle painting --------------------------------------------

    def _update_crop_rect(self, event) -> None:
        """Track the dragged rectangle and repaint the view with it."""
        cur = self._crop_rect_from_event(event)
        if cur is None or self._drag_start is None:
            return
        (sx, sy), (cx, cy) = self._drag_start, cur
        x0, x1 = sorted((int(round(sx)), int(round(cx))))
        y0, y1 = sorted((int(round(sy)), int(round(cy))))
        src_h, src_w = self._norm_cache[2].shape[:2]
        x1 = min(x1, src_w - 1)
        y1 = min(y1, src_h - 1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return  # degenerate (need at least a 2 x 2 px region)
        self._crop_sel = (x0, y0, x1, y1)
        self._redisplay()  # _encode_photo paints _crop_sel onto the image
        w_px, h_px = x1 - x0 + 1, y1 - y0 + 1
        if self._crop_lbl_var is not None:
            self._crop_lbl_var.set(
                f"Selection: {w_px} x {h_px} px "
                f"(x {x0}–{x1}, y {y0}–{y1})\n"
                "Applies to every slice. Escape or Cancel to abort.")
        if self._crop_apply_btn is not None:
            self._crop_apply_btn.configure(state=tk.NORMAL)

    # -- actions -----------------------------------------------------------

    def run_align(self) -> None:
        if self.data is None:
            return
        kind = self.kind
        data = self.data

        def work():
            if kind == "projections":
                from .align import align_projections
                aligned, _shifts = align_projections(data, layout="detector")
                return aligned
            aligned, _res = align_slices(data, estimate_theta=True)
            return aligned

        def done(aligned):
            self._push_history(data, "align slices")
            self.data = np.asarray(aligned)
            self._processed = True
            self._save_btn_states()
            self.info.set(self._info_text())
            self.slice_var.set(0)
            self._show_plane(self.data[0])
            self._refresh_map()
            self.status.set("alignment finished.")

        self._run_task("Align slices", work, done)

    def run_deharden(self) -> None:
        if self.data is None:
            return
        kind, data = self.kind, self.data

        def work():
            if kind == "projections":
                corrected, _info = remove_beam_hardening(
                    data, method="chord", layout="detector")
                return corrected
            corrected, _res = remove_beam_hardening(data, method="radial")
            return corrected

        def done(corrected):
            self._push_history(data, "deharden")
            self.data = np.asarray(corrected)
            self._processed = True
            self._save_btn_states()
            self.info.set(self._info_text())
            self.slice_var.set(0)
            self._show_plane(self.data[0])
            self._refresh_map()
            self.status.set("beam-hardening correction finished.")

        self._run_task("Beam hardening", work, done)

    # -- background correction -----------------------------------------------

    def run_background(self) -> None:
        """Ask for background-correction options, then run it."""
        if self.data is None or self._busy:
            return
        if self._compare_mode is not None:
            self._exit_compare()
        dlg = tk.Toplevel(self.root)
        dlg.title("Background correction")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Flatten the background around the sample:")\
            .grid(row=0, column=0, columnspan=2, sticky=tk.W)
        mode_var = tk.StringVar(value="Subtract (air becomes 0)")
        ttk.Label(frm, text="Mode").grid(row=1, column=0, sticky=tk.W,
                                         pady=(6, 0))
        ttk.Combobox(frm, textvariable=mode_var, width=28,
                     state="readonly",
                     values=["Subtract (air becomes 0)",
                             "Divide by shading"])\
            .grid(row=1, column=1, sticky=tk.W, pady=(6, 0), padx=(8, 0))
        per_slice_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Estimate per slice (else one shared "
                        "surface)", variable=per_slice_var)\
            .grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        bg_var = tk.StringVar(value="dark (air around bright rock)")
        ttk.Label(frm, text="Background").grid(row=3, column=0,
                                               sticky=tk.W, pady=(6, 0))
        ttk.Combobox(frm, textvariable=bg_var, width=28, state="readonly",
                     values=["dark (air around bright rock)",
                             "bright (bright holder)"])\
            .grid(row=3, column=1, sticky=tk.W, pady=(6, 0), padx=(8, 0))
        air_var = tk.StringVar(value="1.0")
        ttk.Label(frm, text="Air percentile").grid(row=4, column=0,
                                                   sticky=tk.W, pady=(6, 0))
        ttk.Entry(frm, textvariable=air_var, width=8)\
            .grid(row=4, column=1, sticky=tk.W, pady=(6, 0), padx=(8, 0))
        ttk.Label(frm, text="Estimates a smooth background surface per "
                  "slice and removes it.\nComplements 'Remove beam "
                  "hardening', which fixes cupping\ninside the sample.",
                  justify=tk.LEFT, foreground="#808080")\
            .grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

        def run():
            mode = "offset" if mode_var.get().startswith("Subtract") \
                else "divide"
            bg = "dark" if bg_var.get().startswith("dark") else "bright"
            try:
                pct = float(air_var.get())
            except ValueError:
                pct = 1.0
            per_slice = bool(per_slice_var.get())  # read on the UI thread
            data = self.data

            def work():
                return correct_background(data, mode=mode,
                                          per_slice=per_slice,
                                          air_percentile=pct,
                                          background=bg)

            def done(payload):
                corrected, res = payload
                self._push_history(data, "background correction")
                self.data = np.asarray(corrected)
                self._processed = True
                self._save_btn_states()
                self.info.set(self._info_text())
                self.slice_var.set(0)
                self._show_plane(self.data[0])
                self._refresh_map()
                self.status.set(
                    "background correction finished — air "
                    f"{res.air_before:.4g} -> {res.air_after:.4g}, "
                    "bg spread "
                    f"{res.background_std_before:.4g} -> "
                    f"{res.background_std_after:.4g}")

            dlg.destroy()
            self._run_task("Background", work, done)

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=dlg.destroy)\
            .pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Run", command=run).pack(side=tk.LEFT)

    # -- export current slice ------------------------------------------------

    def export_slice_as(self) -> None:
        """Export the current slice as PNG, SVG, PDF or CSV."""
        if self._busy:
            self.status.set("busy — wait for the current task to finish.")
            return
        if self.data is None or self._norm_cache is None:
            return
        if self._compare_mode is not None:
            self._exit_compare()
        dlg = tk.Toplevel(self.root)
        dlg.title("Export current slice")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        idx = max(0, min(int(self.slice_var.get()),
                         self.data.shape[0] - 1))
        ttk.Label(frm, text=f"Export slice {idx + 1} of "
                  f"{self.data.shape[0]} as:").grid(row=0, column=0,
                                                    sticky=tk.W)
        fmt_var = tk.StringVar(value=EXPORT_FORMATS[0][0])
        ttk.Combobox(frm, textvariable=fmt_var, width=26,
                     state="readonly",
                     values=[label for label, _ in EXPORT_FORMATS])\
            .grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Label(frm, text="PNG/PDF: rendered view tagged with the "
                  "physical DPI.\nSVG: true vector, sized in millimeters.\n"
                  "CSV: raw slice values (row, col, value).",
                  justify=tk.LEFT, foreground="#808080")\
            .grid(row=2, column=0, sticky=tk.W, pady=(8, 0))

        def run():
            ext = dict(EXPORT_FORMATS).get(fmt_var.get(), ".png")
            stem = self.path.stem if self.path else "slice"
            suggested = f"{stem}_slice{idx + 1}{ext}"
            p = filedialog.asksaveasfilename(
                title="Export slice as", defaultextension=ext,
                initialfile=suggested, parent=dlg)
            if not p:
                return  # keep the dialog open for another try
            target = Path(p)
            # render fresh from the exported slice (clean, no crop
            # overlay) with the current window and color system
            u8 = self._ensure_u8(self.data[idx])
            data, sidx = self.data, idx
            dlg.destroy()

            def work():
                return export_slice(data, u8, target, meta=self._meta,
                                    slice_index=sidx,
                                    colormap=self.colormap)

            def done(path):
                self.status.set(f"exported slice {idx + 1} "
                                f"-> {path.name}")

            self._run_task("Export", work, done)

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=dlg.destroy)\
            .pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Export", command=run).pack(side=tk.LEFT)

    # -- report generation ---------------------------------------------------

    def _session_history(self) -> list[tuple[str, str]]:
        """Session modification history as (label, action) tuples."""
        out = []
        for _arr, label in self._undo_stack:  # stack holds (array, label)
            out.append((label, "did"))
        out.reverse()  # oldest first
        for _arr, label in self._redo_stack:
            out.append((label, "undid"))
        return out

    def generate_report_as(self) -> None:
        """Ask for a destination and write a dataset report."""
        if self._busy:
            self.status.set("busy — wait for the current task to finish.")
            return
        if self.data is None:
            return
        if self._compare_mode is not None:
            self._exit_compare()
        dlg = tk.Toplevel(self.root)
        dlg.title("Generate report")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Write a summary report of the current "
                  "dataset:").grid(row=0, column=0, sticky=tk.W)
        fmt_var = tk.StringVar(value="HTML report, with images (*.html)")
        ttk.Combobox(frm, textvariable=fmt_var, width=34,
                     state="readonly",
                     values=["HTML report, with images (*.html)",
                             "PDF document, with images (*.pdf)",
                             "Word document, with images (*.docx)",
                             "Plain text (*.txt)"])\
            .grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Label(frm, text="Contains shape, value range, physical "
                  "dimensions,\ncupping / air level / background "
                  "spread, the modification\nhistory of this session, "
                  "compare results — plus the current-slice\nsnapshot "
                  "and histogram (HTML/PDF/Word).",
                  justify=tk.LEFT, foreground="#808080")\
            .grid(row=2, column=0, sticky=tk.W, pady=(8, 0))

        def run():
            fmt_map = {"HTML": ".html", "PDF": ".pdf",
                       "Word": ".docx", "Plain": ".txt"}
            ext = next((e for prefix, e in fmt_map.items()
                        if fmt_var.get().startswith(prefix)), ".html")
            stem = self.path.stem if self.path else "dataset"
            suggested = f"{stem}_report{ext}"
            p = filedialog.asksaveasfilename(
                title="Generate report as", defaultextension=ext,
                initialfile=suggested, parent=dlg)
            if not p:
                return  # keep the dialog open
            target = Path(p)
            idx = max(0, min(int(self.slice_var.get()),
                             self.data.shape[0] - 1))
            with_images = ext in {".html", ".pdf", ".docx"}
            u8 = self._ensure_u8(self.data[idx]) if with_images else None
            rd = collect_report_data(
                self.data, kind=self.kind,
                source=self.path.name if self.path else
                (f"{len(self.multi_paths)} stacked files"
                 if self.multi_paths else None),
                meta=self._meta, slice_index=idx if with_images else None,
                u8=u8, colormap=self.colormap if with_images else None,
                history=self._session_history(),
                processed=self._processed,
                compare_result=self._compare_result)
            data, sidx = self.data, idx
            dlg.destroy()

            def work():
                return generate_report(rd, target, data=data,
                                       u8=u8 if with_images else None)

            def done(path):
                self.status.set(f"report written: {path.name}")

            self._run_task("Report", work, done)

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=dlg.destroy)\
            .pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Generate", command=run).pack(side=tk.LEFT)

    # -- ROI measurement ------------------------------------------------------

    def start_roi(self) -> None:
        """Enter ROI mode: drag a rectangle, live stats per slice."""
        if self.data is None or self._busy or self._norm_cache is None:
            return
        if self._compare_mode is not None:
            self._exit_compare()
        if self._roi_win is not None:
            self._close_roi_dialog()
        self._roi_sel = None
        self._redisplay()
        self.root.update_idletasks()
        dlg = tk.Toplevel(self.root)
        dlg.title("ROI measurement")
        dlg.transient(self.root)
        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        stats_var = tk.StringVar(
            value="Drag a rectangle on the image to measure it.\n"
                  "The statistics update as you scroll slices.")
        ttk.Label(frm, textvariable=stats_var, justify=tk.LEFT,
                  font=("TkFixedFont", 9)).pack(anchor=tk.W)
        btns = ttk.Frame(frm)
        btns.pack(anchor=tk.E, pady=(10, 0))

        def crop_to_roi():
            self.apply_roi_crop()

        def save_csv():
            self.save_roi_csv()

        ttk.Button(btns, text="Crop to ROI…", command=crop_to_roi)\
            .pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Save all slices (CSV)…",
                   command=save_csv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Close",
                   command=self._close_roi_dialog).pack(side=tk.LEFT)
        dlg.protocol("WM_DELETE_WINDOW", self._close_roi_dialog)
        self._roi_win = dlg
        self._roi_stats_var = stats_var
        self.status.set("ROI mode: drag a rectangle on the image "
                        "(Escape cancels)")

    def _close_roi_dialog(self) -> None:
        win = getattr(self, "_roi_win", None)
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass
            self._roi_win = None
        self._roi_sel = None
        self._roi_stats_var = None
        if self._norm_cache is not None and not self._busy:
            self._redisplay()

    def _update_roi_rect(self, event) -> None:
        """Track the dragged ROI and re-render with the outline."""
        cur = self._crop_rect_from_event(event)
        if cur is None or self._drag_start is None:
            return
        (sx, sy), (cx, cy) = self._drag_start, cur
        x0, x1 = sorted((int(round(sx)), int(round(cx))))
        y0, y1 = sorted((int(round(sy)), int(round(cy))))
        src_h, src_w = self._norm_cache[2].shape[:2]
        x1 = min(x1, src_w - 1)
        y1 = min(y1, src_h - 1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return  # need at least a 2 x 2 px region
        self._roi_sel = (x0, y0, x1, y1)
        self._redisplay()
        self._update_roi_stats_text()

    def _update_roi_stats_text(self) -> None:
        """Recompute the ROI statistics for the current slice."""
        if self._roi_stats_var is None or self._roi_sel is None \
                or self.data is None:
            return
        idx = max(0, min(int(self.slice_var.get()),
                         self.data.shape[0] - 1))
        try:
            from .constants import AIR_HU
            s = roi_stats(self.data, self._roi_sel, idx, meta=self._meta,
                          air_reference=AIR_HU)
        except (ValueError, IndexError) as exc:
            self._roi_stats_var.set(f"ROI invalid: {exc}")
            return
        area = f"\narea      : {s.area_mm2:.4g} mm²" \
            if s.area_mm2 is not None else ""
        above = (f"\nmean-above-air: {s.mean_above_air:.6g}"
                 if s.mean_above_air is not None else "")
        self._roi_stats_var.set(
            f"ROI (slice {idx + 1}): {s.rect[2] - s.rect[0] + 1} x "
            f"{s.rect[3] - s.rect[1] + 1} px = {s.n_pixels} px"
            f"\nmean      : {s.mean:.6g}"
            f"\nstd       : {s.std:.6g}"
            f"\nmin / max : {s.minimum:.6g} / {s.maximum:.6g}"
            f"\nmedian    : {s.median:.6g}"
            f"{area}{above}")

    def apply_roi_crop(self) -> None:
        """Crop the stack to the measured ROI (undoable)."""
        if self.data is None or self._busy or self._roi_sel is None:
            return
        x0, y0, x1, y1 = self._roi_sel
        self._close_roi_dialog()
        prev = self.data
        self._push_history(prev, "crop to ROI")
        self.data = prev[:, y0:y1 + 1, x0:x1 + 1]
        self._processed = True
        idx = max(0, min(int(self.slice_var.get()),
                         self.data.shape[0] - 1))
        self.slice_var.set(idx)
        self._save_btn_states()
        h, w = self.data.shape[1:3]
        self.info.set(self._info_text())
        self._show_plane(self.data[idx])
        self._refresh_map()
        self.status.set(f"cropped to ROI: {w} x {h} px")

    def save_roi_csv(self) -> None:
        """Save per-slice ROI measurements as CSV."""
        if self.data is None or self._busy or self._roi_sel is None:
            return
        rect = self._roi_sel
        meta = self._meta
        data = self.data
        p = filedialog.asksaveasfilename(
            title="Save ROI measurements as", defaultextension=".csv",
            initialfile=((self.path.stem if self.path else "roi")
                         + "_roi.csv"))
        if not p:
            return
        target = Path(p)

        def work():
            return write_roi_csv(
                roi_stats_series(data, rect, meta=meta, with_air=True),
                target)

        def done(path):
            self.status.set(f"ROI measurements saved: {Path(path).name}")

        self._run_task("ROI CSV", work, done)

    # -- compare -----------------------------------------------------------

    def run_compare(self) -> None:
        if self.data is None or self._busy:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("Compare with another dataset")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Compare current data with:")\
            .grid(row=0, column=0, sticky=tk.W)
        path_var = tk.StringVar()
        ttk.Entry(frm, textvariable=path_var, width=34)\
            .grid(row=1, column=0, sticky=tk.W, pady=(2, 6))

        def browse():
            p = filedialog.askopenfilename(
                parent=dlg, title="Dataset to compare with",
                filetypes=FILE_PATTERNS)
            if p:
                path_var.set(p)

        ttk.Button(frm, text="Browse…", command=browse)\
            .grid(row=1, column=1, padx=(6, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(btns, text="Cancel", command=dlg.destroy)\
            .pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Compare",
                   command=lambda: self._do_compare(dlg, path_var))\
            .pack(side=tk.LEFT)

    def _do_compare(self, dlg, path_var) -> None:
        p = path_var.get().strip()
        if not p:
            return
        if dlg is not None:
            dlg.destroy()
        other = Path(p)
        mine = self.data

        def work():
            return load_volume(other), compare_volumes(mine,
                                                       load_volume(other))

        def done(payload):
            other_data, res = payload
            self._compare_b = other_data
            self._compare_result = res
            self._compare_mode = "side"
            self.diff_btn.configure(state=tk.NORMAL)
            self._sync_menu_states()
            self._show_compare_view()
            self.status.set(res.summary())

        self._run_task("Compare", work, done)

    def _show_compare_view(self) -> None:
        """Render the side-by-side or difference view for one slice."""
        a, b = self.data, self._compare_b
        if a is None or b is None:
            return
        idx = max(0, min(int(self.slice_var.get()),
                         min(a.shape[0], b.shape[0]) - 1))
        pa = a[min(idx, a.shape[0] - 1)]
        pb = b[min(idx, b.shape[0] - 1)]
        h = min(pa.shape[0], pb.shape[0])
        w = min(pa.shape[1], pb.shape[1])
        pa = pa[:h, :w]
        pb = pb[:h, :w]

        if self._compare_mode == "diff":
            self._show_plane(pa - pb, keep_view=True)
        else:  # side by side
            gap = 4
            canvas = np.full((h, w * 2 + gap), np.nan)
            canvas[:, :w] = pa
            canvas[:, w + gap:] = pb
            self._show_plane(canvas, keep_view=True)
        self.slice_lbl.configure(
            text=f"{idx + 1} / {min(a.shape[0], b.shape[0])}  "
                 f"(Escape: exit)")

    def _exit_compare(self, _event=None) -> None:
        """Leave compare mode and show the plain slice view again."""
        if self._compare_mode is None:
            return
        self._compare_mode = None
        self._compare_b = None
        self.diff_btn.configure(state=tk.DISABLED, text="Show difference")
        self._sync_menu_states()
        if self.data is not None and not self._busy:
            idx = max(0, min(int(self.slice_var.get()),
                             self.data.shape[0] - 1))
            self._show_plane(self.data[idx])
        self.status.set("compare view closed.")

    def _toggle_diff(self) -> None:
        """Switch the compare view between side-by-side and difference."""
        if self._compare_b is None or self._busy:
            return
        self._compare_mode = "side" if self._compare_mode == "diff" \
            else "diff"
        self.diff_btn.configure(
            text="Show side-by-side" if self._compare_mode == "diff"
            else "Show difference")
        self._show_compare_view()

    # -- physical dimensions (voxel size in mm) -------------------------------

    def edit_dimensions(self) -> None:
        """Edit the physical dimensions (mm) shown and saved with the data."""
        if self.data is None or self._busy:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("Physical dimensions")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        meta = dict(self._meta or {})
        px = meta.get("pixel_size")
        if isinstance(px, (int, float)):
            px = [px, px]
        px = list(px) if px else [None, None]
        sp = meta.get("slice_spacing")
        ttk.Label(frm, text="Voxel dimensions in millimeters:").grid(
            row=0, column=0, columnspan=2, sticky=tk.W)
        labels = ["Pixel width (x, mm):", "Pixel height (y, mm):",
                  "Slice spacing (z, mm):"]
        defaults = [px[0] if len(px) > 0 else None,
                    px[1] if len(px) > 1 else None, sp]
        vars_ = []
        for i, (lab, val) in enumerate(zip(labels, defaults), start=1):
            ttk.Label(frm, text=lab).grid(row=i, column=0, sticky=tk.W,
                                          pady=(4, 0))
            v = tk.StringVar(value="" if val is None else f"{float(val):g}")
            ttk.Entry(frm, textvariable=v, width=12).grid(
                row=i, column=1, sticky=tk.W, pady=(4, 0), padx=(8, 0))
            vars_.append(v)
        hint = ("Shown in the info panel and embedded when saving "
                "TIFF/NPZ.\nDICOM files carry it in PixelSpacing / "
                "SliceThickness.")
        ttk.Label(frm, text=hint, justify=tk.LEFT, foreground="#808080")\
            .grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

        def ok():
            vals = []
            for v in vars_:
                s = v.get().strip()
                if not s:
                    vals.append(None)
                    continue
                try:
                    vals.append(float(s))
                except ValueError:
                    messagebox.showerror("Physical dimensions",
                                         f"Not a number: {s!r}")
                    return
            if all(v is None for v in vals):
                self._meta = None
            else:
                width, height, spacing = vals
                meta = dict(self._meta or {})
                if width is not None or height is not None:
                    prev = meta.get("pixel_size") or [None, None]
                    if isinstance(prev, (int, float)):
                        prev = [prev, prev]
                    meta["pixel_size"] = [
                        width if width is not None else prev[0],
                        height if height is not None else prev[1]]
                if spacing is not None:
                    meta["slice_spacing"] = spacing
                self._meta = meta
            dlg.destroy()
            self.info.set(self._info_text())
            self.status.set("physical dimensions updated." if self._meta
                            else "physical dimensions cleared.")

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=dlg.destroy)\
            .pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="OK", command=ok).pack(side=tk.LEFT)

    def _on_slice_move(self, _event=None) -> None:
        if self._busy or self.data is None:
            return
        idx = int(round(self.slice_var.get()))
        idx = max(0, min(idx, self.data.shape[0] - 1))
        if self._compare_mode is not None:
            self._show_compare_view()
            return
        if idx == self._shown_index:
            return
        self._show_plane(self.data[idx], keep_view=True)
        self._map_highlight()
        if self._roi_win is not None and self._roi_sel is not None:
            self._update_roi_stats_text()

    def _on_colormap_change(self, _event=None) -> None:
        """Re-render the current slice in the newly selected color system."""
        self.colormap = self.color_var.get()
        if self._busy or self.data is None or self._shown_index is None:
            return
        if self._compare_mode is not None:
            self._show_compare_view()
            return
        idx = max(0, min(self._shown_index, self.data.shape[0] - 1))
        self._show_plane(self.data[idx], keep_view=True)
        self._refresh_map()

    # -- slice map (embedded contact sheet of all slices) --------------------

    def _toggle_map(self) -> None:
        if self._map_win is not None:
            self._close_map()
        else:
            self._open_map()

    def _close_map(self) -> None:
        self._map_generation += 1  # cancel a pending background build
        if self._map_win is not None:
            try:
                self._map_win.destroy()
            except tk.TclError:
                pass
            self._map_win = None
        self._map_cells = []
        self._map_hl = None
        self._map_photo = None

    def _open_map(self, _event=None) -> None:
        """Open the slice map window and (re)build it in the background.

        The montage is computed off the UI thread without the busy
        machinery (locking the window for a large contact sheet would
        be annoying); newer builds supersede older ones via a
        generation counter, and closing the window cancels a pending
        build.
        """
        if self.data is None or self._busy:
            return
        self._close_map()  # rebuild from scratch if already open
        data, cmap = self.data, self.colormap
        self._map_generation += 1
        generation = self._map_generation
        q: queue.Queue = queue.Queue()

        def runner():
            try:
                q.put(("ok", build_slice_montage(data, cmap)))
            except Exception as exc:  # noqa: BLE001 - surface to the user
                q.put(("err", exc))

        threading.Thread(target=runner, daemon=True).start()

        def poll():
            if generation != self._map_generation:
                return  # superseded by a newer build or cancelled
            try:
                state, payload = q.get_nowait()
            except queue.Empty:
                self.root.after(50, poll)
                return
            if state == "err":
                messagebox.showerror("Slice map", str(payload))
                return
            if generation != self._map_generation:
                return  # closed while building
            montage, cells, rows, cols = payload
            self._map_cells = cells
            win = tk.Toplevel(self.root)
            win.title(f"Slice map — {len(cells)} slices")
            win.transient(self.root)
            frame = ttk.Frame(win, padding=4)
            frame.pack(fill=tk.BOTH, expand=True)
            canvas = tk.Canvas(frame, bg="#181818", highlightthickness=0)
            hbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL,
                                 command=canvas.xview)
            vbar = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                                 command=canvas.yview)
            canvas.configure(xscrollcommand=hbar.set,
                             yscrollcommand=vbar.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            vbar.grid(row=0, column=1, sticky="ns")
            hbar.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            mh, mw = montage.shape[:2]
            canvas.configure(scrollregion=(0, 0, mw, mh),
                             width=min(mw, 980), height=min(mh, 640))
            photo = tk.PhotoImage(data=to_pgm(montage))
            self._map_photo = photo  # prevent garbage collection
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.bind("<Button-1>", self._on_map_click)
            win.protocol("WM_DELETE_WINDOW", self._close_map)
            self._map_win = win
            self._map_canvas = canvas
            self._map_hl = None
            self._map_highlight()
            self.status.set(f"slice map: {len(cells)} slices — "
                            f"click a thumbnail to jump")

        poll()

    def _refresh_map(self) -> None:
        """Rebuild the map if it is open (data or color system changed)."""
        if self._map_win is None or self.data is None or self._busy:
            return
        try:
            alive = bool(self._map_win.winfo_exists())
        except tk.TclError:
            alive = False
        if alive:
            self._open_map()
        else:
            self._map_win = None

    def _map_highlight(self) -> None:
        """Move the highlight rectangle to the current slice."""
        if self._map_win is None or self._map_canvas is None \
                or not self._map_cells:
            return
        try:
            if not self._map_win.winfo_exists():
                return
        except tk.TclError:
            return
        idx = max(0, min(int(self.slice_var.get()),
                         len(self._map_cells) - 1))
        x0, y0, x1, y1 = self._map_cells[idx]
        if self._map_hl is None:
            self._map_hl = self._map_canvas.create_rectangle(
                x0, y0, x1, y1, outline="#ff4040", width=3)
        else:
            self._map_canvas.coords(self._map_hl, x0, y0, x1, y1)
        self._map_canvas.tag_raise(self._map_hl)

    def _on_map_click(self, event) -> None:
        """Jump to the slice whose thumbnail was clicked."""
        if self._map_canvas is None:
            return
        cx = self._map_canvas.canvasx(event.x)
        cy = self._map_canvas.canvasy(event.y)
        for i, (x0, y0, x1, y1) in enumerate(self._map_cells):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                if self.data is not None and not self._busy:
                    self.slice_var.set(i)
                    self._on_slice_move()
                return

    # -- undo / redo ----------------------------------------------------------

    @property
    def _dirty(self) -> bool:
        """True when the current data is not the last saved/loaded state.

        Identity-based: snapshots are immutable, so undoing back to the
        saved state clears the unsaved-changes marker automatically.
        """
        return (self.data is not None
                and self.data is not self._saved_array)

    def _push_history(self, previous: np.ndarray, label: str) -> None:
        """Snapshot ``previous`` just before ``self.data`` is replaced."""
        self._undo_stack.append((previous, label))
        del self._undo_stack[:-HISTORY_MAX]
        self._redo_stack.clear()

    def undo(self) -> None:
        """Step back to the state before the last modification (Ctrl+Z)."""
        if self.data is None or self._busy or not self._undo_stack:
            return
        previous, label = self._undo_stack.pop()
        self._redo_stack.append((self.data, label))
        self.data = previous
        self._after_history(f"undid {label}")

    def redo(self) -> None:
        """Re-apply the most recently undone modification (Ctrl+Y)."""
        if self.data is None or self._busy or not self._redo_stack:
            return
        nxt, label = self._redo_stack.pop()
        self._undo_stack.append((self.data, label))
        self.data = nxt
        self._after_history(f"re-applied {label}")

    def _after_history(self, msg: str) -> None:
        self._processed = (self._original_array is not None
                           and self.data is not self._original_array)
        self._reset_window()
        self.info.set(self._info_text())
        idx = max(0, min(int(self.slice_var.get()),
                         self.data.shape[0] - 1))
        self._show_plane(self.data[idx], keep_view=True)
        self._refresh_map()
        self._save_btn_states()
        self.status.set(msg)

    # -- saving -------------------------------------------------------------

    SAVE_FILETYPES = [("TIFF stack", "*.tif *.tiff"),
                      ("NumPy array", ".npy"),
                      ("NumPy compressed", ".npz")]

    def _save_btn_states(self) -> None:
        """Save is enabled when there is data; it targets the loaded file
        (or asks where to write when the data came from many files)."""
        has_data = self.data is not None
        for b in (self.rot_ccw_btn, self.rot_cw_btn, self.dim_btn,
                  self.crop_btn, self.report_btn,
                  self.roi_btn):
            b.configure(state=tk.NORMAL if has_data else tk.DISABLED)
        self.save_btn.configure(state=tk.NORMAL if has_data else tk.DISABLED)
        self.save_as_btn.configure(state=tk.NORMAL if has_data
                                   else tk.DISABLED)
        self.undo_btn.configure(state=tk.NORMAL
                                if has_data and self._undo_stack
                                else tk.DISABLED)
        self.redo_btn.configure(state=tk.NORMAL
                                if has_data and self._redo_stack
                                else tk.DISABLED)
        base = "Rockstone — rock sample processing"
        self.root.title(f"*{base}" if self._dirty and has_data else base)
        self._sync_menu_states()

    def _save_target(self) -> Path | None:
        """Where plain "Save" writes: the loaded file, if well-defined."""
        if self.path is not None and self.path.is_file() \
                and self.path.suffix.lower() in {".tif", ".tiff", ".npy",
                                                 ".npz"}:
            return self.path
        return None

    def save(self) -> None:
        """Save back to the loaded file (Ctrl+S); falls back to Save As."""
        if self.data is None or self._busy:
            return
        target = self._save_target()
        if target is None:
            self.save_as()
            return
        data = self.data

        def work():
            save_volume(data, target, meta=self._meta)

        def done(_=None):
            self._saved_array = self.data
            self._save_btn_states()
            self.status.set(f"saved {target.name}")

        self._run_task("Save", work, done)

    def save_as(self) -> None:
        """Save to a chosen file (Ctrl+Shift+S)."""
        if self.data is None or self._busy:
            return
        suggested = (self.path or (self.multi_paths[0].parent
                                   if self.multi_paths else None))
        initial = ""
        if suggested is not None:
            stem = suggested.stem
            if self._processed:
                stem += "_processed"
            initial = str(suggested.parent / f"{stem}.tif")
        p = filedialog.asksaveasfilename(
            title="Save volume as",
            defaultextension=".tif",
            initialfile=initial,
            filetypes=self.SAVE_FILETYPES)
        if not p:
            return
        target = Path(p)
        data = self.data

        def work():
            save_volume(data, target, meta=self._meta)

        def done(_=None):
            self.path = target
            self.multi_paths = None
            self._saved_array = self.data
            self._save_btn_states()
            self.info.set(self._info_text())
            self.status.set(f"saved {target.name}")

        self._run_task("Save As", work, done)

    def _on_close(self) -> None:
        if not self._confirm_discard():
            return
        self._close_map()
        self.root.destroy()


def launch(open_path: str | Path | None = None) -> int:
    """Create the root window and run the main loop.

    ``open_path`` optionally loads a file at startup (used by
    ``rockstone gui somefile.tif``).
    """
    root = tk.Tk()
    app = RockstoneApp(root)
    if open_path is not None:
        root.after(200, lambda: app.load_path(Path(open_path)))
    root.mainloop()
    return 0


if __name__ == "__main__":  # python -m rockstone.gui
    raise SystemExit(launch())
