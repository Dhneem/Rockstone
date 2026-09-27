# Rockstone — User Manual

A hands-on manual for the Rockstone desktop application. Rockstone
processes rock-sample images — especially micro-CT stacks — with
**slice alignment**, **beam-hardening removal**, **background
correction**, view/measurement tools and multi-format export.

For build/development notes see `README.md`; this document is written
for *users* of the app.

---

## Contents

1. [Getting started](#1-getting-started)
2. [Loading data](#2-loading-data)
3. [The main window](#3-the-main-window)
4. [Viewing: zoom, pan, slices, colors](#4-viewing-zoom-pan-slices-colors)
5. [Brightness, contrast and the histogram](#5-brightness-contrast-and-the-histogram)
6. [Processing: align, deharden, background](#6-processing-align-deharden-background)
7. [Rotate, crop, ROI and physical dimensions](#7-rotate-crop-roi-and-physical-dimensions)
8. [Compare and difference view](#8-compare-and-difference-view)
9. [Undo, redo and saving](#9-undo-redo-and-saving)
10. [Exporting images and data](#10-exporting-images-and-data)
11. [The slice map](#11-the-slice-map)
12. [Keyboard shortcuts](#12-keyboard-shortcuts)
13. [The command line](#13-the-command-line)
14. [Supported file formats](#14-supported-file-formats)
15. [Tips and troubleshooting](#15-tips-and-troubleshooting)

---

## 1. Getting started

### Launch the app

- Double-click **`Rockstone.bat`** (Windows), or run:

  ```
  rockstone gui
  ```

  or

  ```
  python -m rockstone.gui
  ```

- You can pass a file to open at startup: `rockstone gui myfile.tif`.

### Install (first time)

```
pip install -e .
```

Requires Python 3.10+ with numpy, scipy, tifffile, imageio, pydicom and
Pillow (installed automatically).

---

## 2. Loading data

**Open file…** (Ctrl+O) opens one data file. In the file dialog you can
Ctrl/Shift-click to select **several image files at once** — they are
stacked into one volume in natural (numeric) order.

**Open folder…** (Ctrl+Shift+O) loads every slice image inside a folder
(subfolders included) or a **DICOM series** (all slices of one series
in a folder, sorted by position).

What happens on load:

- the **info panel** (top of the side panel) shows file name, data
  kind, shape, value range, cupping severity and the physical voxel
  size if known;
- the first slice is displayed; browse with the **Slice** slider;
- the title bar shows `Rockstone — rock sample processing` and gains a
  `*` whenever there are unsaved changes.

**Data kinds.** Rockstone distinguishes *volumes* (reconstructed
stacks, the usual case) from *projections* (raw detector frames). The
kind is auto-detected and shown in the info panel; it decides which
corrections are offered (e.g. "Linearize projections (chord)" instead
of "Remove beam hardening" for raw data).

---

## 3. The main window

```
+--------------------------------------------------------------+
| Menu bar: File  Edit  View  Process  Help                     |
+--------------------------------------------------------------+
| Toolbar: open | process | compare | zoom | view | save        |
+---------------------------------------------+----------------+
|                                             | info panel     |
|            slice image                      | [Appearance]   |
|        (wheel = zoom, drag = pan)           |  colors        |
|                                             |  brightness    |
|                                             |  contrast      |
|                                             |  histogram     |
|                                             | [Export &      |
|                                             |   compare]     |
|                                             | [Slice]        |
|                                             |  slider        |
+---------------------------------------------+----------------+
| status bar                                                    |
+--------------------------------------------------------------+
```

- **Menu bar** — every command with its shortcut (see §12).
- **Toolbar** — the most-used commands, grouped by separators.
- **Side panel** — *Appearance* (color system, brightness/contrast,
  histogram), *Export & compare*, *Slice* navigation, and the info
  read-out on top.
- **Status bar** — progress and result messages ("alignment
  finished.", "cropped to 128 x 128 px — 24 slices kept", …).

---

## 4. Viewing: zoom, pan, slices, colors

### Zoom and pan

- **Mouse wheel** over the image zooms in/out (10%–2000%).
- **Click-drag** moves (pans) the image.
- Toolbar **− / +** buttons, the **View** menu, or **Ctrl+− / Ctrl++**.
- **Fit** (Ctrl+F) scales the slice to the window.
- Zoom and pan survive slice scrolling and color changes; loading a
  new file resets to 100%.

### Slices

- Drag the vertical **Slice** slider, or pick a thumbnail in the
  **slice map** (§11). The label under the slider shows
  `current / total`.

### Color systems

The **Colors** dropdown re-renders the slice through a lookup table:

| Name      | Good for                          |
|-----------|-----------------------------------|
| Gray      | default, neutral                  |
| Inverted  | pores and dark detail             |
| Hot metal / Viridis / Jet | density gradients |
| Rock      | natural rock look                 |
| Asphalt   | cool neutral grays (pavement)     |
| Ice / Copper / Thermal / Bone | stylistic renders |

Changing colors is display-only and instantly re-renders the slice map
too.

---

## 5. Brightness, contrast and the histogram

The **Brightness** and **Contrast** sliders (−100…100) adjust only the
*display window* — pixel values, the histogram of the data and saved
files are untouched.

- **Brightness** slides the window up/down.
- **Contrast** narrows the window (positive) or widens it (negative).
- **Reset display** returns to the automatic 1–99.5 percentile stretch
  (also in the View menu).

The **histogram panel** shows the intensity distribution of the
currently displayed image. It follows slicing, color systems and the
display window — handy for judging whether a window choice saturates
the sample.

---

## 6. Processing: align, deharden, background

All processing runs in the background (the window stays responsive),
is **undoable** (§9) and marks the data as changed.

### Align slices (Process ▸ Align slices)

Removes slice-to-slice jitter (translation + in-plane rotation) using
phase correlation with Fourier–Mellin rotation estimation. Works for
volumes and raw projections. The status bar reports progress; the info
panel's cupping value is re-computed afterwards.

### Remove beam hardening (Process ▸ Remove beam hardening)

Fixes **cupping** — dense material near the edge reading brighter than
in the center — with a radial multiplicative gain field (volumes), or
linearizes raw projections against chord length. Compare the cupping
value in the info panel before/after to judge the effect.

### Background correction… (Process ▸ Background correction…)

Flattens the level and shading *around* the sample (detector offset,
reconstruction pedestal, scattering halos):

- **Mode** — *Subtract* (air becomes 0; usual for CT stacks) or
  *Divide by shading* (multiplicative shading flattens to a constant).
- **Estimate per slice** — on by default; untick for one shared
  surface.
- **Background** — *dark* (air around bright rock) or *bright*.
- **Air percentile** — the percentile used as the air level (default
  1.0).

Afterwards the status bar reports e.g.
`air 0.13 -> 0.0008, bg spread 0.053 -> 0.005`.

**Recommended order:** Align ▸ Remove beam hardening ▸ Background
correction.

---

## 7. Rotate, crop, ROI and physical dimensions

### Measure ROI… (Edit menu or ROI… button)

Measures a rectangular **region of interest** on the raw data:

1. Click **ROI…** and **drag a rectangle** on the image (cyan outline;
   the red outline is the crop tool).
2. The dialog shows **mean, std, min/max, median, pixel count** and
   the **area in mm²** (when physical dimensions are set), plus a
   mean-above-air statistic referenced to the standard CT air value
   of **-1000 HU**.
3. Scroll slices — the statistics **update live** for the same region.

- **Crop to ROI…** converts the measurement into an undoable crop.
- **Save all slices (CSV)…** writes one row per slice with the same
  statistics — perfect for tracking a feature through the depth of
  the sample.



### Rotate 90° (Edit menu, ⟲/⟳ toolbar buttons)

Rotates the **whole stack** in-plane; slice count and order are
unchanged. Undoable like every other modification.

### Crop… (Edit menu or toolbar)

1. Click **Crop…** — a small dialog appears and the cursor becomes a
   crosshair.
2. Choose the **Shape**:
   - **Rectangle** — crops the stack down to the dragged selection;
   - **Ellipse** — keeps the ellipse inscribed in the dragged
     selection on every slice and fills the outside with **NaN**
     (transparent — excluded from statistics and exports). The image
     size is unchanged, so slices stay registered — perfect for
     cutting the surrounding material away from a cylindrical sample.
3. **Drag the region** on the image. The outline follows the chosen
   shape at any zoom; the dialog shows its size and position in
   pixels.
4. Press **Apply** to keep that region in *every* slice, or
   **Cancel** / **Escape** to abort.

Cropping never changes the slice count, the data kind or the stored
physical dimensions. Undoable.

### Physical dimensions… (Edit menu or Dimensions… button)

Sets the voxel size in millimeters — pixel width, pixel height and
slice spacing. The values:

- appear in the info panel (`voxel: 0.05 x 0.05 x 0.05 mm`);
- are **embedded when saving** TIFF/NPZ, so processed files keep their
  calibration (`rockstone.read_metadata` reads them back);
- are prefilled from DICOM input (`PixelSpacing`, `SliceThickness`) or
  from metadata embedded in the loaded file.

Clearing all fields removes the dimensions again.

---

## 8. Compare and difference view

**Compare…** (View menu or toolbar) asks for a second dataset (e.g.
the original file while the processed one is open) and shows:

- **side-by-side** view of the same slice of both stacks, then
- **Show difference** toggles an intensity-difference image.

The status bar reports difference statistics (max/mean/RMSE, PSNR,
correlation). Press **Escape** or click **Show difference** again to
return to the normal view.

---

## 9. Undo, redo and saving

- **Undo** (Ctrl+Z) / **Redo** (Ctrl+Y or Ctrl+Shift+Z) step through
  the last **10 modifications** (align, deharden, background, rotate,
  crop). A new modification drops the redo fork, as in standard
  editors.
- The unsaved-changes `*` in the title bar is identity-based: undoing
  back to the last saved state clears it automatically.

### Saving

- **Save** (Ctrl+S) writes back to the file the data came from
  (TIFF/NPY/NPZ).
- **Save As…** (Ctrl+Shift+S) asks for a destination; after a
  correction the suggestion is `<name>_processed.tif`.
- Data that came from a folder or a multi-file selection always falls
  back to Save As.
- Closing the app or loading another file with unsaved changes asks
  whether to **save first** (the save there is synchronous, so it is
  safe to re-read the same file).
- Embedded physical dimensions (if set) are written into the file.

---

## 10. Exporting images and data

### Report… (side panel, Process ▸ Generate report…, Ctrl+R)

Writes a **summary report** of the current dataset as **HTML** (with
embedded snapshot and histogram — no external files), **PDF**, **Word
(.docx)** (both with the same images) or **plain text**. Contents:
shape, kind, value range, physical dimensions and sample extent in mm,
cupping / air level / background-spread metrics with hints, the
modification history of the session, compare results, and a note when
there are unsaved changes. The suggested name is `<file>_report.html`
(extensions follow the chosen format). Headlessly:
`rockstone report scan.tif -o report.html` (`--pdf`, `--docx`,
`--txt`, `--no-image`).

### Export… (side panel, Process ▸ Export current slice…, Ctrl+E)

Saves the **current slice**:

| Format | What you get |
|--------|--------------|
| **PNG** | the rendered view as displayed (colors + window), tagged with the physical pixel size as DPI so 1:1 prints are true scale |
| **SVG** | *true vector*: one shape per run of equal pixels, canvas sized in millimeters from the voxel metadata, raw value range recorded |
| **PDF** | the rendered view as a one-page document, sized by the physical DPI |
| **CSV** | the *raw* (float) values of the slice as `row, col, value` rows; NaN → empty cell |

The save dialog suggests `<file>_slice<n>.<ext>` with the current
slice number. Export runs in the background; the status bar confirms.

Note: PNG/PDF/SVG contain the *rendered* image (what you see, including
brightness/contrast); CSV contains the *data*. For publications prefer
SVG (infinite zoom, exact physical scale) plus CSV for the numbers.

---

## 11. The slice map

**Slice map** (toolbar button, or View ▸ Slice map, Ctrl+M) opens a
separate scrollable window showing **every slice at once**. Click a
thumbnail to jump to it; a red rectangle marks the current slice and
follows as you scroll. The map rebuilds automatically after any data or
color change and closes with the main window. Toggle it closed with the
same button or by closing the map window.

---

## 12. Keyboard shortcuts

| Keys | Action |
|------|--------|
| Ctrl+O | Open file… |
| Ctrl+Shift+O | Open folder… |
| Ctrl+S | Save |
| Ctrl+Shift+S | Save As… |
| Ctrl+E | Export current slice… |
| Ctrl+Z | Undo |
| Ctrl+Y / Ctrl+Shift+Z | Redo |
| Ctrl+M | Slice map on/off |
| Ctrl+R | Generate report… |
| Ctrl++ / Ctrl+− | Zoom in / out |
| Ctrl+F | Fit to window |
| Escape | Cancel crop / leave compare view |
| mouse wheel | Zoom |
| click-drag | Pan (or draw the crop/ROI rectangle in crop/ROI mode) |

---

## 13. The command line

All processing is also available headlessly:

```
rockstone info data.tif              # shape, dtype, kind, range, cupping
rockstone align raw.tif -o a.tif     # rigid slice alignment
rockstone deharden vol.tif -o d.tif  # cupping removal
  --method radial|chord|poly  --degree 3  --background dark|bright
rockstone background vol.tif -o b.tif  # air-level / shading flattening
  --mode offset|divide  --shared  --percentile 1.0  --background dark
rockstone run scan.tif -o out.tif    # full pipeline (align + deharden)
  --with-background                  # + background correction
  --no-align --no-deharden           # stage toggles
rockstone compare a.tif b.tif        # difference report
rockstone report scan.tif -o r.html  # dataset report (HTML; --txt, --no-image)
rockstone roi scan.tif --rect 100 100 199 199 --slice 5
                                     # ROI statistics (--all-slices, --csv)
rockstone demo                       # generate a synthetic demo dataset
```

Every command accepts `-o` (output file) and `-p` (per-slice progress
lines). Default output is `<input>_processed.tif`.

---

## 14. Supported file formats

| Source | Details |
|--------|---------|
| TIFF | single- or multi-page stacks (no JPEG-compressed TIFF) |
| DICOM | single files, multi-frame files, or series folders (z-sorted, HU rescale applied); `PixelSpacing`/`SliceThickness` become the physical dimensions |
| NPY / NPZ | raw NumPy arrays (NPZ as saved by Rockstone) |
| Image folders | PNG/JPEG/TIFF slices, subfolders included, natural numeric order |
| Output | TIFF (with metadata), NPY, NPZ; slice export: PNG, SVG, PDF, CSV |

2-D, 3-D and multichannel (RGB) data are handled; the stack convention
is `(slices, rows, columns)`.

---

## 15. Tips and troubleshooting

**The correction made things worse?** Undo (Ctrl+Z) and check the
metrics: cupping (info panel) for beam hardening, air level/background
spread (status bar) for background correction. The numbers are printed
before/after on purpose.

**The sample fill the whole frame?** Crop first (§7) — the background
correction then has clean air to fit, and exports are tighter.

**SVG export is large.** The SVG contains one shape per pixel run; on
noisy 1024² slices this can be big. Export a **PNG** for previews and
SVG for final figures (or after smoothing).

**DICOM series loads with wrong slice order?** The series is sorted by
the DICOM z-position; if the headers are broken, sort the files into a
folder with numbered names and open the folder instead.

**"unsupported output format"** — Save As only offers
`.tif/.tiff/.npy/.npz`; use **Export…** for PNG/SVG/PDF/CSV.

**JPEG-compressed TIFF won't open** — the JPEG codec (imagecodecs) is
not installed; convert the file, or `pip install imagecodecs` and
reload.

**Nothing happens when clicking a button** — the status bar explains:
during background tasks most controls are disabled until the task
finishes.

**Data looks all black / all white** — the display is percentile
stretched; press **Reset display**, or use the brightness/contrast
sliders. The raw values are never modified by display controls.

---

*Rockstone is under active development; see `README.md` for the
feature list, methods and the test suite.*
