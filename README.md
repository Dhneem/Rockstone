# Rockstone

Image processing for rock samples — **rigid slice alignment** and **beam-hardening
removal** for CT volumes and raw projections, as a Python library and a CLI.

> 📖 **User manual:** see [MANUAL.md](MANUAL.md) for a hands-on guide to the
> desktop app (loading data, every feature, shortcuts, CLI, troubleshooting).

## Features

- **Slice alignment** (`align_slices`) — rigid per-slice registration
  (translation + in-plane rotation) of reconstructed stacks:
  - sub-pixel translation via windowed **phase correlation**;
  - in-plane rotation via **Fourier–Mellin** log-polar correlation plus an
    image-domain refinement sweep for sub-degree accuracy;
  - neighbor transforms composed onto a common reference slice;
  - robust aggregation (median of overlapping neighbors) and optional
    smoothing of the transform trajectories across z;
  - works on grayscale and RGB, reports per-pair quality scores.
- **Projection alignment** (`align_projections`) — removes detector jitter
  from raw projection stacks before reconstruction.
- **Beam-hardening removal** (`remove_beam_hardening`):
  - `radial` — fits and flattens the radial cupping profile of reconstructed
    slices (multiplicative gain field, background-aware, per-slice or shared);
  - `chord` — linearizes raw projections against estimated chord length
    (physically-constrained concave model, data-anchored, edge-safe);
  - `poly` — applies a calibrated polynomial intensity remap
    (monotonicity-safe), e.g. from a calibration phantom;
  - `calibration` — computes that polynomial from a scan of a uniform phantom.
- **Compare** (`compare_volumes`) — difference statistics between two
  stacks (e.g. original vs processed): max/mean/RMSE, PSNR, correlation,
  worst-matching slices; shape mismatches are handled by comparing the
  overlapping region. Available in the library, the CLI
  (`rockstone compare a.tif b.tif`) and the GUI (**Compare…** with a
  side-by-side and difference view).
- **Metrics** — `cupping_index`, `radial_profile`, alignment residuals.
- **Formats** — DICOM (single files, multi-frame, or whole series
  folders), TIFF (single- or multi-page), NPY/NPZ, and folders of
  PNG/JPEG/TIFF slices (subfolders included, natural numeric order);
  2-D, 3-D and multichannel data; detector and sinogram layouts.
- **Desktop app** — click a file to open it: a Tkinter GUI with a
  **menu bar** (File / Edit / View / Process / Help) with keyboard
  accelerators, a grouped toolbar, a labeled side panel, a slice
  browser, **zoom & pan** (mouse wheel + drag, or the toolbar buttons),
  a **slice map** contact sheet window with every slice at once
  (toolbar button or Ctrl+M), **90° rotation**
  of the stack, interactive **cropping** (both undoable),
  **export** of the current slice as PNG/SVG/PDF/CSV,
  a one-click **report generator** (HTML, PDF, Word or text — with
  metrics, history and embedded images), **ROI measurement** (live per-slice
  statistics, physical area, CSV series, crop-to-ROI),
  **brightness/contrast** display sliders with
  a **live histogram** of the current slice, editable **physical
  dimensions** (mm per voxel), **Undo / Redo** (Ctrl+Z / Ctrl+Y),
  **Save / Save As** (Ctrl+S / Ctrl+Shift+S, with an unsaved-changes
  guard), switchable **color systems** (Gray, Inverted, Hot metal,
  Viridis, Jet, Rock, Asphalt, Ice, Copper, Thermal, Bone) and one-click
  align / deharden (`rockstone gui` or double-click `Rockstone.bat` on
  Windows).
- **Background correction** (`correct_background`) — estimates a smooth
  background surface per slice (robust block medians of the air region,
  upsampled) and removes it: **subtract** (air level flattens to zero,
  e.g. detector offset / reconstruction pedestal) or **divide**
  (multiplicative shading flattens to a constant). Complements the
  beam-hardening correction, which models cupping inside the sample.
  Available in the library, the GUI (**Background…**) and the CLI
  (`rockstone background`).
- **CLI** — `info`, `align`, `deharden`, `background`, `run`, `demo`,
  `gui`, `compare`.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate ; Linux/macOS: source .venv/bin/activate
pip install -e ".[test]"
```

Requires Python ≥ 3.10 with numpy, scipy, tifffile, imageio, pydicom.

## Quickstart (CLI)

```bash
# generate a synthetic rock volume with known misalignment + cupping,
# process it end-to-end and print a verification report
rockstone demo --slices 24 --size 128

# desktop application: click a file to open it, browse slices,
# run align / beam-hardening removal with one click
rockstone gui
rockstone gui rockstone_demo/rock_stack.tif   # open a file directly
# Windows: or just double-click Rockstone.bat
# tip: "Open file…" accepts a multi-selection (Ctrl/Shift-click) —
# several image files stack into one volume in natural order

# inspect a file
rockstone info rockstone_demo/rock_stack.tif

# full pipeline: align + deharden in one call
rockstone run rockstone_demo/rock_stack.tif --out processed.tif

# or step by step
rockstone align rockstone_demo/rock_stack.tif --out aligned.tif
rockstone deharden aligned.tif --out processed.tif

# compare two datasets (original vs processed, two scans, ...)
rockstone compare rockstone_demo/rock_stack.tif rockstone_demo/rock_processed.tif
```

Demo output on the synthetic dataset (ground truth known):

```
cupping    : 0.289 -> 0.060
pair quality: median 0.395, min 0.284
alignment vs ground truth: max deviation 0.87 px/deg, mean 0.49
residual inter-slice drift after alignment: mean 0.078 px, max 0.163 px
cupping index: 0.289 -> 0.060
```

Each subcommand has `--help` with its full option list (reference slice,
smoothing strength, dehardening method/degree, background handling, etc.).

## Quickstart (Python)

```python
import rockstone as rs

# --- reconstructed stack: align + deharden ------------------------------
vol = rs.load_volume("scan.tif")                # (nz, ny, nx)
aligned, ares = rs.align_slices(vol, estimate_theta=True, smooth=0.0)
corrected, dres = rs.remove_beam_hardening(aligned, method="radial", degree=3)
rs.save_volume(corrected, "scan_processed.tif")

print("cupping:", dres.cupping_before, "->", dres.cupping_after)

# --- raw projections: de-jitter + linearize against chord ---------------
proj = rs.load_projections("projections.tif")   # (n, rows, cols)
proj_al, pres = rs.align_projections(proj)
proj_lin, info = rs.correct_projections_chord(proj_al, layout="detector")
```

One-call batch processing:

```python
report = rs.process("scan.tif", output_path="out.tif",
                    align=True, deharden=True)
print(report.summary())
```

## Methods

### Slice alignment

1. **Translation** — consecutive slices are high-pass filtered (mean
   subtraction), Hann-windowed and cross-correlated in the Fourier domain;
   the peak location gives the integer shift, refined to sub-pixel accuracy
   by parabolic interpolation of the correlation surface.
2. **Rotation** — each pair is resampled to log-polar coordinates around the
   image center; rotation becomes a 1-D shift which the same phase-correlation
   machinery measures (Fourier–Mellin). A coarse estimate from the log-polar
   peak is refined by sweeping candidate angles in the image domain and
   keeping the one that maximizes correlation quality. The 180° ambiguity of
   log-polar matching is resolved by scoring both candidates.
3. **Composition** — per-neighbor transforms are chained onto a common
   reference slice (default: middle). For robustness against a bad pair,
   transforms can be estimated for several neighbor offsets and aggregated
   with a median.
4. **Smoothing** — optional penalized smoothing of the translation/rotation
   trajectories across z suppresses noisy per-slice excursions.

Quality per pair is the peak correlation of the phase-correlation surface
(0–1); low scores flag pairs the estimator does not trust.

### Beam hardening

A polychromatic beam hardens as it traverses the sample, so measured
attenuation drops with thickness. In reconstructions this appears as
**cupping** (dense material near the edge reads brighter than the same
material in the center); in raw projections, as a concave relation between
chord length and line integral.

- **Radial correction** (reconstructed volumes): a ring-averaged radial
  profile is extracted from each slice around the sample center, a smooth
  polynomial is fitted, and a multiplicative gain field is built from the
  fit (tending to 1 at the sample edge, in air the profile is ignored via a
  background percentile). The gain field can be estimated per slice or once
  for the whole volume.
- **Chord correction** (raw projections): the sample support per detector row
  is estimated from the nonzero profile, giving a chord length `u ∈ (0, 1]`
  for every measured value. Binned medians of (chord, value) define the
  measured hardening curve; a concave through-origin model
  `P(u) = a·u − b·u²` is fitted on reliable interior chords (edge-grazing
  rays are excluded as noise-dominated) and the correction gain is the ratio
  of the linear reference to `P(u)`, interpolated per pixel. The gain is
  clipped to a safe range so sparse-chord regions cannot blow up.
- **Polynomial remap**: for a `p(I)` measured on a calibration phantom
  (`method="calibration"` computes it from a scan of uniform material),
  intensities are remapped through the monotonicity-safe inverse.

### Background correction

Where beam hardening curves intensities *inside* the sample, the
**background correction** fixes the level and shading *around* it: a
non-zero air floor (detector offset, reconstruction pedestal) and slow
gradients across the field of view (scattering halos, uneven
illumination in scanned radiographs).

Per slice, the air region is found by thresholding against a low
percentile (with a margin dilated around the sample so its rim cannot
bias the fit), sampled on a coarse block grid by robust medians, and
linearly upsampled into a smooth background surface. Two modes:

- **offset** — subtract the surface: the air level flattens to zero,
  the usual choice for reconstructed CT stacks;
- **divide** — divide by the surface: multiplicative shading flattens
  to a constant (optical/radiograph-style data).

The result reports the air level and background spread before/after so
the effect is quantified. The surface is estimated per slice by default
or shared across the stack. GUI: **Background…**. CLI: `rockstone
background in.tif -o out.tif [--mode offset|divide] [--shared]
[--percentile P] [--background dark|bright]`; the batch `run` command
adds `--with-background` plus `--bg-mode/--bg-shared/--bg-percentile/
--bg-background` (the stage is opt-in so existing workflows are
unchanged). Note: with an extreme background gradient the global
air-vs-sample threshold can misclassify blocks inside the sample; the
fit then degrades gracefully toward the air level rather than corrupting
the data.

## Data conventions

- Volumes are `(nz, ny, nx)`; slices are aligned in-plane about the image
  center. Positive rotation angles rotate content clockwise as displayed
  (y-axis down).
- **Folders of slices**: `load_volume("folder/")` gathers every image
  inside — including subfolders — and stacks them as slices in natural
  (numeric-aware) order, so `slice_2.png` precedes `slice_10.png`.
  Unreadable files are skipped with a warning; pass `recursive=False`
  for top-level-only. A folder containing DICOM files is treated as a
  DICOM series instead.
- **Many files at once**: `load_stack_from_files(paths)` stacks a
  multi-selection of image files into one volume (natural order,
  multi-page TIFFs expand to their pages, grayscale+color mixes are
  reconciled). In the GUI, just Ctrl/Shift-click several files in the
  "Open file…" dialog.
- **DICOM**: single `.dcm` files (multi-frame supported) or a folder
  containing a series are loaded with `load_volume`. Series slices are
  ordered by `ImagePositionPatient` z (falling back to `InstanceNumber`
  then filename), and `RescaleSlope`/`RescaleIntercept` are applied, so
  CT data comes back in Hounsfield units. Color frames are reduced to
  grayscale.
- Projections are `(n_views, n_rows, n_cols)` in *detector* layout; pass
  `layout="sinogram"` for `(n_rows, n_cols, n_views)` data. Results are
  returned in the same layout as the input.
- `_estimate_chords` returns chords normalized to the local sample diameter
  (0 = edge, 1 = through the center).

### Compare

`compare_volumes(a, b)` returns a `ComparisonResult` with difference
statistics — max/mean absolute difference, RMSE, PSNR, Pearson
correlation, an `identical` flag and the indices of the worst-matching
slices (`as_report()` renders a text report). Shape mismatches are
allowed: the overlapping region (center crop) is compared and reported.

In the GUI, **Compare…** asks for a second dataset, shows the metrics,
and switches the viewer to a side-by-side view (left: current data,
right: the other dataset) that follows the slice slider; press the
**Difference** toggle in the side panel to see `a − b` per slice instead,
and **Escape** (or the toggle again) to return to the plain view.

## Metrics

- `cupping_index(image, center=None)` — severity of radial cupping:
  normalized difference between the edge and center of the ring-averaged
  radial profile (0 = flat).
- `radial_profile(image, center=None, n_bins=32)` — the profile itself, for
  plotting and inspection.
- `alignment_residual` / per-pair `scores` — residual inter-slice motion and
  correlation quality after alignment.

## Development

```bash
pytest          # 250 tests: transforms, alignment, beam hardening, background, I/O, DICOM, folders, GUI, compare, view tools, crop, export, menus, report, roi, crop shapes
rockstone demo  # end-to-end self-check with ground truth
.venv/Scripts/python tools/build_manual_pdf.py   # rebuild MANUAL.pdf from MANUAL.md (needs reportlab)
```

Project layout:

```
rockstone/
  transforms.py      rigid 2-D transform model + warp
  align.py           phase correlation, Fourier-Mellin, stack/projection alignment
  beam_hardening.py  radial, chord, poly, calibration corrections
  background.py      air-level / shading flattening (offset, divide)
  export.py          slice export to PNG / SVG / PDF / CSV
  report.py          HTML/text dataset reports (embedded images)
  roi.py             region-of-interest statistics (slice and series)
  metrics.py         cupping index, radial profile
  recon.py           minimal parallel-beam FBP (used by tests/demo)
  io.py              TIFF/NPY/image-folder readers and writers
  pipeline.py        batch processing + reports
  demo.py            synthetic rock data generator + demo
  cli.py             command-line interface
  compare.py         dataset comparison (difference metrics)
  gui.py             Tkinter desktop app (click a file to open it)
tests/               pytest suite
```

### Menus, toolbar and layout

The window has a classic **menu bar**:

- **File** — Open file… (Ctrl+O), Open folder… (Ctrl+Shift+O), Save
  (Ctrl+S), Save As… (Ctrl+Shift+S), Exit;
- **Edit** — Undo (Ctrl+Z), Redo (Ctrl+Y), Rotate 90° clockwise /
  counterclockwise, Crop…, Physical dimensions…;
- **View** — Zoom in/out (Ctrl++ / Ctrl+−), Fit to window (Ctrl+F),
  Slice map window (Ctrl+M), Compare…, Show difference, Reset display;
- **Process** — Align slices, Remove beam hardening, Background
  correction…, Export current slice… (Ctrl+E);
- **Help** — About Rockstone….

Menu entries enable/disable together with their toolbar buttons. The
**toolbar** groups related commands behind separators (open, processing,
compare, zoom, view/transform, save). The **side panel** is organized
into labeled frames: *Appearance* (color system, brightness/contrast,
histogram), *Export & compare*, and *Slice* (navigation).

### Zoom and pan

Zoom with the **mouse wheel** over the image or the toolbar **− / +**
buttons (10×–20×), and pan by **click-dragging**. **Fit** scales the
slice to the window. The zoom level and pan offset persist while you
scroll slices, switch color systems, toggle the difference view or run
align/deharden; loading a new file resets to 100%. Rendering is cached
per (slice, color system), so wheel-zooming only rescales the already
normalized image.

### Undo and redo

Every Align or Remove-beam-hardening run pushes a snapshot onto a
history stack (up to 10 steps). **Undo** (Ctrl+Z) steps back through
them, **Redo** (Ctrl+Y or Ctrl+Shift+Z) forward; a new modification
drops the redo fork, as in standard editors. The unsaved-changes
marker is identity-based, so undoing back to the last saved state
clears it automatically — and after saving, one undo takes you to a
state that is again "clean". History resets when a new file is loaded.

### Saving results

**Save** (Ctrl+S) writes the current data back to the file it came from
(TIFF, NPY or NPZ); if the data came from a folder or a multi-file
selection it falls back to **Save As…** (Ctrl+Shift+S), which asks for a
destination — the suggestion is `*_processed.tif` after a correction has
run. After a Save As, plain Save targets the new file. While there are
unsaved changes (e.g. after Align or Remove beam hardening) the title
bar shows an asterisk, and closing the app or loading another file asks
whether to save first. `save_volume` writes TIFF / NPY / compressed NPZ.

### ROI — region of interest

**ROI…** (toolbar or **Edit ▸ Measure ROI…**) measures a rectangular
region on the *raw data*: drag the rectangle on the image (cyan
outline, distinct from the red crop selection) and the dialog shows
**mean, std, min/max, median, pixel count and the physical area in
mm²** — recomputed live as you scroll through slices, with an
air-referenced mean as an attenuation proxy.

- **Crop to ROI…** turns the measurement into an undoable crop;
- **Save all slices (CSV)…** writes the same ROI measured on *every*
  slice as a CSV series (one row per slice, including
  mean-above-air) for porosity/density trending.

Headless: `rockstone roi scan.tif --rect 100 100 199 199 --slice 5`
(single slice, human-readable report), `--all-slices` (one line per
slice) or `--csv` (`<input>_roi.csv`). The mean-above-air statistic
uses the standard CT air value of **-1000 HU** (`rockstone.AIR_HU`).

### Slice map

**Slice map** in the toolbar (or **View ▸ Slice map**, Ctrl+M) opens a
separate scrollable contact-sheet window showing **every slice at
once** as a thumbnail (grid layout, rendered with the current color
system). Click any thumbnail to jump to that slice; a red rectangle
marks the current slice and follows as you scroll. The map rebuilds
automatically when the data or color system changes (e.g. after
align/deharden) and closes with the main window. Toggle it off with the
same button or Escape of the window. The montage builds in the
background without locking the UI; reopening supersedes any
still-running build.

### Generating reports

**Report…** (side panel, **Process ▸ Generate report…**, Ctrl+R)
writes a self-contained summary document of the current dataset:

- shape, data type, kind, value range;
- physical dimensions, voxel volume and the physical extent of the
  stack in millimeters;
- metrics: cupping index (middle slice), air level, background spread —
  with a hint when cupping suggests a beam-hardening pass;
- the **modification history of the session** (align, crop, rotate, …)
  and compare results if a comparison was run;
- the **HTML variant embeds a PNG snapshot of the current slice and its
  intensity histogram** — no external files, ready to e-mail or archive
  next to the data.

Choose **HTML** (self-contained with images), **PDF**, **Word
(.docx)** or plain text; the CLI writes the same reports headlessly:
`rockstone report scan.tif -o report.html` (`--pdf`, `--docx`,
`--txt`, `--no-image`). A note marks datasets with unsaved changes.

### Color systems

The GUI's **Colors** dropdown re-renders the current slice through a
256-entry lookup table — useful for emphasizing pore structure (Inverted),
density gradients (Hot metal, Viridis, Jet), a natural rock look (Rock),
the cool neutral grays of pavement (Asphalt), or thermal/ice/copper/bone
renderings.
The mappings are defined in `rockstone.gui.COLORMAPS` and can be extended
in a few lines; grayscale data of any bit depth is auto-stretched to the
full dynamic range before mapping.

### Rotate, brightness and histogram

The toolbar **⟲ 90° / ⟳ 90°** buttons rotate the whole stack in-plane:
every slice is rotated, the slice count and order stay the same, and
the operation is undoable like every other modification (Ctrl+Z /
Ctrl+Y).

The **Brightness** and **Contrast** sliders in the side panel adjust
only the *display* window — pixel values and saved files are untouched.
Brightness slides the window, contrast narrows it (positive) or widens
it (negative); **Reset display** returns to the automatic 1–99.5
percentile stretch. The small **histogram panel** below the sliders
shows the intensity distribution of the slice as currently rendered —
it follows slicing, color systems and the display window.

### Crop

The crop tool offers two shapes:

- **Rectangle** — the dragged selection is cropped away in every slice
  (the stack shrinks to the selection);
- **Ellipse** — the ellipse *inscribed* in the dragged selection is
  kept on every slice, everything outside is filled with **NaN**
  (transparent — excluded from statistics). The image size is
  preserved, so slices stay aligned and comparable; ideal for cutting
  surrounding material away from a cylindrical sample.

The shape is drawn live while dragging (rectangle outline vs ellipse
outline) and the whole operation is undoable.

**Crop…** in the toolbar enters crop mode: **drag a rectangle** directly
on the image (a red outline marks the selection at any zoom level, and
the dialog shows its size and position in pixels), then press **Apply**
to keep that region in *every* slice — the stack shape becomes
`(n, height, width)` of the selection. **Escape**, **Cancel** or closing
the dialog aborts; during background tasks (align/deharden/compare/save)
crop mode is closed automatically. Cropping is undoable (Ctrl+Z /
Ctrl+Y) and never changes the slice count, the data kind (volume vs
projections) or the embedded physical dimensions.

### Exporting images and data

**Export…** in the side panel saves the **current slice** in four
formats:

- **PNG** — the rendered view exactly as displayed (color system,
  brightness/contrast window applied), tagged with the physical pixel
  size as DPI metadata so 1:1 prints come out at true scale;
- **SVG** — a *true vector* rendering: every pixel becomes a rectangle,
  runs of equal value in a row are merged into single `<rect>`/`<path>`
  shapes (one merged path per gray level for grayscale images), and the
  canvas is sized in **millimeters** from the stored physical
  dimensions. Faithful at any magnification; the raw value range of the
  slice is recorded in the file;
- **PDF** — the rendered view as a single-page document, sized by the
  physical DPI like the PNG;
- **CSV** — the *raw* (float) values of the slice as `row, col, value`
  rows for spreadsheets or scripted analysis; NaNs become empty cells.

PNG/PDF/SVG need Pillow (declared as a dependency). Exports run in the
background and suggest a `<file>_slice<n>.<ext>` name.

### Physical dimensions

**Dimensions…** in the toolbar sets the voxel size (pixel width, pixel
height and slice spacing, in millimeters). The values appear in the
info panel (`voxel: 0.05 x 0.05 x 0.05 mm`) and are embedded when
saving TIFF (image description) or NPZ (`meta_json` key);
`rockstone.read_metadata` reads them back, so processed files keep
their physical calibration. DICOM input supplies the defaults
automatically (`PixelSpacing`, `SliceThickness`); other formats start
empty until you fill them in.

## Limitations

- Alignment is *rigid* (translation + in-plane rotation): it cannot correct
  tilt, scaling or non-rigid motion between slices.
- Rotation estimates assume the rotation center is near the image center;
  very off-center pivots reduce accuracy.
- The chord correction models the dominant first-order hardening;
  severe nonlinearity (e.g. strong scattering, metal) needs calibration data.
- Corrections are multiplicative intensity remaps — absolute attenuation
  calibration should be re-derived afterwards if quantitative values matter.
