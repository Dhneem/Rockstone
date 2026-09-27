"""Command-line interface for rockstone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .compare import compare_volumes


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", help="output file (TIFF/NPY)")
    parser.add_argument("-p", "--progress", action="store_true",
                        help="print per-slice progress")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rockstone",
        description="Image processing for rock samples: slice alignment "
                    "and beam-hardening removal.",
    )
    p.add_argument("--version", action="version", version=f"rockstone {__import__('rockstone').__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    info = sub.add_parser("info", help="describe a data file/folder")
    info.add_argument("input")

    gui = sub.add_parser("gui", help="open the desktop application")
    gui.add_argument("file", nargs="?", default=None,
                     help="data file to open at startup (optional)")

    al = sub.add_parser("align", help="rigidly align the slices of a stack")
    al.add_argument("input")
    _add_common(al)
    al.add_argument("--no-rotation", action="store_true",
                    help="skip rotation estimation (translation-only)")
    al.add_argument("--smoothing", choices=["none", "poly", "median"],
                    default="none", help="motion-trajectory regularization")
    al.add_argument("--smoothing-degree", type=int, default=3)
    al.add_argument("--reference", default="first",
                    help="'first', 'middle' or a slice index")
    al.add_argument("--order", type=int, default=1, choices=[0, 1],
                    help="interpolation order (0=nearest, 1=bilinear)")

    bh = sub.add_parser("deharden", help="remove beam-hardening artifacts")
    bh.add_argument("input")
    _add_common(bh)
    bh.add_argument("--method", choices=["radial", "chord", "poly"],
                    default="radial",
                    help="radial: reconstructed slices; chord: raw projections; "
                         "poly: calibrated intensity remap")
    bh.add_argument("--degree", type=int, default=3, help="polynomial degree")
    bh.add_argument("--background", choices=["dark", "bright"], default="dark",
                    help="background phase around the sample (radial mode)")
    bh.add_argument("--layout", choices=["detector", "sinogram"],
                    default="detector", help="projection layout (chord mode)")
    bh.add_argument("--center", nargs=2, type=float, metavar=("CX", "CY"),
                    help="explicit sample center (radial mode)")
    bh.add_argument("--coeffs", nargs="+", type=float,
                    help="polynomial coefficients for --method poly")

    bg = sub.add_parser("background",
                        help="flatten the background level/shading "
                             "around the sample")
    bg.add_argument("input")
    _add_common(bg)
    bg.add_argument("--mode", choices=["offset", "divide"],
                    default="offset",
                    help="offset: subtract (air -> 0); divide: remove "
                         "multiplicative shading")
    bg.add_argument("--shared", action="store_true",
                    help="one shared surface instead of per-slice")
    bg.add_argument("--percentile", type=float, default=1.0,
                    help="air percentile (default 1.0)")
    bg.add_argument("--background", choices=["dark", "bright"],
                    default="dark", help="background phase")

    run = sub.add_parser("run", help="full pipeline: align + deharden")
    run.add_argument("input")
    _add_common(run)
    run.add_argument("--no-align", action="store_true")
    run.add_argument("--no-deharden", action="store_true")
    run.add_argument("--with-background", action="store_true",
                     help="also flatten the background (air level/shading)")
    run.add_argument("--kind", choices=["volume", "projections"],
                     help="override data-kind auto-detection")
    _add_align_opts(run)
    _add_deharden_opts(run)
    _add_background_opts(run)

    demo = sub.add_parser("demo", help="generate a synthetic dataset and "
                                       "process it end-to-end")
    demo.add_argument("--out", default="rockstone_demo", help="output folder")
    demo.add_argument("--slices", type=int, default=24)
    demo.add_argument("--size", type=int, default=160)

    comp = sub.add_parser("compare", help="compare two stacks and report "
                                          "difference statistics")
    comp.add_argument("a", help="first dataset (reference)")
    comp.add_argument("b", help="second dataset")

    rep = sub.add_parser("report", help="write an HTML/text report for "
                                        "a dataset")
    rep.add_argument("input")
    rep.add_argument("-o", "--output",
                     help="report file (.html or .txt); default "
                          "<input>_report.html")
    rep.add_argument("--no-image", action="store_true",
                     help="skip the embedded slice snapshot")
    rep.add_argument("--txt", action="store_true",
                     help="write plain text instead of HTML")
    rep.add_argument("--pdf", action="store_true",
                     help="write a PDF report instead of HTML")
    rep.add_argument("--docx", action="store_true",
                     help="write a Word report instead of HTML")

    roi = sub.add_parser("roi", help="measure a rectangular region of "
                                     "interest on one or all slices")
    roi.add_argument("input")
    roi.add_argument("--rect", nargs=4, type=int,
                     metavar=("X0", "Y0", "X1", "Y1"), required=True,
                     help="ROI rectangle (inclusive pixel corners)")
    roi.add_argument("--slice", type=int, default=0,
                     help="slice index (ignored with --all-slices)")
    roi.add_argument("--all-slices", action="store_true",
                     help="measure the ROI on every slice")
    roi.add_argument("--csv", action="store_true",
                     help="write per-slice measurements as CSV "
                          "(<input>_roi.csv)")
    return p


def _add_align_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--align-smoothing", choices=["none", "poly", "median"],
                        default="none")
    parser.add_argument("--align-reference", default="first")
    parser.add_argument("--no-rotation", action="store_true")


def _add_deharden_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--deharden-method", choices=["radial", "chord", "poly"],
                        default="radial")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--background", choices=["dark", "bright"], default="dark")
    parser.add_argument("--layout", choices=["detector", "sinogram"],
                        default="detector")


def _add_background_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bg-mode", choices=["offset", "divide"],
                        default="offset", help="subtract or divide the "
                        "background surface")
    parser.add_argument("--bg-shared", action="store_true",
                        help="one shared surface instead of per-slice")
    parser.add_argument("--bg-percentile", type=float, default=1.0,
                        help="air percentile")
    parser.add_argument("--bg-background", choices=["dark", "bright"],
                        default="dark", help="background phase")


def _detect_output(args) -> Path:
    if getattr(args, "output", None):
        return Path(args.output)
    input_path = Path(args.input)
    return input_path.with_name(input_path.stem + "_processed.tif")


def cmd_info(args) -> int:
    from .io import load_volume

    data = load_volume(args.input)
    nz, ny, nx = data.shape
    kind = "projections" if nz > 1.5 * max(ny, nx) else "volume"
    print(f"file      : {args.input}")
    print(f"shape     : {data.shape}  (nz={nz}, ny={ny}, nx={nx})")
    print(f"dtype     : {data.dtype}")
    print(f"range     : [{np.nanmin(data):.4g}, {np.nanmax(data):.4g}]")
    print(f"kind      : {kind} (heuristic)")
    from .metrics import cupping_index

    if kind == "volume":
        print(f"cupping   : {cupping_index(data[nz // 2]):.3f} (center slice)")
    return 0


def cmd_align(args) -> int:
    from .align import align_slices
    from .io import load_volume, save_volume

    vol = load_volume(args.input)
    print(f"aligning {vol.shape[0]} slices of {args.input} ...")
    aligned, res = align_slices(
        vol,
        estimate_theta=not args.no_rotation,
        smoothing=args.smoothing,
        smoothing_degree=args.smoothing_degree,
        reference=args.reference,
        interpolation_order=args.order,
        progress=args.progress,
    )
    out = _detect_output(args)
    save_volume(aligned, out)
    scores = np.asarray(res.residual_scores[1:])
    print(f"saved {out}")
    print(f"pair quality: median {np.median(scores):.3f}  min {scores.min():.3f}")
    print("cumulative rotation (deg): "
          + ", ".join(f"{np.degrees(t):+.2f}" for t in res.rotations[:: max(1, len(res.rotations) // 6)]))
    return 0


def cmd_deharden(args) -> int:
    from .beam_hardening import remove_beam_hardening
    from .io import load_volume, save_volume

    data = load_volume(args.input)
    print(f"dehardening {args.input} (method={args.method}) ...")
    corrected, res = remove_beam_hardening(
        data,
        method=args.method,
        degree=args.degree,
        background=args.background,
        center=tuple(args.center) if args.center else None,
        layout=args.layout,
        coefficients=np.array(args.coeffs) if args.coeffs else None,
        progress=args.progress,
    )
    out = _detect_output(args)
    save_volume(corrected, out)
    print(f"saved {out}")
    if not np.isnan(res.cupping_before):
        print(f"cupping: {res.cupping_before:.3f} -> {res.cupping_after:.3f}")
    if res.coefficients is not None:
        c = np.asarray(res.coefficients)
        print("coefficients: " + np.array2string(np.ravel(c)[-4:], precision=4))
    return 0


def cmd_background(args) -> int:
    from .background import correct_background
    from .io import load_volume, save_volume

    data = load_volume(args.input)
    print(f"background correction on {args.input} (mode={args.mode}) ...")
    corrected, res = correct_background(
        data,
        mode=args.mode,
        per_slice=not args.shared,
        air_percentile=args.percentile,
        background=args.background,
    )
    out = _detect_output(args)
    save_volume(corrected, out)
    print(f"saved {out}")
    print(f"air: {res.air_before:.4g} -> {res.air_after:.4g}")
    print("background spread: "
          f"{res.background_std_before:.4g} -> "
          f"{res.background_std_after:.4g}")
    return 0


def cmd_run(args) -> int:
    from .pipeline import process

    report = process(
        args.input,
        output_path=args.output,
        align=not args.no_align,
        deharden=not args.no_deharden,
        kind=args.kind,
        align_options={
            "estimate_theta": not args.no_rotation,
            "smoothing": args.align_smoothing,
            "reference": args.align_reference,
        },
        deharden_options={
            "method": args.deharden_method,
            "degree": args.degree,
            "background": args.background,
            "layout": args.layout,
        },
        background=getattr(args, "with_background", False),
        background_options={
            "mode": args.bg_mode,
            "per_slice": not args.bg_shared,
            "air_percentile": args.bg_percentile,
            "background": args.bg_background,
        },
        progress=args.progress,
    )
    print(report.summary())
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo

    return run_demo(out_dir=args.out, n_slices=args.slices, size=args.size)


def cmd_gui(args) -> int:
    from .gui import launch

    return launch(open_path=args.file)


def cmd_report(args) -> int:
    from .io import load_volume, read_metadata
    from .report import (collect_report_data, generate_report,
                         render_slice_u8)

    data = load_volume(args.input)
    meta = read_metadata(args.input)
    nz, ny, nx = data.shape
    kind = "projections" if nz > 1.5 * max(ny, nx) else "volume"
    idx = data.shape[0] // 2
    u8 = None if args.no_image else render_slice_u8(data[idx])
    rd = collect_report_data(
        data, kind=kind, source=str(args.input), meta=meta,
        slice_index=idx if u8 is not None else None,
        u8=u8, colormap="Gray" if u8 is not None else None)
    ext = ".pdf" if args.pdf else ".docx" if args.docx \
        else ".txt" if args.txt else ".html"
    if args.output:
        out = Path(args.output)
    else:
        out = Path(args.input).with_name(
            Path(args.input).stem + "_report" + ext)
    generate_report(rd, out, data=data, u8=u8)
    print(f"report written: {out}")
    return 0


def cmd_roi(args) -> int:
    from .background import _percentile_level
    from .io import load_volume, read_metadata
    from .roi import roi_stats, roi_stats_series, write_roi_csv

    data = load_volume(args.input)
    meta = read_metadata(args.input)
    rect = tuple(args.rect)
    if args.all_slices or args.csv:
        series = roi_stats_series(data, rect, meta=meta, with_air=True)
        if args.csv:
            out = Path(args.input).with_name(
                Path(args.input).stem + "_roi.csv")
            write_roi_csv(series, out)
            print(f"wrote {out} ({len(series)} rows)")
        else:
            for s in series:
                d = s.as_dict()
                print(f"slice {d['slice']:>3}: mean {d['mean']:.6g}  "
                      f"std {d['std']:.6g}  min/max {d['min']:.6g}/"
                      f"{d['max']:.6g}"
                      + (f"  area {d['area_mm2']:.4g} mm2"
                         if "area_mm2" in d else ""))
    else:
        idx = max(0, min(args.slice, data.shape[0] - 1))
        air = _percentile_level(
            np.asarray(data[idx], dtype=np.float64), 1.0)
        s = roi_stats(data, rect, idx, meta=meta, air_reference=air)
        d = s.as_dict()
        print(f"ROI {d['x0']},{d['y0']}..{d['x1']},{d['y1']} on slice "
              f"{d['slice']} ({d['n_pixels']} px)")
        print(f"  mean   : {d['mean']:.6g}")
        print(f"  std    : {d['std']:.6g}")
        print(f"  min/max: {d['min']:.6g} / {d['max']:.6g}")
        print(f"  median : {d['median']:.6g}")
        if "area_mm2" in d:
            print(f"  area   : {d['area_mm2']:.4g} mm2")
        if "mean_above_air" in d:
            print(f"  mean-above-air: {d['mean_above_air']:.6g}")
    return 0


def cmd_compare(args) -> int:
    from .io import load_volume

    a = load_volume(args.a)
    b = load_volume(args.b)
    res = compare_volumes(a, b)
    print(res.as_report())
    print(f"a: {args.a}")
    print(f"b: {args.b}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "info": cmd_info,
        "align": cmd_align,
        "deharden": cmd_deharden,
        "background": cmd_background,
        "run": cmd_run,
        "demo": cmd_demo,
        "gui": cmd_gui,
        "compare": cmd_compare,
        "report": cmd_report,
        "roi": cmd_roi,
    }
    try:
        return handlers[args.cmd](args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
