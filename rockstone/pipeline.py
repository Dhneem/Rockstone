"""One-call processing pipeline: align + deharden + background + save +
report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .align import align_projections, align_slices
from .background import correct_background
from .beam_hardening import remove_beam_hardening
from .io import load_volume, save_volume
from .metrics import cupping_index

__all__ = ["ProcessingReport", "process"]


@dataclass
class ProcessingReport:
    """Summary of a :func:`process` run."""

    input_path: str
    kind: str  # "volume" or "projections"
    shape: tuple[int, ...]
    steps_run: list[str] = field(default_factory=list)
    output_path: str | None = None
    alignment: object | None = None
    beam_hardening: object | None = None
    background: object | None = None
    cupping_before: float = float("nan")
    cupping_after: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"input      : {self.input_path} ({self.kind}, shape {self.shape})",
        ]
        if self.output_path:
            lines.append(f"output     : {self.output_path}")
        lines.append("steps      : " + (", ".join(self.steps_run) or "none"))
        if not np.isnan(self.cupping_before):
            lines.append(
                f"cupping    : {self.cupping_before:.3f} -> {self.cupping_after:.3f}"
            )
        if self.background is not None:
            lines.append(
                f"air level  : {self.background.air_before:.4g} -> "
                f"{self.background.air_after:.4g}"
            )
        if self.alignment is not None and hasattr(self.alignment, "residual_scores"):
            scores = np.asarray(self.alignment.residual_scores)
            if scores.size > 1:
                lines.append(
                    f"pair quality: median {np.median(scores[1:]):.3f}, "
                    f"min {scores[1:].min():.3f}"
                )

        for n in self.notes:
            lines.append(f"note       : {n}")
        return "\n".join(lines)


def process(
    input_path: str | Path,
    output_path: str | Path | None = None,
    align: bool = True,
    deharden: bool = True,
    background: bool = False,
    kind: str | None = None,
    align_options: dict | None = None,
    deharden_options: dict | None = None,
    background_options: dict | None = None,
    progress: bool = False,
) -> ProcessingReport:
    """Run the full rock-CT workflow on ``input_path``.

    Parameters
    ----------
    input_path:
        Stack or projection file/folder (see :mod:`rockstone.io`).
    output_path:
        Where to write the processed data (TIFF/NPY). Defaults to
        ``<input>_processed.tif`` next to the input.
    align, deharden, background:
        Toggle the stages; at least one must be requested. The
        background stage is opt-in (``background=True``) so existing
        workflows keep their results.
    kind:
        ``"volume"`` or ``"projections"``; auto-detected when omitted.
    align_options / deharden_options / background_options:
        Extra keyword arguments forwarded to :func:`align_slices` /
        :func:`align_projections`, :func:`remove_beam_hardening` and
        :func:`correct_background`.
    progress:
        Show per-slice/per-projection progress lines.

    Returns
    -------
    :class:`ProcessingReport` (also pretty-printable via ``.summary()``).
    """
    input_path = Path(input_path)
    align_options = dict(align_options or {})
    deharden_options = dict(deharden_options or {})
    background_options = dict(background_options or {})
    if not align and not deharden and not background:
        raise ValueError("nothing to do: enable `align`, `deharden` "
                         "and/or `background`")

    data = load_volume(input_path)
    if kind is None:
        nz, ny, nx = data.shape
        kind = "projections" if nz > 1.5 * max(ny, nx) else "volume"

    report = ProcessingReport(
        input_path=str(input_path), kind=kind, shape=tuple(data.shape)
    )
    current = data.astype(np.float64)

    if kind == "projections":
        if align:
            current, res = align_projections(current, progress=progress,
                                             **align_options)
            report.steps_run.append("align_projections")
            report.alignment = res
        if deharden:
            opts = {"method": "chord", "layout": "detector"}
            opts.update(deharden_options)
            current, res = remove_beam_hardening(current, progress=progress, **opts)
            report.steps_run.append("deharden_chord")
            report.beam_hardening = res
            report.notes.append(
                "cupping metrics require reconstructed slices; skipped for projections"
            )
    else:
        mid = data[data.shape[0] // 2]
        report.cupping_before = cupping_index(mid)
        if align:
            current, res = align_slices(current, progress=progress, **align_options)
            report.steps_run.append("align_slices")
            report.alignment = res
        if deharden:
            opts = {"method": "radial"}
            opts.update(deharden_options)
            current, res = remove_beam_hardening(current, progress=progress, **opts)
            report.steps_run.append("deharden_radial")
            report.beam_hardening = res
            mid_out = current[current.shape[0] // 2]
            report.cupping_after = cupping_index(mid_out)

    if background:
        current, res = correct_background(current, **background_options)
        report.steps_run.append(
            f"background_{res.mode}" + ("" if res.per_slice else "_shared")
        )
        report.background = res

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_processed.tif")
    save_volume(np.asarray(current), output_path)
    report.output_path = str(output_path)
    return report
