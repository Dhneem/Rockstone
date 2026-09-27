"""Rockstone: image processing for rock samples.

High-level helpers:

- :func:`load_volume` / :func:`save_volume`       -- read/write slice stacks
- :func:`load_stack_from_files`                   -- stack several image files into a volume
- :func:`load_projections` / :func:`save_projections` -- read/write raw data
- :func:`align_slices`                            -- rigid per-slice alignment
- :func:`remove_beam_hardening`                   -- cupping / chord correction
- :func:`correct_background`                      -- flatten the air level/shading
- ``AIR_HU``                                      -- standard air value (-1000 HU)
- :func:`process`                                 -- one-call batch pipeline
- :func:`compare_volumes`                         -- difference metrics between two stacks
- :func:`read_metadata`                           -- physical dimensions stored in a file
"""

from .align import align_slices, align_projections
from .background import correct_background
from .beam_hardening import remove_beam_hardening
from .constants import AIR_HU
from .compare import compare_volumes
from .io import (
    load_volume,
    load_stack_from_files,
    save_volume,
    load_projections,
    save_projections,
    read_data,
    read_metadata,
)
from .metrics import cupping_index, radial_profile
from .pipeline import process

__version__ = "0.1.0"

__all__ = [
    "align_slices",
    "align_projections",
    "correct_background",
    "remove_beam_hardening",
    "AIR_HU",
    "load_volume",
    "load_stack_from_files",
    "save_volume",
    "load_projections",
    "save_projections",
    "read_data",
    "read_metadata",
    "cupping_index",
    "radial_profile",
    "process",
    "compare_volumes",
]
