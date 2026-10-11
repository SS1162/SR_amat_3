"""Image file access for inference.

Named ``image_io`` (not ``io``) to avoid shadowing the standard-library module.
Image loading lives in ``algo.image_loading`` (shared with training).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_IMAGE_EXTENSIONS",
    "OUTPUT_SUFFIX",
    "OUTPUT_EXTENSION",
    "resolve_inputs",
    "save_image",
    "output_path_for",
]

# Matches the formats the Inference tab advertises (PNG, JPG, TIFF); compared case-insensitively.
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
OUTPUT_SUFFIX = "_sr"
OUTPUT_EXTENSION = ".tif"
HIDDEN_FILE_PREFIX = "."
# Splits a name into text and digit runs, e.g. "img10.png" -> ["img", "10", ".png"].
DIGIT_RUN_PATTERN = re.compile(r"(\d+)")

UINT16_MAX = np.iinfo(np.uint16).max
TIFF_FORMAT = "TIFF"


def _natural_sort_key(path: Path) -> tuple[tuple[tuple[int, int | str], ...], str]:
    """Return a key ordering digit runs numerically ("img2" before "img10"), case-insensitively.

    The exact name breaks ties (e.g. "A.png" vs "a.png") so the order never depends on the filesystem.
    """
    parts = DIGIT_RUN_PATTERN.split(path.name.lower())
    natural = tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)
    return natural, path.name


def _is_supported_image(path: Path) -> bool:
    """Return True for a visible file with a supported (case-insensitive) extension."""
    return (
        path.is_file()
        and not path.name.startswith(HIDDEN_FILE_PREFIX)
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def resolve_inputs(input_path: Path) -> list[Path]:
    """Return the images to process, at their ORIGINAL location (nothing is copied).

    A file yields ``[input_path]``. A folder yields the visible files directly inside it
    (no recursion) whose extension is in ``SUPPORTED_IMAGE_EXTENSIONS``, in natural order.
    The sorting is a hard requirement: ``start_from`` indexes into this list, so the
    same folder must always produce the same order.
    """
    if input_path.is_file():
        return [input_path]
    images = [path for path in input_path.iterdir() if _is_supported_image(path)]
    return sorted(images, key=_natural_sort_key)


def save_image(image: np.ndarray, path: Path) -> None:
    """Write ``image`` (2D float32 in [0, 1]) to ``path`` as a 16-bit TIFF.

    Overwrites an existing file at ``path``. Values outside [0, 1] are clipped.
    """
    scaled = np.rint(np.clip(image, 0.0, 1.0) * UINT16_MAX).astype(np.uint16)
    Image.fromarray(scaled).save(path, format=TIFF_FORMAT)


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    """Return ``<output_dir>/<input_stem>_sr.tif`` for the given input image."""
    return output_dir / f"{input_path.stem}{OUTPUT_SUFFIX}{OUTPUT_EXTENSION}"
