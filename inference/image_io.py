"""Image file access for inference.

Named ``image_io`` (not ``io``) to avoid shadowing the standard-library module.

STATUS: ``resolve_inputs`` is a signature only; its docstring defines the contract.
See docs/inference_required_changes.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_IMAGE_EXTENSIONS",
    "OUTPUT_SUFFIX",
    "OUTPUT_EXTENSION",
    "resolve_inputs",
    "load_image",
    "save_image",
    "output_path_for",
]

# Matches the formats the Inference tab advertises (PNG, JPG, TIFF); compared case-insensitively.
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
OUTPUT_SUFFIX = "_sr"
OUTPUT_EXTENSION = ".tif"

# Pillow modes that are already single-channel grayscale; their raw values are kept as is.
GRAYSCALE_MODES = ("L", "I;16", "I;16L", "I;16B", "I", "F")
# Mode used for anything else (colour, palette, bilevel); all of those are 8 bits per channel.
FALLBACK_GRAYSCALE_MODE = "L"
UINT16_MAX = np.iinfo(np.uint16).max
TIFF_FORMAT = "TIFF"


def resolve_inputs(input_path: Path) -> list[Path]:
    """Return the images to process, at their ORIGINAL location (nothing is copied).

    A file yields ``[input_path]``. A folder yields its files whose extension is in
    ``SUPPORTED_IMAGE_EXTENSIONS``, sorted deterministically (by file name). The
    sorting is a hard requirement: ``start_from`` indexes into this list, so the
    same folder must always produce the same order.
    """
    raise NotImplementedError


def load_image(path: Path) -> np.ndarray:
    """Read one grayscale image directly from ``path`` (no copying, no caching).

    Returns the raw 2D array in its original dtype (e.g. uint8 / uint16).
    No normalization: that is ``algo.preprocessing.normalize_image``'s job.
    Colour / palette images are converted to 8-bit grayscale (``"L"``).
    """
    with Image.open(path) as image:
        if image.mode not in GRAYSCALE_MODES:
            image = image.convert(FALLBACK_GRAYSCALE_MODE)
        return np.array(image)


def save_image(image: np.ndarray, path: Path) -> None:
    """Write ``image`` (2D float32 in [0, 1]) to ``path`` as a 16-bit TIFF.

    Overwrites an existing file at ``path``. Values outside [0, 1] are clipped.
    """
    scaled = np.rint(np.clip(image, 0.0, 1.0) * UINT16_MAX).astype(np.uint16)
    Image.fromarray(scaled).save(path, format=TIFF_FORMAT)


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    """Return ``<output_dir>/<input_stem>_sr.tif`` for the given input image."""
    return output_dir / f"{input_path.stem}{OUTPUT_SUFFIX}{OUTPUT_EXTENSION}"
