"""Raw grayscale image loading shared by training and inference.

Kept separate from ``algo.preprocessing`` so that module stays free of PIL.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = ["load_image"]

# Pillow modes that are already single-channel grayscale; their raw values are kept as is.
GRAYSCALE_MODES = ("L", "I;16", "I;16L", "I;16B", "I", "F")
# Mode used for anything else (colour, palette, bilevel); all of those are 8 bits per channel.
FALLBACK_GRAYSCALE_MODE = "L"


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
