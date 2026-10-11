"""Image normalization shared by training and inference.

Kept free of heavy imports (no PIL / torchvision) so inference can import it
without pulling in training dependencies.

Normalization is by dtype: uint8 is divided by 255 and uint16 by 65535. Known
limitation: 12-bit data stored in uint16 uses only 1/16 of the range and appears dark.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["normalize_image"]

IMAGE_NDIM = 2
# Divisor per supported raw dtype: the dtype's maximum value.
DTYPE_MAX = {
    np.dtype(np.uint8): float(np.iinfo(np.uint8).max),
    np.dtype(np.uint16): float(np.iinfo(np.uint16).max),
}


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize a raw grayscale image to a 2D float32 array in [0, 1].
    Shared by training and inference so both normalize identically.

    Raises ``ValueError`` for non-2D input or a dtype other than uint8 / uint16.
    """
    if image.ndim != IMAGE_NDIM:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")
    divisor = DTYPE_MAX.get(image.dtype)
    if divisor is None:
        raise ValueError(f"Unsupported image dtype {image.dtype}; expected uint8 or uint16")
    return (image.astype(np.float32) / np.float32(divisor)).astype(np.float32)
