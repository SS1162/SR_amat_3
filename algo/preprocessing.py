"""Image normalization shared by training and inference.

Kept free of heavy imports (no PIL / torchvision) so inference can import it
without pulling in training dependencies.

STATUS: signature only. See docs/inference_required_changes.md.
"""

from __future__ import annotations

import numpy as np

__all__ = ["normalize_image"]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize a raw grayscale image to a 2D float32 array in [0, 1].
    Shared by training and inference so both normalize identically."""
    raise NotImplementedError
