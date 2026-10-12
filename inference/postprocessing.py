"""Conversion of model output to a displayable / savable image."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["postprocess_output"]


def postprocess_output(output: torch.Tensor) -> np.ndarray:
    """Convert the model output tensor of shape [1, 1, H, W] (on any
    device) to a 2D float32 array in [0, 1] on the CPU."""
    return output[0, 0].clamp(0, 1).float().cpu().numpy()
