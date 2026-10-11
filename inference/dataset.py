"""DataLoader dataset for inference.

Lives in its own importable module because spawned DataLoader workers re-import
it. CPU only: nothing here touches CUDA.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from algo import image_loading, preprocessing

from .errors import ImageLoadError

logger = logging.getLogger(__name__)

__all__ = ["InferenceImageDataset"]


class InferenceImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Loads and preprocesses input images by index.

    Holds only the list of paths, so it stays picklable for spawned workers.
    """

    def __init__(self, paths: list[Path]) -> None:
        """Store the image paths; nothing is opened here."""
        self._paths = paths

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return the ``[1, H, W]`` float32 CPU tensor of image ``index`` and the index itself."""
        path = self._paths[index]
        try:
            raw = image_loading.load_image(path)
            normalized = preprocessing.normalize_image(raw)
            tensor = to_tensor(normalized)
        except Exception as exc:
            raise ImageLoadError("Failed to load image", file_name=path.name) from exc
        return tensor, index
