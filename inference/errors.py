"""Defined failures of an inference run.

Every error must be constructible from a single message string: PyTorch re-raises
DataLoader-worker exceptions by calling the exception type with one message. For
that reason no class here defines a ``message`` attribute (PyTorch would then call
the type with ``message=`` instead).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "InferenceError",
    "NoInputImagesError",
    "InputResolveError",
    "ModelLoadError",
    "ImageLoadError",
    "ModelRunError",
    "ImageSaveError",
]


class InferenceError(Exception):
    """Base class for every defined inference failure.

    The optional ``file_name`` is stored on the instance and also appended to the
    message, so it survives a re-raise from a DataLoader worker.
    """

    def __init__(self, message: str, file_name: str | None = None) -> None:
        """Build the error from a message and an optional related file name."""
        if file_name is not None:
            message = f"{message} (file: {file_name})"
        super().__init__(message)
        self.file_name = file_name


class NoInputImagesError(InferenceError):
    """The input path resolved to no supported images."""


class InputResolveError(InferenceError):
    """The input path could not be resolved to a list of images."""


class ModelLoadError(InferenceError):
    """The checkpoint could not be read or turned into a model."""


class ImageLoadError(InferenceError):
    """An input image could not be loaded or preprocessed."""


class ModelRunError(InferenceError):
    """The model failed while processing an image."""


class ImageSaveError(InferenceError):
    """A super-resolved image could not be written."""
