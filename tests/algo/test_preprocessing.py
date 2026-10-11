"""Unit tests for algo.preprocessing.normalize_image."""

from __future__ import annotations

import numpy as np
import pytest

from algo.preprocessing import normalize_image

UINT8_MAX = np.iinfo(np.uint8).max
UINT16_MAX = np.iinfo(np.uint16).max


@pytest.mark.parametrize(
    ("dtype", "max_value"),
    [(np.uint8, UINT8_MAX), (np.uint16, UINT16_MAX)],
)
def test_integer_dtypes_map_to_unit_range(dtype: type[np.integer], max_value: int) -> None:
    """uint8 / uint16 are divided by their dtype maximum and returned as float32 in [0, 1]."""
    image = np.array([[0, max_value // 2], [max_value // 4, max_value]], dtype=dtype)

    normalized = normalize_image(image)

    assert normalized.dtype == np.float32
    assert normalized.shape == image.shape
    np.testing.assert_allclose(normalized, image.astype(np.float64) / max_value, rtol=1e-6)
    assert normalized.min() == 0.0 and normalized.max() == 1.0


def test_unsupported_dtype_raises() -> None:
    """A dtype other than uint8 / uint16 raises ValueError naming the dtype."""
    image = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="float64"):
        normalize_image(image)
