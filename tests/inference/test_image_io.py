"""Unit tests for inference.image_io.resolve_inputs."""

from __future__ import annotations

from pathlib import Path

from inference.image_io import resolve_inputs


def _touch(folder: Path, *names: str) -> None:
    """Create empty files ``names`` inside ``folder``."""
    for name in names:
        (folder / name).write_bytes(b"")


def test_folder_is_filtered_and_natural_sorted(tmp_path: Path) -> None:
    """Unsupported extensions and hidden files are skipped, suffixes match case-insensitively, order is natural."""
    _touch(tmp_path, "img10.png", "img2.PNG", "img1.tif", ".hidden.png", "notes.txt")

    paths = resolve_inputs(tmp_path)

    assert [path.name for path in paths] == ["img1.tif", "img2.PNG", "img10.png"]


def test_single_file_is_returned_as_is(tmp_path: Path) -> None:
    """A file path yields a one-element list with that path."""
    image = tmp_path / "single.tiff"
    image.write_bytes(b"")

    assert resolve_inputs(image) == [image]
