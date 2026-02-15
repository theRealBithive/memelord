"""Tests for retina.image_validation."""

from pathlib import Path

import pytest

from retina import image_validation
from tests.conftest import minimal_png_bytes


def test_image_validation_module_imports() -> None:
    """image_validation module can be imported."""
    assert image_validation is not None


def test_is_readable_image_returns_true_for_valid_image(tmp_path: Path) -> None:
    """is_readable_image returns True when PIL can open and decode the file."""
    path = tmp_path / "valid.png"
    path.write_bytes(minimal_png_bytes())
    assert image_validation.is_readable_image(path) is True


def test_is_readable_image_returns_false_for_junk_bytes(tmp_path: Path) -> None:
    """is_readable_image returns False for corrupted or non-image content."""
    path = tmp_path / "junk.webp"
    path.write_bytes(b"not an image file")
    assert image_validation.is_readable_image(path) is False


def test_is_readable_image_returns_false_for_missing_file(tmp_path: Path) -> None:
    """is_readable_image returns False when file does not exist."""
    path = tmp_path / "missing.png"
    assert path.exists() is False
    assert image_validation.is_readable_image(path) is False
