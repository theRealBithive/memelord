"""Validate downloaded image files (PIL-readable) for retina scrapers."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError


def is_readable_image(path: Path) -> bool:
    """
    Return True if path is a file that PIL can open and decode as RGB.

    Use after writing a download to disk; if False, the file is corrupted or
    not a valid image and should be removed.
    """
    try:
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except (UnidentifiedImageError, OSError):
        return False
