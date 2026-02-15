"""Tests for core.caption: BLIP-based alt text generation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import pytest


from core import caption
from tests.conftest import minimal_png_bytes


def test_describe_for_alt_returns_string_within_max_length(
    tmp_path: Path,
) -> None:
    """describe_for_alt returns a non-empty string of at most max_length chars."""
    img = tmp_path / "tiny.png"
    img.write_bytes(minimal_png_bytes())
    max_len = 125
    result = caption.describe_for_alt(img, max_length=max_len)
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result) <= max_len


def test_describe_for_alt_fallback_when_empty() -> None:
    """When caption is empty after strip, describe_for_alt returns 'Image'."""
    with patch.object(caption, "_processor", None), patch.object(
        caption, "_model", None
    ):
        with patch("core.caption._get_model") as mock_get:
            mock_proc = MagicMock()
            mock_model = MagicMock()
            mock_get.return_value = (mock_proc, mock_model)
            mock_proc.return_value = {"input_ids": torch.zeros(1, 1).long()}
            mock_model.generate.return_value = torch.zeros(1, 5).long()
            mock_proc.decode.return_value = "   \n  "
            caption._processor = mock_proc
            caption._model = mock_model

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(minimal_png_bytes())
                p = Path(f.name)
            try:
                out = caption.describe_for_alt(p, max_length=125)
            finally:
                p.unlink(missing_ok=True)
    assert out == "Image"
