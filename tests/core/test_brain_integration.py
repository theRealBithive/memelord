"""
Opt-in check against the real DINOv3 weights.

Contract: V8 — der neue Encoder ist DINOv3 ViT-B/16 mit Metas veröffentlichter
Vorverarbeitung und liefert 768-dimensionale Vektoren.

Skipped by default (see the ``integration`` gate in tests/conftest.py). Run it
after a torch/transformers bump or on a new machine with

    uv run pytest --run-integration tests/core/test_brain_integration.py

The weights live in a gated repository, so this needs HF_TOKEN for an account
that Meta has granted access. Without it get_transform() raises
EncoderUnavailableError with the licence hint; that failure is the test doing
its job, not flakiness.

Why synthetic images instead of fixtures from data/: the test must not depend
on the operator's library, and two images that differ only in colour and
texture are enough to show that the encoder separates content (cos < 0.95)
while staying deterministic for a repeated input (cos ≈ 1).
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import brain

pytestmark = pytest.mark.integration


def _flat_image(path: Path) -> Path:
    Image.new("RGB", (300, 200), color=(120, 30, 200)).save(path)
    return path


def _checkerboard_image(path: Path) -> Path:
    image = Image.new("RGB", (640, 480), color=(10, 200, 40))
    square = 40
    for x in range(0, 640, square):
        for y in range(0, 480, square):
            is_white_square = (x // square + y // square) % 2 == 0
            if is_white_square:
                image.paste((255, 255, 255), (x, y, x + square, y + square))
    image.save(path)
    return path


def test_transform_matches_published_preprocessing(tmp_path: Path) -> None:
    """Meta's processor resizes to INPUT_SIZE (448) squared and returns a float32 CHW tensor (taste V18)."""
    transform = brain.get_transform()
    tensor = transform(Image.open(_flat_image(tmp_path / "a.png")).convert("RGB"))
    assert tuple(tensor.shape) == (3, brain.INPUT_SIZE, brain.INPUT_SIZE)
    assert str(tensor.dtype) == "torch.float32"


def test_real_encoder_separates_content_and_repeats_identical_input(tmp_path: Path) -> None:
    flat = _flat_image(tmp_path / "flat.png")
    board = _checkerboard_image(tmp_path / "board.png")

    encoder = brain.get_encoder("cpu")
    embeddings, encoded_paths = brain.encode(
        encoder, [flat, board, flat], transform=brain.get_transform(), device="cpu", batch_size=3
    )

    assert encoded_paths == [flat, board, flat]
    assert embeddings.shape == (3, brain.EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()

    similarity = brain.cosine_similarity_matrix(embeddings, embeddings)
    assert similarity[0, 2] > 0.999, "the same file must encode to the same vector"
    assert similarity[0, 1] < 0.95, "different content must not look like a duplicate"
