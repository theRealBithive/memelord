"""
Opt-in check against the real SigLIP2 weights.

Contract: V9 — der Such-Encoder ist SigLIP2 ViT-B/16 bei 224 px mit Googles
veröffentlichter Vorverarbeitung; Suchtexte werden auf 64 Token aufgefüllt;
Vektoren sind 768-dimensionale float32-Vektoren.

Skipped by default (see the ``integration`` gate in tests/conftest.py). Run it
after a torch/transformers bump or on a new machine with

    uv run pytest --run-integration tests/core/test_siglip_integration.py

The repo is not gated, so no HF_TOKEN is needed, but the first run downloads
about 1.5 GB. The test covers the risks the fakes cannot: R1 (wrong padding
degrades queries), R2 (wrong pooling gives vectors outside the text space) and
R3 (vision and text towers both load as sub-models of the combined checkpoint).
Two plain-colour images are enough: a correct text/image alignment ranks the
matching colour first, in English and in German.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import brain, siglip

pytestmark = pytest.mark.integration


def _plain_image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (224, 224), color=color).save(path)
    return path


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


@pytest.fixture(scope="module")
def image_vectors(tmp_path_factory) -> np.ndarray:
    folder = tmp_path_factory.mktemp("siglip")
    red = _plain_image(folder / "red.png", (255, 0, 0))
    blue = _plain_image(folder / "blue.png", (0, 0, 255))
    embeddings, valid = brain.encode(
        siglip.get_image_encoder("cpu"),
        [red, blue],
        transform=siglip.get_image_transform(),
        device="cpu",
        progress_label="siglip-test",
    )
    assert valid == [red, blue]
    return embeddings


def test_image_encoder_returns_two_768_float32_vectors(image_vectors: np.ndarray) -> None:
    assert image_vectors.shape == (2, siglip.SEARCH_DIM)
    assert image_vectors.dtype == np.float32


def test_text_tower_is_the_text_model_from_the_combined_checkpoint() -> None:
    model, _tokenizer = siglip.get_text_encoder()
    assert model.__class__.__name__ == "SiglipTextModel"


@pytest.mark.parametrize(
    ("query", "expected_row"),
    [
        ("a red square", 0),
        ("a blue square", 1),
        ("ein rotes Quadrat", 0),
        ("ein blaues Quadrat", 1),
    ],
)
def test_text_prefers_the_image_of_the_named_colour(
    image_vectors: np.ndarray, query: str, expected_row: int
) -> None:
    text_vector = siglip.encode_text(query)
    similarities = _unit(image_vectors) @ text_vector
    other_row = 1 - expected_row
    assert similarities[expected_row] > similarities[other_row]
