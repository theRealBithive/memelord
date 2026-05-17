"""Tests for core.brain (DINOv2 + classifier)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from core import brain


def test_brain_module_imports() -> None:
    """Brain module can be imported."""
    assert brain is not None


def test_get_transform_returns_compose() -> None:
    """get_transform returns a torchvision Compose."""
    t = brain.get_transform()
    assert t is not None
    # Compose is callable and accepts PIL Image
    img = Image.new("RGB", (100, 100), color="red")
    out = t(img)
    assert out.shape == (3, 224, 224)


def test_is_image_path_true_for_image_extensions() -> None:
    """is_image_path returns True for supported extensions."""
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"):
        assert brain.is_image_path(Path(f"x{ext}")) is True


def test_is_image_path_false_for_non_image() -> None:
    """is_image_path returns False for non-image extensions."""
    assert brain.is_image_path(Path("x.txt")) is False
    assert brain.is_image_path(Path("x")) is False


def test_encode_empty_paths_returns_empty_array() -> None:
    """encode with no paths returns (0, 768) float32 and empty path list."""
    mock_encoder = torch.nn.Linear(3, 768)  # unused, we only check shape
    result, valid = brain.encode(mock_encoder, [])
    assert result.shape == (0, 768)
    assert result.dtype == np.float32
    assert valid == []


def test_encode_returns_shape_n_768(tmp_path: Path) -> None:
    """encode with a mock encoder and one image returns (1, 768) and the path."""
    Image.new("RGB", (224, 224), color="blue").save(tmp_path / "img.png")
    path = tmp_path / "img.png"

    class MockEncoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.size(0), 768, device=x.device, dtype=x.dtype)

    encoder = MockEncoder()
    result, valid = brain.encode(encoder, [path])
    assert result.shape == (1, 768)
    assert result.dtype == np.float32
    assert valid == [path]


def test_encode_skips_missing_file(tmp_path: Path) -> None:
    """encode silently skips a path that no longer exists on disk."""
    Image.new("RGB", (4, 4)).save(tmp_path / "real.png")
    real = tmp_path / "real.png"
    missing = tmp_path / "gone.png"

    class MockEncoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.size(0), 768)

    result, valid = brain.encode(MockEncoder(), [real, missing])
    assert result.shape == (1, 768)
    assert valid == [real]


def test_save_classifier_load_classifier_roundtrip(tmp_path: Path) -> None:
    """save_classifier and load_classifier roundtrip."""
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(np.random.randn(5, 768), [0, 1, 0, 1, 0])
    path = tmp_path / "weights.pkl"
    brain.save_classifier(clf, path)
    assert path.exists()
    loaded = brain.load_classifier(path)
    assert loaded.predict_proba(np.random.randn(1, 768)).shape == (1, 2)


def test_predict_proba_single_embedding_returns_scalar() -> None:
    """predict_proba with (768,) returns a scalar probability."""
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(np.random.randn(4, 768), [0, 1, 0, 1])
    emb = np.random.randn(768).astype(np.float32)
    proba = brain.predict_proba(clf, emb)
    assert np.isscalar(proba) or proba.shape == ()
    assert 0 <= float(proba) <= 1


def test_predict_proba_batch_returns_array() -> None:
    """predict_proba with (N, 768) returns shape (N,)."""
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(np.random.randn(4, 768), [0, 1, 0, 1])
    emb = np.random.randn(3, 768).astype(np.float32)
    proba = brain.predict_proba(clf, emb)
    assert proba.shape == (3,)
    assert np.all((proba >= 0) & (proba <= 1))
