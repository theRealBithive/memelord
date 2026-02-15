"""Tests for core.trainer."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from core import brain, trainer


def test_trainer_module_imports() -> None:
    """Trainer module can be imported."""
    assert trainer is not None


def test_collect_image_paths_empty_dirs(tmp_path: Path) -> None:
    """collect_image_paths returns empty lists when corpus/void are empty."""
    (tmp_path / "corpus").mkdir()
    (tmp_path / "void").mkdir()
    corpus, void = trainer.collect_image_paths(tmp_path)
    assert corpus == []
    assert void == []


def test_collect_image_paths_finds_only_images(tmp_path: Path) -> None:
    """collect_image_paths returns only files with image extensions."""
    (tmp_path / "corpus").mkdir()
    (tmp_path / "void").mkdir()
    (tmp_path / "corpus" / "a.jpg").touch()
    (tmp_path / "corpus" / "b.png").touch()
    (tmp_path / "corpus" / "readme.txt").touch()
    (tmp_path / "void" / "c.webp").touch()
    corpus, void = trainer.collect_image_paths(tmp_path)
    assert len(corpus) == 2
    assert len(void) == 1
    assert all(p.suffix.lower() in brain.IMAGE_EXTENSIONS for p in corpus + void)


def test_run_saves_weights(tmp_path: Path) -> None:
    """run() encodes corpus/void, fits classifier, saves weights."""
    (tmp_path / "corpus").mkdir()
    (tmp_path / "void").mkdir()
    (tmp_path / "corpus" / "pos.png").touch()
    (tmp_path / "void" / "neg.png").touch()
    # Create minimal valid images so encode can load them
    from PIL import Image

    Image.new("RGB", (10, 10), color="red").save(tmp_path / "corpus" / "pos.png")
    Image.new("RGB", (10, 10), color="blue").save(tmp_path / "void" / "neg.png")

    weights_path = tmp_path / "weights.pkl"
    with patch.object(brain, "get_encoder") as mock_get_encoder:
        with patch.object(brain, "encode") as mock_encode:
            mock_get_encoder.return_value = None
            # One call for corpus (1 sample), one for void (1 sample)
            mock_encode.side_effect = [
                np.array([[0.1] * 768], dtype=np.float32),
                np.array([[0.2] * 768], dtype=np.float32),
            ]
            trainer.run(data_dir=tmp_path, weights_path=weights_path)
    assert weights_path.exists()
    clf = brain.load_classifier(weights_path)
    assert clf.predict_proba(np.array([[0.1] * 768])).shape == (1, 2)
