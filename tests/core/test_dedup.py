"""Tests for core.dedup."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from core import brain, dedup


def test_is_sha_duplicate() -> None:
    """SHA index membership detects exact duplicates."""
    index = dedup.DedupIndex(content_hashes={"abc"})
    assert dedup.is_sha_duplicate("abc", index)
    assert not dedup.is_sha_duplicate("def", index)


def test_is_embedding_duplicate_above_threshold() -> None:
    """High cosine similarity is treated as duplicate."""
    emb = np.ones(768, dtype=np.float32)
    bank = np.ones((2, 768), dtype=np.float32) * 0.99
    index = dedup.DedupIndex(embeddings=bank)
    assert dedup.is_embedding_duplicate(emb, index, threshold=0.92)


def test_is_embedding_duplicate_empty_bank() -> None:
    """Empty embedding bank never matches."""
    emb = np.ones(768, dtype=np.float32)
    index = dedup.DedupIndex()
    assert not dedup.is_embedding_duplicate(emb, index, threshold=0.92)


def test_content_hash_for_file(tmp_path: Path) -> None:
    """content_hash_for_file matches hashlib SHA-256."""
    import hashlib

    path = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(path)
    assert (
        dedup.content_hash_for_file(path)
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
