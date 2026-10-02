"""Tests for core.dedup."""

from pathlib import Path

import numpy as np
from PIL import Image

from core import dedup, phash


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


def _unit_vector(axis: int) -> np.ndarray:
    vector = np.zeros(768, dtype=np.float32)
    vector[axis] = 1.0
    return vector


def test_add_registers_all_three_signals() -> None:
    index = dedup.DedupIndex()
    first = np.linspace(0.0, 1.0, 768, dtype=np.float64)
    second = np.linspace(1.0, 2.0, 768, dtype=np.float64)

    index.add("hash-a", "ff00", first)
    index.add("hash-b", "0f0f", second)

    assert index.content_hashes == {"hash-a", "hash-b"}
    assert index.phash_ints == [int("ff00", 16), int("0f0f", 16)]
    assert index.embeddings.shape == (2, 768)
    assert index.embeddings.dtype == np.float32
    np.testing.assert_allclose(index.embeddings[0], first, rtol=1e-6)
    np.testing.assert_allclose(index.embeddings[1], second, rtol=1e-6)


def test_add_without_phash_or_embedding_only_records_the_hash() -> None:
    index = dedup.DedupIndex()

    index.add("hash-only", "", None)

    assert index.content_hashes == {"hash-only"}
    assert index.phash_ints == []
    assert index.embeddings.shape == (0, 768)


def test_is_phash_duplicate_for_path_returns_verdict_and_hash(tmp_path: Path) -> None:
    gradient = Image.linear_gradient("L").resize((64, 64)).convert("RGB")
    path = tmp_path / "gradient.png"
    gradient.save(path)
    expected_hash = phash.compute_phash(path)

    empty_index = dedup.DedupIndex()
    seen_index = dedup.DedupIndex(phash_ints=[int(expected_hash, 16)])

    assert dedup.is_phash_duplicate_for_path(path, empty_index, max_distance=0) == (False, expected_hash)
    assert dedup.is_phash_duplicate_for_path(path, seen_index, max_distance=0) == (True, expected_hash)


def test_is_embedding_duplicate_at_exactly_the_threshold() -> None:
    """A similarity equal to the threshold counts as a duplicate (>=, not >)."""
    index = dedup.DedupIndex(embeddings=_unit_vector(0).reshape(1, -1))

    assert dedup.is_embedding_duplicate(_unit_vector(0), index, threshold=1.0) is True
    assert dedup.is_embedding_duplicate(_unit_vector(1), index, threshold=1.0) is False
