"""Tests for embedding pack/unpack and cosine similarity."""

import numpy as np
import pytest

from core import brain


def test_embedding_roundtrip() -> None:
    """embedding_to_bytes and bytes_to_embedding are inverse."""
    emb = np.arange(768, dtype=np.float32) * 0.001
    data = brain.embedding_to_bytes(emb)
    restored = brain.bytes_to_embedding(data)
    np.testing.assert_array_equal(emb, restored)


def test_cosine_similarity_matrix_identical() -> None:
    """Identical vectors have cosine similarity 1."""
    v = np.ones(768, dtype=np.float32)
    bank = v.reshape(1, -1)
    sims = brain.cosine_similarity_matrix(v, bank)
    assert float(sims[0]) == pytest.approx(1.0, abs=1e-5)
