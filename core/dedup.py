"""Hybrid deduplication: SHA-256, pHash, and DINO embedding similarity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import brain, phash


@dataclass
class DedupIndex:
    """In-memory index of known images for scrape-time dedup."""

    content_hashes: set[str] = field(default_factory=set)
    phashes: list[str] = field(default_factory=list)
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, brain.EMBEDDING_DIM), dtype=np.float32)
    )

    @classmethod
    def from_db(cls) -> DedupIndex:
        """Build index from all non-deleted Image rows."""
        from ratings.models import Image

        rows = list(
            Image.objects.filter(file_deleted=False).values_list(
                "content_hash", "phash", "embedding"
            )
        )
        content_hashes = {r[0] for r in rows}
        phashes = [r[1] for r in rows if r[1]]
        embeddings_list = [brain.bytes_to_embedding(r[2]) for r in rows if r[2]]
        embeddings = (
            np.vstack(embeddings_list)
            if embeddings_list
            else np.zeros((0, brain.EMBEDDING_DIM), dtype=np.float32)
        )
        return cls(
            content_hashes=content_hashes,
            phashes=phashes,
            embeddings=embeddings,
        )

    def add(self, content_hash: str, ph: str, emb: np.ndarray | None) -> None:
        """Register a newly inserted image in the index."""
        self.content_hashes.add(content_hash)
        if ph:
            self.phashes.append(ph)
        if emb is not None:
            row = np.asarray(emb, dtype=np.float32).reshape(1, -1)
            self.embeddings = (
                row if self.embeddings.size == 0 else np.vstack([self.embeddings, row])
            )


def content_hash_for_file(path: Path) -> str:
    """SHA-256 hex digest of file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha_duplicate(content_hash: str, index: DedupIndex) -> bool:
    return content_hash in index.content_hashes


def is_phash_duplicate_for_path(
    path: Path, index: DedupIndex, max_distance: int
) -> tuple[bool, str]:
    """
    Compute pHash for path and check against index.

    Returns (is_duplicate, phash_hex).
    """
    ph = phash.compute_phash(path)
    dup = phash.is_phash_duplicate(ph, index.phashes, max_distance)
    return dup, ph


def is_embedding_duplicate(
    emb: np.ndarray,
    index: DedupIndex,
    threshold: float,
) -> bool:
    """True if cosine similarity to any indexed embedding exceeds threshold."""
    if index.embeddings.size == 0:
        return False
    sims = brain.cosine_similarity_matrix(emb, index.embeddings)
    return bool(np.max(sims) >= threshold)
