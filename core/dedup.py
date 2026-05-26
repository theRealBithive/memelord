"""Hybrid deduplication: SHA-256, pHash, and DINO embedding similarity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import brain, phash


@dataclass
class DedupIndex:
    """
    Holds all previously seen images in memory for O(1)/O(N) dedup during a
    scrape run. Three layers are checked in ascending cost order — SHA-256
    (hash-set lookup), pHash (linear Hamming scan), DINO cosine (GEMM) — so
    exact and near-exact duplicates are rejected before the GPU is ever touched.
    Populated once via from_db() at the start of a scrape, then grown with
    add() as new images are inserted, so intra-session duplicates are also
    caught without a DB round-trip per candidate.
    """

    content_hashes: set[str] = field(default_factory=set)
    phashes: list[str] = field(default_factory=list)
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, brain.EMBEDDING_DIM), dtype=np.float32)
    )

    @classmethod
    def from_db(cls) -> DedupIndex:
        """
        Loads all three dedup signals from the DB in a single query so the
        scraper doesn't hit the DB once per candidate. is_purged rows are
        included so purged content_hashes block re-downloads permanently.
        """
        from ratings.models import Image

        rows = list(
            Image.objects.all().values_list(
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
        """
        Keeps the index current after each DB insert so a second image with
        near-identical content scraped in the same session is caught by the
        DINO check rather than slipping through because the index only reflects
        the state at scrape startup. vstack on an empty (0, 768) array raises,
        so the first embedding replaces the zero-row matrix outright.
        """
        self.content_hashes.add(content_hash)
        if ph:
            self.phashes.append(ph)
        if emb is not None:
            row = np.asarray(emb, dtype=np.float32).reshape(1, -1)
            self.embeddings = (
                row if self.embeddings.size == 0 else np.vstack([self.embeddings, row])
            )


def content_hash_for_file(path: Path) -> str:
    """
    SHA-256 is the dedup primary key — it's stored as content_hash on every
    Image row and checked first in the pipeline because it's a hash-set lookup
    with no IO beyond reading the file once. Used by tests and utilities;
    the scraper computes the hash inline during _sha_filter().
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha_duplicate(content_hash: str, index: DedupIndex) -> bool:
    return content_hash in index.content_hashes


def is_phash_duplicate_for_path(
    path: Path, index: DedupIndex, max_distance: int
) -> tuple[bool, str]:
    """
    Second gate in the SHA → pHash → DINO pipeline. pHash catches resized,
    re-cropped, or JPEG-recompressed versions of known images that have a
    different SHA-256. Returns the computed phash alongside the bool so the
    caller can store it in the DB without a second read of the file.
    """
    ph = phash.compute_phash(path)
    dup = phash.is_phash_duplicate(ph, index.phashes, max_distance)
    return dup, ph


def is_embedding_duplicate(
    emb: np.ndarray,
    index: DedupIndex,
    threshold: float,
) -> bool:
    """
    Final gate: catches semantically identical images that survived SHA and
    pHash (different crop, palette shift, added watermark). The default
    threshold of 0.92 is high enough to reject near-duplicates while
    preserving genuinely different images that happen to share a composition.
    """
    if index.embeddings.size == 0:
        return False
    sims = brain.cosine_similarity_matrix(emb, index.embeddings)
    return bool(np.max(sims) >= threshold)
