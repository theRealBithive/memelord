"""One in-process matrix per embedding generation, for ranking images by cosine."""

import threading

import numpy as np

from core import brain
from ratings.models import Image


class VectorBank:
    """
    Holds every current vector of one generation (taste or search) as a single
    unit-normalised matrix, so a text query or an anchor image ranks the whole
    library with one matmul instead of one SQLite read per request.

    Why a bank at all: at 25,000 images the blobs are 75 MB. Reading them from
    SQLite for every search would cost hundreds of milliseconds and the same
    memory churn again, while the matmul itself takes a few milliseconds. The
    matrix lives in the web process; gunicorn forks two workers, so there are
    two copies, 77 MB each at 25k images.

    Why the version is a pair of counts (rows with the current stamp, purged
    rows): both only grow during normal operation, and every event that
    changes the set of vectors moves at least one of them. An index slice
    stamps new rows, a purge raises the purged count (the row keeps its
    vector, the candidate filter drops it), a fresh start takes both to zero.
    A vector is never replaced under the same stamp, because "stale" means
    missing or foreign, so there is no silent content change at an unchanged
    version (contract V16, risk R16). Two COUNT queries on indexed columns
    are far cheaper than a content hash of 75 MB.

    Why field names as strings: the taste bank reads `embedding` /
    `embedding_model`, the search bank `search_embedding` /
    `search_embedding_model`. Two classes with an identical body would be
    harder to keep in step than one class with two field names; this is the
    only place the parameterisation shows.
    """

    def __init__(self, blob_field: str, stamp_field: str, stamp: str) -> None:
        self.blob_field = blob_field
        self.stamp_field = stamp_field
        self.stamp = stamp
        self._lock = threading.Lock()
        self._version: tuple[int, int] | None = None
        self._hashes: list[str] = []
        self._matrix = np.zeros((0, brain.EMBEDDING_DIM), dtype=np.float32)

    def current_version(self) -> tuple[int, int]:
        """The pair of counts described in the class docstring."""
        stamped = Image.objects.filter(**{self.stamp_field: self.stamp}).count()
        purged = Image.objects.filter(is_purged=True).count()
        return (stamped, purged)

    def refresh_if_changed(self) -> bool:
        """Reload the matrix when the version moved; True when it did reload."""
        with self._lock:
            return self._refresh_if_changed_locked()

    def rank(self, query: np.ndarray) -> tuple[list[str], np.ndarray]:
        """
        Every hash of the bank ordered by cosine similarity to `query`, highest
        first, with the similarities in the same order.

        The whole call holds the lock: the ranking reads `_hashes` and
        `_matrix` as a pair, and a reload on another thread of the dev server
        must not swap one of them halfway through. Ties are broken by the
        stable sort over hashes loaded in content_hash order, so two identical
        queries give identical orders (contract V6).

        The similarity is a plain matmul, not brain.cosine_similarity_matrix:
        that helper re-normalises and copies the whole bank on every call (two
        77 MB copies per search at 25k images), which is exactly the churn the
        bank exists to avoid. The rows are unit length since _load and the
        query is normalised here, so the dot product is the cosine.
        """
        with self._lock:
            self._refresh_if_changed_locked()
            if not self._hashes:
                return [], np.zeros(0, dtype=np.float32)
            similarities = self._matrix @ _unit(query)
            order = np.argsort(-similarities, kind="stable")
            ranked_hashes = [self._hashes[index] for index in order]
            return ranked_hashes, similarities[order]

    def size(self) -> int:
        return len(self._hashes)

    def reset(self) -> None:
        """Forget the matrix and the version, so the next call reloads. Tests only."""
        with self._lock:
            self._version = None
            self._hashes = []
            self._matrix = np.zeros((0, brain.EMBEDDING_DIM), dtype=np.float32)

    def _refresh_if_changed_locked(self) -> bool:
        version = self.current_version()
        if version == self._version:
            return False
        self._load()
        self._version = version
        return True

    def _load(self) -> None:
        """
        Rows are streamed through iterator() (2000 per fetch, Django's
        default) and ordered by content_hash, so the matrix is built without
        holding 25k model instances and its row order does not depend on
        insertion history. Purged rows are kept: the matrix mirrors the stamp,
        and the caller's candidate set is what excludes them.
        """
        rows = (
            Image.objects.filter(**{self.stamp_field: self.stamp})
            .exclude(**{f"{self.blob_field}__isnull": True})
            .order_by("content_hash")
            .values_list("content_hash", self.blob_field)
            .iterator()
        )
        hashes: list[str] = []
        vectors: list[np.ndarray] = []
        for content_hash, blob in rows:
            hashes.append(content_hash)
            vectors.append(brain.bytes_to_embedding(bytes(blob)))
        if not vectors:
            self._hashes = []
            self._matrix = np.zeros((0, brain.EMBEDDING_DIM), dtype=np.float32)
            return
        matrix = np.stack(vectors).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._hashes = hashes
        self._matrix = matrix / norms


def _unit(vector: np.ndarray) -> np.ndarray:
    """The vector scaled to length 1; a zero vector stays zero, like the rows in _load."""
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(flat)
    if norm == 0:
        return flat
    return flat / norm


def first_allowed(
    ranked_hashes: list[str],
    similarities: np.ndarray,
    allowed: set[str],
    limit: int,
) -> list[tuple[str, float]]:
    """
    Walk a full ranking and keep the first `limit` hashes that are in the
    candidate set. Filtering after ranking (instead of building a per-request
    matrix from the candidates) is what lets the matrix be shared between
    requests with different filters; the walk stops as soon as the page is
    full, so for a broad filter it touches only the head of the ranking.
    """
    picked: list[tuple[str, float]] = []
    for content_hash, similarity in zip(ranked_hashes, similarities, strict=True):
        if content_hash not in allowed:
            continue
        picked.append((content_hash, float(similarity)))
        if len(picked) >= limit:
            break
    return picked
