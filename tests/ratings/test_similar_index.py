"""Tests for the view-level similar-image kNN cache."""

import os
import uuid

import django
import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.test import TestCase

from core import brain
from ratings import views
from ratings.models import Image


def _create_rated_image(*, embedding: bytes | None) -> Image:
    """Insert a rated Image row with the given embedding blob (may be corrupt).

    `score` must be non-null because _get_similar_index filters on rated rows.
    The embedding blob is passed through verbatim so callers can simulate the
    truncated/wrong-size writes the size-guard is meant to defend against.
    """
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.png",
        source_label="t",
        score=3,
        embedding=embedding,
    )


class GetSimilarIndexCorruptBlobTests(TestCase):
    """LOG-4: one corrupt embedding blob must not 500 the review/gallery views."""

    def setUp(self) -> None:
        # The test DB is shared with the dev app; clear rows and the module-scope
        # TTL cache so we observe a fresh rebuild and only see rows we inserted.
        Image.objects.all().delete()
        views._invalidate_similar_index()

    def test_corrupt_blob_is_skipped_not_raised(self) -> None:
        # One healthy 768-d embedding plus one truncated blob. Before the fix
        # np.stack would raise ValueError on the mismatched shapes and every
        # caller of _get_similar_rated would 500.
        good = brain.embedding_to_bytes(np.ones(768, dtype=np.float32))
        good_img = _create_rated_image(embedding=good)
        _create_rated_image(embedding=b"\x00" * 100)  # truncated

        idx = views._get_similar_index()

        assert idx is not None
        assert idx["hashes"] == [good_img.content_hash]
        assert idx["embeddings"].shape == (1, 768)

    def test_all_corrupt_yields_none(self) -> None:
        # If every row is corrupt the index is empty — return None rather than
        # an empty matrix so _get_similar_rated short-circuits cleanly.
        _create_rated_image(embedding=b"\x00" * 100)
        _create_rated_image(embedding=b"\x01" * 2048)

        assert views._get_similar_index() is None
