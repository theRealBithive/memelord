"""Tests for DedupIndex.from_db."""

import os
import uuid

import django
import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.test import TestCase

from core import brain, dedup, phash
from ratings.models import Image


def _create_image(*, is_purged: bool = False, with_signals: bool = True) -> Image:
    """Insert an Image row with the optional dedup signals filled in.

    Signals are stored on every row by the scraper at insert time, so the
    realistic shape is "phash + embedding present" — only is_purged varies.
    """
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.png",
        source_label="t",
        phash=("a1b2c3d4e5f6a7b8" if with_signals else ""),
        embedding=(brain.embedding_to_bytes(np.zeros(768, dtype=np.float32)) if with_signals else None),
        embedding_model=(brain.ENCODER_ID if with_signals else ""),
        is_purged=is_purged,
    )


class DedupIndexFromDbTests(TestCase):
    def setUp(self) -> None:
        # Tests share a DB with the live app; wipe Image so the dedup index
        # we build only contains rows we inserted. TestCase rolls back at end.
        Image.objects.all().delete()

    def test_includes_live_and_purged(self) -> None:
        """from_db loads all rows — both live and purged — so re-scraping a
        purged URL is permanently blocked by its content_hash."""
        live = _create_image()
        purged = _create_image(is_purged=True)

        index = dedup.DedupIndex.from_db()

        self.assertEqual(index.content_hashes, {live.content_hash, purged.content_hash})

    def test_dedup_index_from_db_empty(self) -> None:
        """from_db with no rows returns an empty index shaped for downstream stack/dot ops."""
        index = dedup.DedupIndex.from_db()

        self.assertEqual(index.content_hashes, set())
        self.assertEqual(index.phash_ints, [])
        self.assertEqual(index.embeddings.shape, (0, brain.EMBEDDING_DIM))

    def test_rows_without_signals_do_not_break_embedding_stack(self) -> None:
        """Rows missing phash/embedding (e.g. legacy inserts) are filtered out
        of the per-signal lists so np.vstack doesn't see a None and crash."""
        with_emb = _create_image()
        _create_image(with_signals=False)

        index = dedup.DedupIndex.from_db()

        self.assertIn(with_emb.content_hash, index.content_hashes)
        self.assertEqual(len(index.phash_ints), 1)
        self.assertEqual(index.embeddings.shape, (1, brain.EMBEDDING_DIM))

    def test_stored_fingerprint_is_recognised_again_and_a_distant_one_is_not(self) -> None:
        """The pHash layer must flag the exact fingerprint it stored at scrape
        time and let an unrelated one through — the index is only useful if the
        stored hex survives the round trip through the DB unchanged."""
        _create_image()  # stores phash a1b2c3d4e5f6a7b8

        index = dedup.DedupIndex.from_db()

        self.assertTrue(phash.is_phash_duplicate("a1b2c3d4e5f6a7b8", index.phash_ints, 0))
        self.assertFalse(phash.is_phash_duplicate("0000000000000000", index.phash_ints, 5))

    def test_embedding_bank_stays_float32_after_adding_to_an_empty_index(self) -> None:
        """The bank is float32 by contract (3 KB per row); a float64 start
        matrix would silently double the memory of every index built from an
        empty database as soon as the first row is added."""
        index = dedup.DedupIndex.from_db()

        index.add("h" * 64, "", np.ones(768, dtype=np.float32))

        self.assertEqual(index.embeddings.dtype, np.float32)
        self.assertEqual(index.embeddings.shape, (1, brain.EMBEDDING_DIM))
