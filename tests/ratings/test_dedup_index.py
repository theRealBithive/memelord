"""Tests for DedupIndex.from_db."""

import os
import unittest
from unittest.mock import patch

import django
import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from core import brain, dedup


class DedupIndexFromDbTests(unittest.TestCase):
    def test_dedup_index_from_db(self) -> None:
        """DedupIndex.from_db builds index from DB row tuples."""
        emb = np.zeros(768, dtype=np.float32)
        emb_bytes = brain.embedding_to_bytes(emb)

        with patch("ratings.models.Image") as MockImage:
            MockImage.objects.filter.return_value.values_list.return_value = [
                ("h1", "0123456789abcdef", emb_bytes),
            ]
            index = dedup.DedupIndex.from_db()

        self.assertIn("h1", index.content_hashes)
        self.assertIn("0123456789abcdef", index.phashes)
        self.assertEqual(index.embeddings.shape, (1, 768))

    def test_dedup_index_from_db_empty(self) -> None:
        """from_db with no rows returns an empty index."""
        with patch("ratings.models.Image") as MockImage:
            MockImage.objects.filter.return_value.values_list.return_value = []
            index = dedup.DedupIndex.from_db()

        self.assertEqual(len(index.content_hashes), 0)
        self.assertEqual(len(index.phashes), 0)
        self.assertEqual(index.embeddings.shape, (0, brain.EMBEDDING_DIM))
