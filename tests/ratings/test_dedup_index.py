"""Django DB tests for DedupIndex.from_db."""

import os
import tempfile
import uuid
from pathlib import Path

import django
import numpy as np
from django.test import TestCase, override_settings
from PIL import Image as PilImage

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from core import brain, dedup
from ratings.models import Image


class DedupIndexFromDbTests(TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmpdir.name)
        (self.data_dir / "inbox").mkdir()
        test_db = {
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": str(self.data_dir / "test.db"),
            }
        }
        self._settings = override_settings(DATA_DIR=self.data_dir, DATABASES=test_db)
        self._settings.enable()

    def tearDown(self) -> None:
        self._settings.disable()
        self._tmpdir.cleanup()

    def test_dedup_index_from_db(self) -> None:
        """DedupIndex.from_db loads hashes and phashes from Image rows."""
        emb = np.zeros(768, dtype=np.float32)
        Image.objects.create(
            content_hash="h1",
            file_path="inbox/h1.jpg",
            source_label="t",
            phash="0123456789abcdef",
            embedding=brain.embedding_to_bytes(emb),
        )
        index = dedup.DedupIndex.from_db()
        self.assertIn("h1", index.content_hashes)
        self.assertIn("0123456789abcdef", index.phashes)
        self.assertEqual(index.embeddings.shape, (1, 768))
