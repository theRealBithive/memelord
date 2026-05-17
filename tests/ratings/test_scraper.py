"""Tests for ratings.scraper dedup and NSFW pipeline."""

import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import SimpleTestCase
from PIL import Image as PilImage

from core import brain, dedup
from ratings import scraper
from ratings.models import Image


class ScraperDedupUnitTests(SimpleTestCase):
    databases = []

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmpdir.name)
        (self.data_dir / "inbox").mkdir()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _vision(self) -> scraper.VisionConfig:
        return scraper.VisionConfig(
            phash_max_distance=5,
            dino_dedup_threshold=0.92,
            nsfw_threshold=0.30,
        )

    def test_sha_filter_deletes_exact_duplicate(self) -> None:
        """SHA duplicate removes file and skips insert."""
        path = self.data_dir / "inbox" / f"{uuid.uuid4().hex}.png"
        PilImage.new("RGB", (8, 8), color=(1, 2, 3)).save(path)
        h = dedup.content_hash_for_file(path)
        index = dedup.DedupIndex(content_hashes={h})
        result = scraper._sha_filter([(path, "http://x", "lbl")], index)
        self.assertEqual(result, [])
        self.assertFalse(path.exists())

    @patch("ratings.scraper.Image")
    @patch.object(brain, "encode")
    def test_process_downloads_inserts_new_image(
        self, mock_encode: MagicMock, mock_image_model: MagicMock
    ) -> None:
        """New image passes dedup and is inserted with phash and embedding."""
        mock_image_model.INBOX = Image.INBOX
        emb = np.zeros((1, 768), dtype=np.float32)

        path = self.data_dir / "inbox" / f"{uuid.uuid4().hex}.png"
        PilImage.new("RGB", (16, 16), color=(50, 100, 150)).save(path)
        mock_encode.return_value = (emb, [path])

        index = dedup.DedupIndex()
        n = scraper._process_downloads(
            [(path, "http://example.com/i.png", "test")],
            self.data_dir,
            index,
            MagicMock(),
            MagicMock(),
            self._vision(),
            None,
        )
        self.assertEqual(n, 1)
        mock_image_model.objects.create.assert_called_once()
        kwargs = mock_image_model.objects.create.call_args.kwargs
        self.assertEqual(kwargs["location"], Image.INBOX)
        self.assertFalse(kwargs["is_nsfw"])
        self.assertTrue(kwargs["phash"])
        self.assertIsNotNone(kwargs["embedding"])

    @patch("ratings.scraper.Image")
    @patch.object(brain, "encode")
    def test_process_downloads_does_not_set_is_nsfw_without_classifier(
        self, mock_encode: MagicMock, mock_image_model: MagicMock
    ) -> None:
        """Without NSFW classifier, is_nsfw stays False."""
        path = self.data_dir / "inbox" / f"{uuid.uuid4().hex}.png"
        PilImage.new("RGB", (16, 16)).save(path)
        mock_encode.return_value = (np.zeros((1, 768), dtype=np.float32), [path])
        scraper._process_downloads(
            [(path, "", "b")],
            self.data_dir,
            dedup.DedupIndex(),
            MagicMock(),
            MagicMock(),
            self._vision(),
            None,
        )
        self.assertFalse(mock_image_model.objects.create.call_args.kwargs["is_nsfw"])
