"""Tests for core.trainer."""

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import django
import numpy as np
from django.test import TestCase, override_settings
from PIL import Image

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from core import brain, trainer
from ratings.models import Image as ImageModel


class TrainerTests(TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmpdir.name)
        (self.data_dir / "images").mkdir()
        self.settings_override = override_settings(DATA_DIR=self.data_dir)
        self.settings_override.enable()

    def tearDown(self) -> None:
        self.settings_override.disable()
        self._tmpdir.cleanup()

    def _write_image(self, rel: str) -> Path:
        path = self.data_dir / rel
        Image.new("RGB", (10, 10), color="red").save(path)
        return path

    def test_collect_image_paths_empty(self) -> None:
        """collect_image_paths returns empty when no DB rows."""
        good, bad = trainer.collect_image_paths(self.data_dir)
        self.assertEqual(good, [])
        self.assertEqual(bad, [])

    def test_trashed_image_collected_as_negative(self) -> None:
        """Score 0 (trash) is kept on disk and collected into the negative set."""
        h = uuid.uuid4().hex
        path = f"images/{h}.jpg"
        self._write_image(path)
        ImageModel.objects.create(
            content_hash=h, file_path=path, source_label="t", score=0
        )
        good, bad = trainer.collect_image_paths(self.data_dir)
        self.assertEqual(good, [])
        self.assertIn(str(self.data_dir / path), [str(p) for p in bad])

    def _make_scored(self, score: int) -> str:
        h = uuid.uuid4().hex
        path = f"images/{h}.jpg"
        self._write_image(path)
        ImageModel.objects.create(
            content_hash=h, file_path=path, source_label="t", score=score
        )
        return h

    def test_positive_sample_weights_formula(self) -> None:
        """Score 5-6 → 3.0, score 3-4 → 1.0; negatives are not in the positive map."""
        for score in (3, 4, 5, 6):
            self._make_scored(score)
        for score in (0, 1, 2):
            self._make_scored(score)

        weights = trainer._get_sample_weights(self.data_dir)

        for img in ImageModel.objects.filter(score__gte=3):
            w = weights[str(self.data_dir / img.file_path)]
            expected = 3.0 if img.score >= 5 else 1.0
            self.assertAlmostEqual(w, expected, msg=f"score={img.score}")

        for img in ImageModel.objects.filter(score__lte=2):
            self.assertNotIn(str(self.data_dir / img.file_path), weights)

    def test_negative_sample_weights_formula(self) -> None:
        """Score 0 (trash) → 3.0, score 1-2 → 1.0; positives not in the negative map."""
        for score in (0, 1, 2):
            self._make_scored(score)
        for score in (3, 5):
            self._make_scored(score)

        weights = trainer._get_negative_weights(self.data_dir)

        for img in ImageModel.objects.filter(score__lte=2):
            w = weights[str(self.data_dir / img.file_path)]
            expected = 3.0 if img.score == 0 else 1.0
            self.assertAlmostEqual(w, expected, msg=f"score={img.score}")

        for img in ImageModel.objects.filter(score__gte=3):
            self.assertNotIn(str(self.data_dir / img.file_path), weights)

    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_saves_taste_weights(self, mock_encode, mock_get_encoder) -> None:
        """run() fits taste classifier and saves weights."""
        pos = self._write_image("images/pos.png")
        neg = self._write_image("images/neg.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
        )
        mock_get_encoder.return_value = None
        embeddings = np.array([[0.1] * 768, [0.2] * 768], dtype=np.float32)
        mock_encode.return_value = (embeddings, [pos, neg])

        weights_path = self.data_dir / "weights.pkl"
        trainer.run(data_dir=self.data_dir, weights_path=weights_path)
        self.assertTrue(weights_path.exists())

    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_uses_cached_embeddings_without_encoding(
        self, mock_encode, mock_get_encoder
    ) -> None:
        """With every row's embedding cached, run() skips the DINOv2 encoder."""
        self._write_image("images/pos.png")
        self._write_image("images/neg.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
            embedding=brain.embedding_to_bytes(
                np.full(768, 0.1, dtype=np.float32)
            ),
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
            embedding=brain.embedding_to_bytes(
                np.full(768, 0.2, dtype=np.float32)
            ),
        )

        weights_path = self.data_dir / "weights.pkl"
        trainer.run(data_dir=self.data_dir, weights_path=weights_path)

        self.assertTrue(weights_path.exists())
        mock_encode.assert_not_called()
        mock_get_encoder.assert_not_called()

    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_encodes_only_uncached_rows(
        self, mock_encode, mock_get_encoder
    ) -> None:
        """With a partial cache, run() encodes exactly the rows missing an embedding."""
        self._write_image("images/pos.png")
        neg = self._write_image("images/neg.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
            embedding=brain.embedding_to_bytes(
                np.full(768, 0.1, dtype=np.float32)
            ),
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
        )
        mock_get_encoder.return_value = None
        mock_encode.return_value = (
            np.full((1, 768), 0.2, dtype=np.float32),
            [neg],
        )

        weights_path = self.data_dir / "weights.pkl"
        trainer.run(data_dir=self.data_dir, weights_path=weights_path)

        self.assertTrue(weights_path.exists())
        mock_encode.assert_called_once()
        # encode() must receive only the uncached path, not the whole training set.
        encoded_paths = mock_encode.call_args.args[1]
        self.assertEqual(encoded_paths, [neg])

    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_reencodes_corrupt_cached_embedding(
        self, mock_encode, mock_get_encoder
    ) -> None:
        """A wrong-dimension cached blob must not crash train; it falls through to encode."""
        self._write_image("images/pos.png")
        bad = self._write_image("images/neg.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
            embedding=brain.embedding_to_bytes(
                np.full(768, 0.1, dtype=np.float32)
            ),
        )
        # A 512-d blob bypasses embedding_to_bytes' dimension guard but raises
        # ValueError in bytes_to_embedding — simulates an encoder-dimension swap.
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
            embedding=np.full(512, 0.2, dtype=np.float32).tobytes(),
        )
        mock_get_encoder.return_value = None
        mock_encode.return_value = (
            np.full((1, 768), 0.2, dtype=np.float32),
            [bad],
        )

        weights_path = self.data_dir / "weights.pkl"
        trainer.run(data_dir=self.data_dir, weights_path=weights_path)

        self.assertTrue(weights_path.exists())
        # The corrupt row must be re-encoded, not propagated as a crash.
        mock_encode.assert_called_once()
        encoded_paths = mock_encode.call_args.args[1]
        self.assertEqual(encoded_paths, [bad])

    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_saves_nsfw_weights_when_labels_exist(
        self, mock_encode, mock_get_encoder
    ) -> None:
        """run() saves NSFW classifier when both classes exist."""
        self._write_image("images/pos.png")
        self._write_image("images/neg.png")
        self._write_image("images/safe.png")
        self._write_image("images/nsfw.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/safe.png",
            source_label="t",
            is_nsfw=False,
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/nsfw.png",
            source_label="t",
            is_nsfw=True,
        )
        mock_get_encoder.return_value = None
        all_paths = [
            self.data_dir / "images/pos.png",
            self.data_dir / "images/neg.png",
            self.data_dir / "images/safe.png",
            self.data_dir / "images/nsfw.png",
        ]
        mock_encode.return_value = (np.random.randn(4, 768).astype(np.float32), all_paths)

        taste_path = self.data_dir / "taste.pkl"
        nsfw_path = self.data_dir / "nsfw.pkl"
        trainer.run(
            data_dir=self.data_dir,
            weights_path=taste_path,
            nsfw_weights_path=nsfw_path,
        )
        self.assertTrue(taste_path.exists())
        self.assertTrue(nsfw_path.exists())
