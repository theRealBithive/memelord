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

_TEST_DB = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}


@override_settings(DATABASES=_TEST_DB)
class TrainerTests(TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmpdir.name)
        for loc in ("corpus", "void", "inbox"):
            (self.data_dir / loc).mkdir()
        self.settings_override = override_settings(
            DATA_DIR=self.data_dir, DATABASES=_TEST_DB
        )
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
        corpus, void = trainer.collect_image_paths(self.data_dir)
        self.assertEqual(corpus, [])
        self.assertEqual(void, [])

    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_saves_taste_weights(self, mock_encode, mock_get_encoder) -> None:
        """run() fits taste classifier and saves weights."""
        pos = self._write_image("corpus/pos.png")
        neg = self._write_image("void/neg.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="corpus/pos.png",
            source_label="t",
            location=ImageModel.CORPUS,
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="void/neg.png",
            source_label="t",
            location=ImageModel.VOID,
        )
        mock_get_encoder.return_value = None
        mock_encode.return_value = np.array(
            [[0.1] * 768, [0.2] * 768], dtype=np.float32
        )

        weights_path = self.data_dir / "weights.pkl"
        trainer.run(data_dir=self.data_dir, weights_path=weights_path)
        self.assertTrue(weights_path.exists())

    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_saves_nsfw_weights_when_labels_exist(
        self, mock_encode, mock_get_encoder
    ) -> None:
        """run() saves NSFW classifier when both classes exist."""
        self._write_image("corpus/pos.png")
        self._write_image("void/neg.png")
        self._write_image("inbox/safe.png")
        self._write_image("inbox/nsfw.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="corpus/pos.png",
            source_label="t",
            location=ImageModel.CORPUS,
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="void/neg.png",
            source_label="t",
            location=ImageModel.VOID,
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="inbox/safe.png",
            source_label="t",
            location=ImageModel.INBOX,
            is_nsfw=False,
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="inbox/nsfw.png",
            source_label="t",
            location=ImageModel.INBOX,
            is_nsfw=True,
        )
        mock_get_encoder.return_value = None
        mock_encode.return_value = np.random.randn(4, 768).astype(np.float32)

        taste_path = self.data_dir / "taste.pkl"
        nsfw_path = self.data_dir / "nsfw.pkl"
        trainer.run(
            data_dir=self.data_dir,
            weights_path=taste_path,
            nsfw_weights_path=nsfw_path,
        )
        self.assertTrue(taste_path.exists())
        self.assertTrue(nsfw_path.exists())
