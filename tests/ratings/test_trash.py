"""Trash (void) moves from the corpus review queue."""

import os
import tempfile
import uuid
from pathlib import Path

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from ratings.models import Image


@override_settings(DEBUG=True)
class TrashCorpusTests(TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmpdir.name)
        (self.data_dir / "inbox").mkdir()
        self.addCleanup(self._tmpdir.cleanup)

        Image.objects.all().delete()
        self.client = Client()
        username = f"trash_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

        self.content_hash = uuid.uuid4().hex
        self.rel_inbox = f"inbox/{self.content_hash}.jpg"
        (self.data_dir / self.rel_inbox).write_bytes(b"\xff\xd8\xff")
        Image.objects.create(
            content_hash=self.content_hash,
            file_path=self.rel_inbox,
            source_label="test",
            location=Image.INBOX,
        )

    def test_trash_corpus_moves_file_to_void_directory(self) -> None:
        with override_settings(DATA_DIR=self.data_dir):
            url = reverse("trash_corpus", args=[self.content_hash])
            response = self.client.post(url, HTTP_HX_REQUEST="true")

        self.assertEqual(response.status_code, 200)
        image = Image.objects.get(content_hash=self.content_hash)
        self.assertEqual(image.location, Image.VOID)
        self.assertTrue(image.file_path.startswith("void/"))
        self.assertFalse(image.file_deleted)
        self.assertTrue((self.data_dir / image.file_path).is_file())
        self.assertFalse((self.data_dir / self.rel_inbox).exists())

    def test_trash_corpus_skips_void_when_source_file_missing(self) -> None:
        (self.data_dir / self.rel_inbox).unlink()
        with override_settings(DATA_DIR=self.data_dir):
            url = reverse("trash_corpus", args=[self.content_hash])
            response = self.client.post(url, HTTP_HX_REQUEST="true")

        self.assertEqual(response.status_code, 200)
        image = Image.objects.get(content_hash=self.content_hash)
        self.assertEqual(image.location, Image.INBOX)
        self.assertTrue(image.file_deleted)
