"""Tests for the below-cutoff view (images with score <= 2)."""

import os
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from ratings.models import Image


def _image(score=None):
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.jpg",
        source_label="test",
        score=score,
    )


@override_settings(DEBUG=True)
class BelowCutoffViewTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self.client = Client()
        username = f"bc_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    def test_shows_only_low_scored_images(self) -> None:
        low = _image(score=1)
        _image(score=2)
        gallery = _image(score=3)
        unrated = _image(score=None)

        response = self.client.get(reverse("below_cutoff"))

        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn(low.content_hash, content)
        self.assertNotIn(gallery.content_hash, content)
        self.assertNotIn(unrated.content_hash, content)

    def test_excludes_purged_images(self) -> None:
        h = uuid.uuid4().hex
        Image.objects.create(
            content_hash=h,
            file_path=f"images/{h}.jpg",
            source_label="test",
            score=1,
            is_purged=True,
        )
        response = self.client.get(reverse("below_cutoff"))
        self.assertNotIn(h, response.content.decode())

    def test_anonymous_redirects_to_login(self) -> None:
        response = Client().get(reverse("below_cutoff"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)
