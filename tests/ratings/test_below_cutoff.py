"""Tests for the below-cutoff view and queue filter."""

import os
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from ratings.models import Image, ReviewThresholds
from ratings.queue_rules import below_cutoff_q


def _image(*, score=None, predicted=None, is_nsfw=False):
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.jpg",
        source_label="test",
        score=score,
        predicted_score=predicted,
        is_nsfw=is_nsfw,
    )


@override_settings(DEBUG=True)
class BelowCutoffViewTests(TestCase):
    def setUp(self) -> None:
        ReviewThresholds.objects.all().delete()
        Image.objects.all().delete()
        self.client = Client()
        username = f"bc_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    def _set_thresholds(self, sfw: int, nsfw: int = 1) -> None:
        ReviewThresholds.objects.update_or_create(
            pk=1, defaults={"sfw_threshold": sfw, "nsfw_threshold": nsfw}
        )

    def test_shows_user_low_scored_images(self) -> None:
        low = _image(score=1)
        _image(score=2)
        gallery = _image(score=3)
        unrated_high = _image(score=None, predicted=0.9)

        response = self.client.get(reverse("below_cutoff"))

        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn(low.content_hash, content)
        self.assertNotIn(gallery.content_hash, content)
        self.assertNotIn(unrated_high.content_hash, content)

    def test_shows_unrated_model_rejects(self) -> None:
        self._set_thresholds(sfw=4, nsfw=1)  # SFW cutoff 0.5
        reject = _image(score=None, predicted=0.2)
        review = _image(score=None, predicted=0.8)

        content = self.client.get(reverse("below_cutoff")).content.decode()

        self.assertIn(reject.content_hash, content)
        self.assertNotIn(review.content_hash, content)

    def test_unrated_model_reject_shows_unrated_label_and_prediction(self) -> None:
        self._set_thresholds(sfw=4, nsfw=1)
        reject = _image(score=None, predicted=0.2)

        content = self.client.get(reverse("below_cutoff")).content.decode()

        self.assertIn("gallery-item-score--unrated", content)
        self.assertIn(">unrated<", content)
        self.assertIn(f'data-predicted="{reject.predicted_score}"', content)

    def test_unrated_null_predicted_not_in_below(self) -> None:
        """Unclassified images belong in review only until classify runs."""
        self._set_thresholds(sfw=6, nsfw=1)
        fresh = _image(score=None, predicted=None)

        content = self.client.get(reverse("below_cutoff")).content.decode()

        self.assertNotIn(fresh.content_hash, content)

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


@override_settings(SFW_THRESHOLD_BUCKET=1, NSFW_THRESHOLD_BUCKET=1)
class BelowCutoffQueueRulesTests(TestCase):
    def setUp(self) -> None:
        ReviewThresholds.objects.all().delete()
        Image.objects.all().delete()

    def _set_thresholds(self, sfw: int, nsfw: int) -> None:
        ReviewThresholds.objects.update_or_create(
            pk=1, defaults={"sfw_threshold": sfw, "nsfw_threshold": nsfw}
        )

    def test_review_hidden_image_in_below_q(self) -> None:
        from ratings.views import _review_qs

        reject = _image(score=None, predicted=0.1)
        self._set_thresholds(sfw=6, nsfw=1)

        review_hashes = set(
            _review_qs(show_nsfw=False).values_list("content_hash", flat=True)
        )
        below_filter = below_cutoff_q(6, 1, show_nsfw=False)
        below_hashes = set(
            Image.objects.filter(below_filter).values_list("content_hash", flat=True)
        )

        self.assertNotIn(reject.content_hash, review_hashes)
        self.assertIn(reject.content_hash, below_hashes)
