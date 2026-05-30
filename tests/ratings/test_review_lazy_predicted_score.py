"""Repro: rating stalls when a lazily-computed predicted_score drops the
just-shown image below the vision cutoff mid-session."""

import os
import uuid
from datetime import timedelta

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from unittest import mock

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from ratings.models import Image, ReviewThresholds


@override_settings(DEBUG=True)
class ReproStall(TestCase):
    def setUp(self):
        Image.objects.all().delete()
        # Vision dial at bucket 3 → cutoff = (3-1)/6 ≈ 0.333.
        ReviewThresholds.objects.update_or_create(
            pk=1, defaults={"sfw_threshold": 3, "nsfw_threshold": 3}
        )
        self.client = Client()
        u = f"r_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(u, password="x")
        self.client.login(username=u, password="x")

    def _img(self, age_hours, predicted=None, embedding=None):
        h = uuid.uuid4().hex
        Image.objects.create(
            content_hash=h,
            file_path=f"images/{h}.jpg",
            source_label="t",
            predicted_score=predicted,
            embedding=embedding,
        )
        Image.objects.filter(content_hash=h).update(
            downloaded_at=timezone.now() - timedelta(hours=age_hours)
        )
        return h

    def test_lazy_predicted_score_below_cutoff_does_not_stall(self):
        # img1: already above cutoff. img2: predicted_score NULL but has an
        # embedding, so rendering it lazily computes a *below-cutoff* score.
        # img3: a clean above-cutoff image waiting behind it.
        # Rate-and-advance picks the newest-downloaded unseen image next, so
        # img2 must be the newest for it to be the card shown after img1.
        img1 = self._img(age_hours=3, predicted=0.9)
        img2 = self._img(age_hours=1, predicted=None, embedding=b"\x00" * 4)
        img3 = self._img(age_hours=2, predicted=0.9)

        # Stub the classifier so rendering img2 computes 0.1 (below 0.333) and
        # _taste_prediction persists it.
        with mock.patch("ratings.views._get_taste_clf", return_value=object()), \
             mock.patch("core.brain.bytes_to_embedding", return_value=None), \
             mock.patch("core.brain.predict_proba", return_value=0.1):
            # Land on the page, then rate img1 → advances to img2 (renders +
            # persists img2.predicted_score = 0.1).
            self.client.get(reverse("review_corpus"))
            r1 = self.client.post(
                reverse("score_corpus", args=[img1]), {"score": "4"}
            ).content.decode()
            self.assertIn(img2, r1)

            # img2 has now silently dropped below the cutoff.
            self.assertAlmostEqual(
                Image.objects.get(content_hash=img2).predicted_score, 0.1
            )

            # Rate img2 → must advance to img3, NOT re-show img2.
            r2 = self.client.post(
                reverse("score_corpus", args=[img2]), {"score": "5"}
            ).content.decode()

        self.assertEqual(Image.objects.get(content_hash=img2).score, 5)
        self.assertIn(img3, r2, "did not advance to the next image")
        self.assertNotIn(img2, r2, "re-displayed the just-rated image (stall bug)")
