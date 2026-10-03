"""Tests for the stats dashboard."""

import os
import uuid
from datetime import timedelta

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from sklearn.linear_model import LogisticRegression

from core import taste
from ratings.models import Image
from ratings.views import common


@override_settings(DEBUG=True)
class StatsAvgInboxHoursTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self.client = Client()
        username = f"stats_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    def test_avg_inbox_hours_ignores_invalid_duration_pairs(self) -> None:
        now = timezone.now()
        h = uuid.uuid4().hex
        Image.objects.create(
            content_hash=h,
            file_path=f"images/{h}.jpg",
            source_label="test",
            score=4,
            rated_at=now,
        )
        Image.objects.filter(content_hash=h).update(
            downloaded_at=now - timedelta(hours=10)
        )
        bad = uuid.uuid4().hex
        Image.objects.create(
            content_hash=bad,
            file_path=f"images/{bad}.jpg",
            source_label="test",
            score=4,
            rated_at=now - timedelta(hours=1),
        )
        Image.objects.filter(content_hash=bad).update(downloaded_at=now)

        response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ">10h<")
        self.assertNotContains(response, ">5h<")

    def _rated(self, label: str, score: int, *, is_nsfw: bool = False) -> None:
        h = uuid.uuid4().hex
        Image.objects.create(
            content_hash=h, file_path=f"images/{h}.jpg", source_label=label,
            score=score, rated_at=timezone.now(), is_nsfw=is_nsfw,
        )

    def test_rated_by_source_shows_good_bad_counts_and_which_model_judges(self) -> None:
        """Taste contract: V10"""
        for score in (3, 6, 0, 1, 2):
            self._rated("tg", score)
        for score in (4, 4, 1):
            self._rated("wsg", score)
        X = np.vstack([np.full(768, 1.0), np.full(768, -1.0)])
        clf = LogisticRegression().fit(X, [1, 0])
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "w.pkl"
            taste.save_taste_model(taste.TasteModel(shared=clf, per_source={"tg": clf}), weights)
            with mock.patch.object(common, "WEIGHTS_PATH", weights):
                response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        rows = response.content.decode().split('<div class="bar-row">')
        tg_row = next(row for row in rows if ">tg<" in row)
        self.assertIn(">2/3<", tg_row)
        self.assertIn('<span class="bar-own">own model</span>', tg_row)
        wsg_row = next(row for row in rows if ">wsg<" in row)
        self.assertIn(">2/1<", wsg_row)
        self.assertIn('<span class="bar-own bar-own--shared">shared</span>', wsg_row)

    def test_flagged_images_form_the_nsfw_row_and_leave_their_sources(self) -> None:
        """Taste contract: V27 — '(nsfw)' is a row of its own, shown only with NSFW on; the source rows count safe images only."""
        for score in (3, 4, 5, 1, 2):
            self._rated("tg", score)
        for label, score in (("tg", 6), ("tg", 0), ("wsg", 4)):
            self._rated(label, score, is_nsfw=True)
        X = np.vstack([np.full(768, 1.0), np.full(768, -1.0)])
        clf = LogisticRegression().fit(X, [1, 0])
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "w.pkl"
            taste.save_taste_model(
                taste.TasteModel(shared=clf, per_source={taste.NSFW_GROUP: clf}), weights
            )
            with mock.patch.object(common, "WEIGHTS_PATH", weights):
                hidden = self.client.get(reverse("stats")).content.decode()
                self.client.post(reverse("nsfw_toggle"))
                shown = self.client.get(reverse("stats")).content.decode()

        self.assertNotIn(">(nsfw)<", hidden)
        hidden_tg = next(row for row in hidden.split('<div class="bar-row">') if ">tg<" in row)
        self.assertIn(">3/2<", hidden_tg)
        rows = shown.split('<div class="bar-row">')
        nsfw_row = next(row for row in rows if ">(nsfw)<" in row)
        self.assertIn(">2/1<", nsfw_row)
        self.assertIn('<span class="bar-own">own model</span>', nsfw_row)
        shown_tg = next(row for row in rows if ">tg<" in row)
        self.assertIn(">3/2<", shown_tg, "the flagged tg rows count in the NSFW row, not here")
        self.assertIn('<span class="bar-own bar-own--shared">shared</span>', shown_tg)
        self.assertFalse(any(">wsg<" in row for row in rows), "wsg has no safe rated image, so no row")


class WeightsLastModifiedTests(TestCase):
    def test_missing_file_means_never_trained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(common, "WEIGHTS_PATH", Path(tmp) / "none.pkl"):
            self.assertIsNone(common.weights_last_modified())

    def test_existing_file_reports_its_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "w.pkl"
            weights.write_bytes(b"x")
            expected_mtime = weights.stat().st_mtime
            with mock.patch.object(common, "WEIGHTS_PATH", weights):
                stamp = common.weights_last_modified()
        self.assertIsNotNone(stamp)
        self.assertAlmostEqual(stamp.timestamp(), expected_mtime, places=3)
