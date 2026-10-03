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
from ratings import views
from ratings.models import Image


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

    def _rated(self, label: str, score: int) -> None:
        h = uuid.uuid4().hex
        Image.objects.create(
            content_hash=h, file_path=f"images/{h}.jpg", source_label=label,
            score=score, rated_at=timezone.now(),
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
            with mock.patch.object(views, "WEIGHTS_PATH", weights):
                response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        rows = response.content.decode().split('<div class="bar-row">')
        tg_row = next(row for row in rows if ">tg<" in row)
        self.assertIn(">2/3<", tg_row)
        self.assertIn('<span class="bar-own">own model</span>', tg_row)
        wsg_row = next(row for row in rows if ">wsg<" in row)
        self.assertIn(">2/1<", wsg_row)
        self.assertIn('<span class="bar-own bar-own--shared">shared</span>', wsg_row)
