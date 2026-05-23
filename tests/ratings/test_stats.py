"""Tests for the stats dashboard."""

import os
import uuid
from datetime import timedelta

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

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
            file_path=f"corpus/{h}.jpg",
            source_label="test",
            location=Image.CORPUS,
            rated_at=now,
        )
        Image.objects.filter(content_hash=h).update(
            downloaded_at=now - timedelta(hours=10)
        )
        bad = uuid.uuid4().hex
        Image.objects.create(
            content_hash=bad,
            file_path=f"corpus/{bad}.jpg",
            source_label="test",
            location=Image.CORPUS,
            rated_at=now - timedelta(hours=1),
        )
        Image.objects.filter(content_hash=bad).update(downloaded_at=now)

        response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ">10h<")
        self.assertNotContains(response, ">5h<")
