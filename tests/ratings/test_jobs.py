"""
Background jobs in the UI (UI overhaul).

Contract: V8 Scrape und Train sind aus dem Leerzustand der Review-Queue und aus
             Config erreichbar. Ein laufender Job ist auf jeder Seite in der
             Navigation sichtbar.
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from unittest import mock

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from ratings.models import Image

PAGES = ("review_corpus", "gallery", "below_cutoff", "stats", "logs", "config", "tag_list")
RUNNING_TASK = SimpleNamespace(stopped=None, result=None)


@override_settings(DEBUG=True)
class JobsInTheUiTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        user = get_user_model().objects.create_user(f"jobs-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def _start_training_in_session(self) -> None:
        session = self.client.session
        session["training_task_id"] = "active-task"
        session["training_started_at"] = timezone.now().isoformat()
        session.save()

    @mock.patch("django_q.tasks.fetch", return_value=RUNNING_TASK)
    def test_running_job_is_visible_in_the_nav_on_every_page(self, _fetch) -> None:
        """Contract: V8"""
        self._start_training_in_session()
        for url_name in PAGES:
            html = self.client.get(reverse(url_name)).content.decode()
            self.assertIn('id="nav-job" class="nav-job"', html, url_name)
            self.assertIn("Training", html, url_name)
            self.assertIn(reverse("job_indicator"), html, url_name)

    def test_idle_indicator_is_hidden_and_does_not_poll(self) -> None:
        """Contract: V8"""
        html = self.client.get(reverse("job_indicator")).content.decode()
        self.assertIn("nav-job--idle", html)
        self.assertNotIn("hx-get", html)

    @mock.patch("django_q.tasks.fetch", return_value=RUNNING_TASK)
    def test_active_indicator_polls_itself(self, _fetch) -> None:
        """Contract: V8"""
        self._start_training_in_session()
        html = self.client.get(reverse("job_indicator")).content.decode()
        self.assertIn('hx-trigger="every 5s"', html)
        self.assertNotIn("nav-job--idle", html)

    def test_empty_queue_and_config_offer_scrape_and_train(self) -> None:
        """Contract: V8"""
        for url_name, container in (("review_corpus", 'id="done-job"'), ("config", 'id="scrape-result"')):
            html = self.client.get(reverse(url_name)).content.decode()
            self.assertIn(f'hx-post="{reverse("trigger_scrape")}"', html, url_name)
            self.assertIn(f'hx-post="{reverse("trigger_train")}"', html, url_name)
            self.assertIn(container, html, url_name)

    @mock.patch("django_q.tasks.async_task", return_value="new-task")
    def test_starting_a_job_updates_the_nav_indicator_out_of_band(self, _async) -> None:
        """Contract: V8 (the page that started the job sees the indicator at once)"""
        html = self.client.post(reverse("trigger_train")).content.decode()
        self.assertIn('id="nav-job" class="nav-job"', html)
        self.assertIn('hx-swap-oob="true"', html)
        self.assertIn("Training", html)
