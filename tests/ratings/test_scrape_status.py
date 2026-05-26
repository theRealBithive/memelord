"""Scrape session lifecycle: scrape_status polling and trigger_scrape enqueue.

Mirrors test_train_status.py — scraping moved from a synchronous request to a
background django-q task because full-thread 4chan scraping now runs for minutes
and would exceed gunicorn's request timeout.
"""

import os
import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone


def _completed_task(*, success: bool = True, result=None) -> SimpleNamespace:
    return SimpleNamespace(
        stopped=timezone.now(),
        success=success,
        result={"ok": True, "total": 3, "counts": {"4chan/wg": 3}}
        if result is None
        else result,
    )


def _running_task() -> SimpleNamespace:
    return SimpleNamespace(stopped=None, success=False, result=None)


@override_settings(DEBUG=True)
class ScrapeStatusSessionTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        username = f"scrape_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")
        session = self.client.session
        session["scrape_task_id"] = "active-task"
        session["scrape_started_at"] = timezone.now().isoformat()
        session.save()

    @patch("django_q.tasks.fetch")
    def test_completed_poll_clears_session_and_shows_counts(self, mock_fetch) -> None:
        mock_fetch.return_value = _completed_task()

        response = self.client.get(reverse("scrape_status", args=["active-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("scrape_task_id", self.client.session)
        body = response.content.decode()
        self.assertIn("train-ok", body)
        self.assertIn("3 new image", body)

    @patch("django_q.tasks.fetch")
    def test_completed_poll_does_not_clear_session_for_unrelated_task_id(
        self, mock_fetch
    ) -> None:
        mock_fetch.return_value = _completed_task()

        response = self.client.get(reverse("scrape_status", args=["other-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session["scrape_task_id"], "active-task")
        self.assertIn("train-ok", response.content.decode())

    @patch("django_q.tasks.fetch")
    def test_failed_task_renders_error(self, mock_fetch) -> None:
        mock_fetch.return_value = _completed_task(
            result={"ok": False, "error": "boom"}
        )

        response = self.client.get(reverse("scrape_status", args=["active-task"]))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn("train-err", body)
        self.assertIn("boom", body)

    @patch("django_q.tasks.fetch")
    def test_worker_crash_without_result_is_failure(self, mock_fetch) -> None:
        # django-q marks a hard worker crash as success=False with no dict result;
        # ok must be False even though result.get("ok", True) would default True.
        mock_fetch.return_value = SimpleNamespace(
            stopped=timezone.now(), success=False, result=None
        )

        response = self.client.get(reverse("scrape_status", args=["active-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertIn("train-err", response.content.decode())

    @patch("django_q.tasks.fetch")
    def test_missing_task_clears_session_when_ids_match(self, mock_fetch) -> None:
        # Backdate so elapsed > 14400s — a None fetch result is only "lost" once
        # the cluster timeout has passed.
        session = self.client.session
        session["scrape_started_at"] = (
            timezone.now() - timedelta(hours=5)
        ).isoformat()
        session.save()
        mock_fetch.return_value = None

        response = self.client.get(reverse("scrape_status", args=["active-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("scrape_task_id", self.client.session)
        self.assertIn("train-err", response.content.decode())
        self.assertIn("not found", response.content.decode())

    @patch("django_q.tasks.fetch")
    def test_missing_task_keeps_session_when_ids_differ(self, mock_fetch) -> None:
        mock_fetch.return_value = None

        response = self.client.get(reverse("scrape_status", args=["other-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session["scrape_task_id"], "active-task")
        self.assertIn("train-pending", response.content.decode())

    @patch("django_q.tasks.fetch")
    def test_running_task_keeps_session(self, mock_fetch) -> None:
        mock_fetch.return_value = _running_task()

        response = self.client.get(reverse("scrape_status", args=["active-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session["scrape_task_id"], "active-task")
        self.assertIn("train-pending", response.content.decode())


@override_settings(DEBUG=True)
class TriggerScrapeSessionTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        username = f"scrape_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    @patch("django_q.tasks.async_task", return_value="new-task")
    @patch("django_q.tasks.fetch", return_value=None)
    def test_enqueues_async_task_and_stores_session(
        self, mock_fetch, mock_async
    ) -> None:
        response = self.client.post(reverse("trigger_scrape"))

        self.assertEqual(response.status_code, 200)
        mock_async.assert_called_once_with("ratings.tasks.run_scrape")
        self.assertEqual(self.client.session["scrape_task_id"], "new-task")
        self.assertIn("train-pending", response.content.decode())

    @patch("django_q.tasks.async_task", return_value="new-task")
    @patch("django_q.tasks.fetch", return_value=None)
    def test_stale_session_allows_fresh_enqueue(self, mock_fetch, mock_async) -> None:
        session = self.client.session
        session["scrape_task_id"] = "lost-task"
        session["scrape_started_at"] = timezone.now().isoformat()
        session.save()

        response = self.client.post(reverse("trigger_scrape"))

        self.assertEqual(response.status_code, 200)
        mock_async.assert_called_once()
        self.assertEqual(self.client.session["scrape_task_id"], "new-task")

    @patch("django_q.tasks.fetch", return_value=_running_task())
    def test_running_session_returns_pending_without_new_enqueue(
        self, mock_fetch
    ) -> None:
        session = self.client.session
        session["scrape_task_id"] = "running-task"
        session["scrape_started_at"] = timezone.now().isoformat()
        session.save()

        with patch("django_q.tasks.async_task") as mock_async:
            response = self.client.post(reverse("trigger_scrape"))

        self.assertEqual(response.status_code, 200)
        mock_async.assert_not_called()
        self.assertEqual(self.client.session["scrape_task_id"], "running-task")
        self.assertIn("train-pending", response.content.decode())


@override_settings(DEBUG=True)
class ScrapeCtxStaleTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        username = f"scrape_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    @patch("django_q.tasks.fetch", return_value=_running_task())
    def test_scrape_ctx_resumes_poller_on_page_load(self, mock_fetch) -> None:
        session = self.client.session
        session["scrape_task_id"] = "running-task"
        session["scrape_started_at"] = timezone.now().isoformat()
        session.save()

        response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session["scrape_task_id"], "running-task")
        self.assertContains(response, "train-pending")

    @patch("django_q.tasks.fetch", return_value=None)
    def test_scrape_ctx_clears_lost_task_on_page_load(self, mock_fetch) -> None:
        session = self.client.session
        session["scrape_task_id"] = "lost-task"
        session["scrape_started_at"] = (
            timezone.now() - timedelta(hours=5)
        ).isoformat()
        session.save()

        response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("scrape_task_id", self.client.session)
