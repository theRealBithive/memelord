"""Training session lifecycle: train_status polling and trigger_train enqueue."""

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

from ratings.models import LogEntry


def _completed_task(*, ok: bool = True, error: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        stopped=timezone.now(),
        result={"ok": ok, "error": error},
    )


def _running_task() -> SimpleNamespace:
    return SimpleNamespace(stopped=None, result=None)


@override_settings(DEBUG=True)
class TrainStatusSessionTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        username = f"train_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")
        session = self.client.session
        session["training_task_id"] = "active-task"
        session["training_started_at"] = timezone.now().isoformat()
        session.save()

    @patch("django_q.tasks.fetch")
    def test_completed_poll_clears_session_only_for_matching_task_id(
        self, mock_fetch
    ) -> None:
        mock_fetch.return_value = _completed_task()

        response = self.client.get(reverse("train_status", args=["active-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("training_task_id", self.client.session)
        self.assertIn("train-ok", response.content.decode())

    @patch("django_q.tasks.fetch")
    def test_completed_poll_does_not_clear_session_for_unrelated_task_id(
        self, mock_fetch
    ) -> None:
        mock_fetch.return_value = _completed_task()

        response = self.client.get(reverse("train_status", args=["other-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session["training_task_id"], "active-task")
        self.assertIn("train-ok", response.content.decode())

    @patch("django_q.tasks.fetch")
    def test_missing_task_clears_session_when_ids_match(self, mock_fetch) -> None:
        # Backdate the start time so elapsed > 14400s — the code only treats a
        # None fetch result as "lost" once the cluster timeout has passed.
        session = self.client.session
        session["training_started_at"] = (
            timezone.now() - timedelta(hours=5)
        ).isoformat()
        session.save()
        mock_fetch.return_value = None

        response = self.client.get(reverse("train_status", args=["active-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("training_task_id", self.client.session)
        self.assertIn("train-err", response.content.decode())
        self.assertIn("not found", response.content.decode())

    @patch("django_q.tasks.fetch")
    def test_missing_task_keeps_session_when_ids_differ(self, mock_fetch) -> None:
        mock_fetch.return_value = None

        response = self.client.get(reverse("train_status", args=["other-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session["training_task_id"], "active-task")
        self.assertIn("train-pending", response.content.decode())

    @patch("django_q.tasks.fetch")
    def test_running_task_keeps_session(self, mock_fetch) -> None:
        mock_fetch.return_value = _running_task()

        response = self.client.get(reverse("train_status", args=["active-task"]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session["training_task_id"], "active-task")
        self.assertIn("train-pending", response.content.decode())


def _log(message: str, *, source: str = "train", when=None) -> None:
    """Create a LogEntry, backdating timestamp when given (auto_now_add ignores
    the create() value, so we update afterwards)."""
    entry = LogEntry.objects.create(level="INFO", source=source, message=message)
    if when is not None:
        LogEntry.objects.filter(pk=entry.pk).update(timestamp=when)


@override_settings(DEBUG=True)
class TrainEtaTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        username = f"train_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")
        session = self.client.session
        session["training_task_id"] = "active-task"
        session["training_started_at"] = timezone.now().isoformat()
        session.save()

    @patch("django_q.tasks.fetch", return_value=_running_task())
    def test_running_poll_renders_human_friendly_eta(self, mock_fetch) -> None:
        _log("train: 50/200 encoded (12.3 img/s, ETA 90s)")

        response = self.client.get(reverse("train_status", args=["active-task"]))

        self.assertContains(response, "ETA 1m 30s")

    @patch("django_q.tasks.fetch", return_value=_running_task())
    def test_latest_train_eta_wins(self, mock_fetch) -> None:
        _log("train: 50/200 encoded (12.3 img/s, ETA 90s)")
        _log("train: 150/200 encoded (12.3 img/s, ETA 30s)")

        response = self.client.get(reverse("train_status", args=["active-task"]))

        self.assertContains(response, "ETA 30s")
        self.assertNotContains(response, "ETA 1m 30s")

    @patch("django_q.tasks.fetch", return_value=_running_task())
    def test_stale_pre_run_eta_is_ignored(self, mock_fetch) -> None:
        # An ETA line from a previous run (before training_started_at) must not
        # leak into the current run's display.
        _log(
            "train: 10/200 encoded (12.3 img/s, ETA 600s)",
            when=timezone.now() - timedelta(hours=2),
        )

        response = self.client.get(reverse("train_status", args=["active-task"]))

        self.assertNotContains(response, "ETA")

    @patch("django_q.tasks.fetch", return_value=_running_task())
    def test_classify_eta_does_not_drive_display(self, mock_fetch) -> None:
        # The later classify_images encode pass logs its own ETA lines under a
        # different label; they must not reset the displayed (trainer) ETA.
        _log("train: 200/200 encoded (12.3 img/s, ETA 5s)")
        _log("classify_images: 10/500 encoded (12.3 img/s, ETA 400s)")

        response = self.client.get(reverse("train_status", args=["active-task"]))

        self.assertContains(response, "ETA 5s")
        self.assertNotContains(response, "ETA 6m 40s")


@override_settings(DEBUG=True)
class TriggerTrainSessionTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        username = f"train_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    @patch("django_q.tasks.async_task", return_value="new-task")
    @patch("django_q.tasks.fetch", return_value=None)
    def test_stale_session_allows_fresh_enqueue(self, mock_fetch, mock_async) -> None:
        session = self.client.session
        session["training_task_id"] = "lost-task"
        session["training_started_at"] = timezone.now().isoformat()
        session.save()

        response = self.client.post(reverse("trigger_train"))

        self.assertEqual(response.status_code, 200)
        mock_async.assert_called_once()
        self.assertEqual(self.client.session["training_task_id"], "new-task")

    @patch("django_q.tasks.fetch", return_value=_running_task())
    def test_running_session_returns_pending_without_new_enqueue(
        self, mock_fetch
    ) -> None:
        session = self.client.session
        session["training_task_id"] = "running-task"
        session["training_started_at"] = timezone.now().isoformat()
        session.save()

        with patch("django_q.tasks.async_task") as mock_async:
            response = self.client.post(reverse("trigger_train"))

        self.assertEqual(response.status_code, 200)
        mock_async.assert_not_called()
        self.assertEqual(self.client.session["training_task_id"], "running-task")
        self.assertIn("train-pending", response.content.decode())


@override_settings(DEBUG=True)
class TrainingCtxStaleTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        username = f"train_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    @patch("django_q.tasks.fetch", return_value=None)
    def test_training_ctx_clears_lost_task_on_page_load(self, mock_fetch) -> None:
        # Backdate so elapsed > 14400s — _training_task_stale only returns True
        # for a None fetch result once the cluster timeout has elapsed.
        session = self.client.session
        session["training_task_id"] = "lost-task"
        session["training_started_at"] = (
            timezone.now() - timedelta(hours=5)
        ).isoformat()
        session.save()

        response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("training_task_id", self.client.session)
        self.assertNotContains(response, 'class="nav-training"')

    @patch("django_q.tasks.fetch", return_value=_completed_task())
    def test_training_ctx_clears_finished_task_on_page_load(self, mock_fetch) -> None:
        session = self.client.session
        session["training_task_id"] = "done-task"
        session["training_started_at"] = timezone.now().isoformat()
        session.save()

        response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("training_task_id", self.client.session)

    @patch("django_q.tasks.fetch", return_value=_running_task())
    def test_training_ctx_keeps_active_running_task(self, mock_fetch) -> None:
        session = self.client.session
        session["training_task_id"] = "running-task"
        session["training_started_at"] = timezone.now().isoformat()
        session.save()

        response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session["training_task_id"], "running-task")
        self.assertContains(response, 'class="nav-training"')

    @patch("django_q.tasks.fetch", return_value=_running_task())
    def test_training_ctx_clears_task_past_timeout(self, mock_fetch) -> None:
        session = self.client.session
        session["training_task_id"] = "running-task"
        session["training_started_at"] = (
            timezone.now() - timedelta(hours=5)
        ).isoformat()
        session.save()

        response = self.client.get(reverse("stats"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("training_task_id", self.client.session)
