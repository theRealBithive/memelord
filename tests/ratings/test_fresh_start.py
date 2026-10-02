"""
Fresh start (UI overhaul).

Contract: V12 Neustart („alles weg, Konfiguration bleibt“): löscht alle Bilder
              (Zeilen und Dateien, thumbs), Scores, Embeddings, Vorhersagen,
              Bild-Tag-Zuordnungen, Klassifikator-Gewichte (Taste + NSFW), setzt
              Scrape-Cursor der Quellen zurück; behält Quellen, Kanäle,
              Review-Schwellen, Zeitplan, Benutzer, Tag-Vokabular, Logs; verlangt
              getipptes Bestätigungswort im Sheet; gesperrt während Scrape/Train
              läuft; schreibt Log-Eintrag; danach Queue leer, Stats null, Badges
              null; CLI `manage.py fresh_start`.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from django_q.models import OrmQ
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.django import TestCase as HypothesisTestCase

from ratings.models import (
    Image,
    LogEntry,
    NotificationChannel,
    ReviewThresholds,
    Source,
    Tag,
)
from ratings.reset import CONFIRM_WORD, fresh_start

ROW = st.fixed_dictionaries(
    {
        "score": st.sampled_from([None, 0, 1, 2, 3, 4, 5, 6]),
        "nsfw": st.booleans(),
        "purged": st.booleans(),
        "embedding": st.booleans(),
        "tagged": st.booleans(),
        "file": st.booleans(),
        "thumb": st.booleans(),
    }
)
SOURCE = st.fixed_dictionaries({"cursor": st.one_of(st.none(), st.text(min_size=1, max_size=10))})


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _create_library(data_dir: Path, rows: list[dict], tag: Tag) -> int:
    """Insert the rows and write their files; returns how many files were written."""
    files = 0
    for spec in rows:
        h = uuid.uuid4().hex
        image = Image.objects.create(
            content_hash=h,
            file_path=f"images/{h}.png",
            source_label="t",
            score=spec["score"],
            is_nsfw=spec["nsfw"],
            is_purged=spec["purged"],
            embedding=b"\x00" * 3072 if spec["embedding"] else None,
            embedding_model="dinov3_vitb16" if spec["embedding"] else "",
            predicted_score=0.5 if spec["embedding"] else None,
        )
        if spec["tagged"]:
            image.tags.add(tag)
        if spec["file"]:
            path = data_dir / "images" / f"{h}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
            files += 1
        if spec["thumb"]:
            path = data_dir / "thumbs" / "images" / f"{h}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"jpg")
            files += 1
    return files


def _files_below(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*") if p.is_file()]


class FreshStartProperty(HypothesisTestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        Source.objects.all().delete()
        LogEntry.objects.all().delete()

    @settings(max_examples=25, deadline=None)
    @given(
        rows=st.lists(ROW, max_size=6),
        sources=st.lists(SOURCE, max_size=3),
        weights_present=st.tuples(st.booleans(), st.booleans()),
    )
    def test_everything_about_images_goes_and_the_configuration_stays(
        self, rows, sources, weights_present
    ) -> None:
        """Contract: V12 (property).

        Rows cover rated/unrated/trash, NSFW, purged, with and without vector,
        tag, file and thumbnail; sources with and without cursor; each weight
        file present or not. The conservation half (tags, sources, channels,
        thresholds, earlier log lines) is asserted unguarded for every example.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            tag = Tag.objects.create(name=_unique("tag"))
            NotificationChannel.objects.create(
                name=_unique("chan"), service=NotificationChannel.MATTERMOST
            )
            ReviewThresholds.objects.update_or_create(
                pk=1, defaults={"sfw_threshold": 4, "nsfw_threshold": 2}
            )
            for spec in sources:
                Source.objects.create(type="4chan", name=_unique("src"), cursor=spec["cursor"])
            weights_paths = [data_dir / "taste.pkl", data_dir / "nsfw.pkl"]
            for path, present in zip(weights_paths, weights_present, strict=True):
                if present:
                    path.write_bytes(b"pickle")
            LogEntry.objects.create(level="INFO", source="scrape", message="earlier run")
            files_written = _create_library(data_dir, rows, tag)
            before = {
                "tags": Tag.objects.count(),
                "sources": Source.objects.count(),
                "channels": NotificationChannel.objects.count(),
                "logs": LogEntry.objects.count(),
            }

            result = fresh_start(data_dir, weights_paths)

            # Gone: rows, files, thumbs, weights, cursors, image↔tag links.
            self.assertEqual(Image.objects.count(), 0)
            self.assertEqual(result["images"], len(rows))
            self.assertEqual(_files_below(data_dir / "images"), [])
            self.assertEqual(_files_below(data_dir / "thumbs"), [])
            self.assertEqual(result["files"], files_written)
            self.assertFalse(any(path.exists() for path in weights_paths))
            self.assertEqual(result["weights"], sum(weights_present))
            self.assertEqual(Source.objects.filter(cursor__isnull=False).count(), 0)
            self.assertEqual(result["cursors"], sum(1 for s in sources if s["cursor"] is not None))
            self.assertEqual(tag.images.count(), 0)
            # Kept: configuration and history (unguarded for every example).
            self.assertEqual(Tag.objects.count(), before["tags"])
            self.assertEqual(Source.objects.count(), before["sources"])
            self.assertEqual(NotificationChannel.objects.count(), before["channels"])
            thresholds = ReviewThresholds.objects.get(pk=1)
            self.assertEqual((thresholds.sfw_threshold, thresholds.nsfw_threshold), (4, 2))
            self.assertEqual(LogEntry.objects.count(), before["logs"] + 1)
            self.assertTrue(LogEntry.objects.filter(source="scrape", message="earlier run").exists())
            self.assertTrue(LogEntry.objects.filter(source="reset").exists())


@override_settings(DEBUG=True)
class FreshStartViewTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self.user = get_user_model().objects.create_user(_unique("reset"), password="pw")
        self.client.force_login(self.user)
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.settings_override = override_settings(
            DATA_DIR=self.data_dir,
            WEIGHTS_PATH=self.data_dir / "taste.pkl",
            NSFW_WEIGHTS_PATH=self.data_dir / "nsfw.pkl",
        )
        self.settings_override.enable()
        h = uuid.uuid4().hex
        Image.objects.create(content_hash=h, file_path=f"images/{h}.png", source_label="t", score=5)
        (self.data_dir / "taste.pkl").write_bytes(b"pickle")

    def tearDown(self) -> None:
        self.settings_override.disable()
        self._tmp.cleanup()

    def _post(self, word: str | None):
        data = {} if word is None else {"confirm": word}
        return self.client.post(reverse("fresh_start"), data, follow=True)

    def test_config_offers_the_reset_behind_a_typed_confirmation(self) -> None:
        """Contract: V12"""
        html = self.client.get(reverse("config")).content.decode()
        self.assertIn(f'data-confirm-word="{CONFIRM_WORD}"', html)
        self.assertIn('name="confirm"', html)
        self.assertIn(reverse("fresh_start"), html)

    def test_anonymous_request_is_sent_to_login(self) -> None:
        """OWASP A01"""
        self.client.logout()
        response = self.client.post(reverse("fresh_start"), {"confirm": CONFIRM_WORD})
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response["Location"])
        self.assertEqual(Image.objects.count(), 1)

    def test_get_is_not_allowed(self) -> None:
        """OWASP A01"""
        self.assertEqual(self.client.get(reverse("fresh_start")).status_code, 405)
        self.assertEqual(Image.objects.count(), 1)

    def test_wrong_or_missing_word_deletes_nothing_and_says_so(self) -> None:
        """Contract: V12 (OWASP A04: the word is checked on the server)"""
        for word in (None, "", "reset", "DELETE", "RESE"):
            with self.subTest(word=word):
                response = self._post(word)
                self.assertEqual(Image.objects.count(), 1)
                self.assertTrue((self.data_dir / "taste.pkl").exists())
                self.assertIn(f"Type {CONFIRM_WORD}", response.content.decode())

    def test_surrounding_whitespace_around_the_word_is_forgiven(self) -> None:
        """Contract: V12 (a phone keyboard likes to append a space)"""
        response = self._post(f" {CONFIRM_WORD} ")
        self.assertEqual(Image.objects.count(), 0)
        self.assertIn("Fresh start done", response.content.decode())

    @mock.patch("django_q.tasks.fetch", return_value=SimpleNamespace(stopped=None, result=None))
    def test_running_job_blocks_the_reset(self, _fetch) -> None:
        """Contract: V12"""
        session = self.client.session
        session["training_task_id"] = "active-task"
        session["training_started_at"] = timezone.now().isoformat()
        session.save()
        response = self._post(CONFIRM_WORD)
        self.assertEqual(Image.objects.count(), 1)
        self.assertIn("is running", response.content.decode())

    @mock.patch("django_q.tasks.fetch", return_value=SimpleNamespace(stopped=None, result=None))
    def test_running_scrape_blocks_the_reset(self, _fetch) -> None:
        """Contract: V12 — a scrape started from this browser counts the same as a training run."""
        session = self.client.session
        session["scrape_task_id"] = "active-scrape"
        session["scrape_started_at"] = timezone.now().isoformat()
        session.save()
        response = self._post(CONFIRM_WORD)
        self.assertEqual(Image.objects.count(), 1)
        self.assertIn("is running", response.content.decode())

    def test_a_task_queued_by_another_session_blocks_the_reset(self) -> None:
        """Contract: V12 — the broker table, not only this browser's session, counts as running."""
        OrmQ.objects.create(key="memelord", payload="{}")
        response = self._post(CONFIRM_WORD)
        self.assertEqual(Image.objects.count(), 1)
        self.assertIn("is running", response.content.decode())

    def test_correct_word_wipes_the_library_and_reports(self) -> None:
        """Contract: V12"""
        response = self._post(CONFIRM_WORD)
        self.assertEqual(Image.objects.count(), 0)
        self.assertFalse((self.data_dir / "taste.pkl").exists())
        html = response.content.decode()
        self.assertIn("Fresh start done", html)
        self.assertIn('id="badge-queue">0<', html)


@override_settings(DEBUG=True)
class FreshStartCommandTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.settings_override = override_settings(
            DATA_DIR=self.data_dir,
            WEIGHTS_PATH=self.data_dir / "taste.pkl",
            NSFW_WEIGHTS_PATH=self.data_dir / "nsfw.pkl",
        )
        self.settings_override.enable()
        h = uuid.uuid4().hex
        Image.objects.create(content_hash=h, file_path=f"images/{h}.png", source_label="t")

    def tearDown(self) -> None:
        self.settings_override.disable()
        self._tmp.cleanup()

    def test_yes_flag_runs_without_a_prompt(self) -> None:
        """Contract: V12 (CLI)"""
        out = StringIO()
        call_command("fresh_start", "--yes", stdout=out)
        self.assertIn("Fresh start done", out.getvalue())
        self.assertEqual(Image.objects.count(), 0)

    def test_prompt_rejects_anything_but_the_word(self) -> None:
        """Contract: V12 (CLI)"""
        with mock.patch("builtins.input", return_value="yes please"):
            with self.assertRaises(CommandError):
                call_command("fresh_start")
        self.assertEqual(Image.objects.count(), 1)
