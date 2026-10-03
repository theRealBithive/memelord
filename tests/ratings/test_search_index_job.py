"""
The search index job: slices, the chain, its start points and its status.

Contract (full list in tests/ratings/test_search.py):

V3  Der Index-Job startet von selbst nach jedem Hintergrund-Scrape, auf
    Knopfdruck in der Konfiguration oder per Kommando.
V10 Können die SigLIP2-Gewichte nicht geladen werden, endet der Index-Job mit
    einer einzigen klaren Meldung und reiht sich nicht erneut ein.
V13 Der Index-Job belegt den Hintergrund-Worker nie länger als eine Scheibe am
    Stück. Bleibt danach etwas übrig, reiht er sich selbst erneut ein. Es ist
    nie mehr als eine Index-Kette gleichzeitig eingereiht.
V14 Der Index-Job endet von selbst: wenn nichts mehr aussteht, oder wenn eine
    ganze Scheibe kein einziges Bild kodieren konnte. Im zweiten Fall nennt er
    die Anzahl und verweist auf `repair_orphans`. Er dreht nie endlos.
V15 Die Konfigurationsseite zeigt „x von y Bildern indexiert" und den
    Start-Knopf. Während der Kette zeigt sie den Fortschritt, die Navigation
    zeigt den laufenden Job, und Fresh Start ist gesperrt. Dieser Zustand kommt
    aus der Datenbank, nicht aus der Browser-Session.
"""

from __future__ import annotations

import os
import uuid
from unittest import mock

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from django_q.models import OrmQ, Task
from django_q.tasks import async_task

from core import siglip
from core.brain import EncoderUnavailableError
from ratings import search, tasks
from ratings.context_processors import _index_job_ctx
from ratings.models import Image, LogEntry
from ratings.reset import CONFIRM_WORD

CURRENT = siglip.SEARCH_ENCODER_ID


def _row(*, indexed: bool, purged: bool = False) -> Image:
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.png",
        source_label="t",
        is_purged=purged,
        search_embedding=b"\x00" * 3072 if indexed else None,
        search_embedding_model=CURRENT if indexed else "",
    )


def _finished_task(*, success: bool, result) -> Task:
    now = timezone.now()
    return Task.objects.create(
        id=uuid.uuid4().hex, name=uuid.uuid4().hex[:8], func=search.INDEX_TASK,
        started=now, stopped=now, success=success, result=result,
    )


def _clear_queue() -> None:
    OrmQ.objects.all().delete()
    Task.objects.filter(func=search.INDEX_TASK).delete()


class RunSearchIndexTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        LogEntry.objects.all().delete()
        _clear_queue()

    def _run(self, slice_result: dict):
        with mock.patch.object(search, "encode_stale_search_embeddings", return_value=slice_result) as encode, \
             mock.patch("django_q.tasks.async_task") as enqueue:
            result = tasks.run_search_index()
        return result, encode, enqueue

    def test_a_slice_with_work_left_queues_the_next_slice(self) -> None:
        """Contract: V13"""
        _row(indexed=False)
        _row(indexed=False)
        result, encode, enqueue = self._run({"encoded": 1, "missing_file": 0, "unreadable": 0})
        encode.assert_called_once()
        self.assertEqual(encode.call_args.kwargs["limit"], search.INDEX_SLICE_SIZE)
        enqueue.assert_called_once_with(search.INDEX_TASK)
        self.assertEqual(result, {"ok": True, "encoded": 1, "remaining": 2})

    def test_the_last_slice_does_not_requeue(self) -> None:
        """Contract: V14"""
        _row(indexed=True)
        result, _, enqueue = self._run({"encoded": 1, "missing_file": 0, "unreadable": 0})
        enqueue.assert_not_called()
        self.assertEqual(result["remaining"], 0)
        self.assertTrue(LogEntry.objects.filter(source="index", message__contains="complete").exists())

    def test_a_slice_that_encodes_nothing_stops_the_chain_and_names_repair_orphans(self) -> None:
        """Contract: V14 (risk R11)"""
        _row(indexed=False)
        result, _, enqueue = self._run({"encoded": 0, "missing_file": 1, "unreadable": 0})
        enqueue.assert_not_called()
        self.assertEqual(result, {"ok": True, "encoded": 0, "remaining": 1})
        warning = LogEntry.objects.filter(source="index", level="WARNING").first()
        self.assertIsNotNone(warning)
        self.assertIn("repair_orphans", warning.message)
        self.assertIn("1 image", warning.message)

    def test_missing_weights_end_the_chain_with_the_clear_message(self) -> None:
        """Contract: V10"""
        _row(indexed=False)
        with mock.patch.object(search, "encode_stale_search_embeddings", side_effect=EncoderUnavailableError("set HF_TOKEN")), \
             mock.patch("django_q.tasks.async_task") as enqueue:
            result = tasks.run_search_index()
        enqueue.assert_not_called()
        self.assertEqual(result, {"ok": False, "error": "set HF_TOKEN"})
        self.assertTrue(LogEntry.objects.filter(source="index", message__contains="set HF_TOKEN").exists())

    def test_any_other_failure_is_reported_and_not_requeued(self) -> None:
        """Contract: V14"""
        _row(indexed=False)
        with mock.patch.object(search, "encode_stale_search_embeddings", side_effect=RuntimeError("disk full")), \
             mock.patch("django_q.tasks.async_task") as enqueue:
            result = tasks.run_search_index()
        enqueue.assert_not_called()
        self.assertEqual(result, {"ok": False, "error": "disk full"})


class EnqueueTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        _clear_queue()

    def test_nothing_stale_means_nothing_queued(self) -> None:
        """Contract: V13"""
        _row(indexed=True)
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertFalse(search.enqueue_index_job_if_needed())
        enqueue.assert_not_called()

    def test_stale_rows_start_exactly_one_chain(self) -> None:
        """Contract: V3, V13"""
        _row(indexed=False)
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertTrue(search.enqueue_index_job_if_needed())
        enqueue.assert_called_once_with(search.INDEX_TASK)

    def test_a_queued_index_task_blocks_a_second_chain(self) -> None:
        """Contract: V13 (risk R12)"""
        _row(indexed=False)
        async_task(search.INDEX_TASK)
        self.assertTrue(search.index_job_queued())
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertFalse(search.enqueue_index_job_if_needed())
        enqueue.assert_not_called()

    def test_another_kind_of_queued_task_does_not_block(self) -> None:
        """Contract: V13 (risk R12)"""
        _row(indexed=False)
        async_task("ratings.tasks.run_scrape")
        self.assertFalse(search.index_job_queued())
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertTrue(search.enqueue_index_job_if_needed())
        enqueue.assert_called_once_with(search.INDEX_TASK)

    def test_background_scrape_starts_the_chain(self) -> None:
        """Contract: V3"""
        with mock.patch("ratings.scraper.run", return_value={"4chan/wg": 3}), \
             mock.patch("ratings.scraper.vision_config_from_settings"), \
             mock.patch.object(search, "enqueue_index_job_if_needed", return_value=True) as enqueue:
            result = tasks.run_scrape()
        self.assertTrue(result["ok"])
        enqueue.assert_called_once_with()

    def test_failed_scrape_does_not_start_the_chain(self) -> None:
        """Contract: V3 (a scrape that raised has nothing new to index)"""
        with mock.patch("ratings.scraper.run", side_effect=RuntimeError("boom")), \
             mock.patch("ratings.scraper.vision_config_from_settings"), \
             mock.patch.object(search, "enqueue_index_job_if_needed") as enqueue:
            result = tasks.run_scrape()
        self.assertFalse(result["ok"])
        enqueue.assert_not_called()


class LastReportTests(TestCase):
    def setUp(self) -> None:
        _clear_queue()

    def test_no_task_yet_means_no_report(self) -> None:
        """Contract: V15"""
        self.assertIsNone(search.last_index_report())

    def test_a_failed_slice_is_reported_with_its_error(self) -> None:
        """Contract: V10, V15"""
        _finished_task(success=True, result={"ok": False, "error": "set HF_TOKEN"})
        self.assertEqual(search.last_index_report(), "Last index run failed: set HF_TOKEN")

    def test_a_crashed_worker_is_reported_too(self) -> None:
        """Contract: V15"""
        _finished_task(success=False, result="Traceback …")
        self.assertEqual(search.last_index_report(), "Last index run failed: Traceback …")

    def test_a_worker_that_left_no_result_is_reported_in_plain_words(self) -> None:
        """Contract: V15 (a killed worker leaves success=False and no result; the page still says what happened)"""
        _finished_task(success=False, result=None)
        self.assertEqual(
            search.last_index_report(),
            "Last index run failed: Task exited without a result.",
        )

    def test_the_stop_rule_is_reported_with_the_count(self) -> None:
        """Contract: V14, V15"""
        _finished_task(success=True, result={"ok": True, "encoded": 0, "remaining": 4})
        self.assertEqual(
            search.last_index_report(),
            "4 image(s) could not be indexed (file missing or unreadable). "
            "Run `manage.py repair_orphans`, then index again.",
        )

    def test_an_ordinary_slice_has_nothing_to_report(self) -> None:
        """Contract: V15"""
        _finished_task(success=True, result={"ok": True, "encoded": 1000, "remaining": 24000})
        self.assertIsNone(search.last_index_report())


@override_settings(DEBUG=True)
class IndexViewsTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        _clear_queue()
        user = get_user_model().objects.create_user(f"idx-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def test_trigger_requires_post_and_login(self) -> None:
        """Contract: V8 (OWASP A01)"""
        self.assertEqual(self.client.get(reverse("trigger_search_index")).status_code, 405)
        self.client.logout()
        self.assertEqual(self.client.post(reverse("trigger_search_index")).status_code, 302)
        self.assertEqual(self.client.get(reverse("search_index_status")).status_code, 302)

    def test_trigger_queues_the_chain_and_returns_the_polling_fragment(self) -> None:
        """Contract: V13, V15"""
        _row(indexed=False)
        _row(indexed=True)
        html = self.client.post(reverse("trigger_search_index")).content.decode()
        self.assertTrue(search.index_job_queued())
        self.assertIn('hx-trigger="every 3s"', html)
        self.assertIn(reverse("search_index_status"), html)
        self.assertIn("1 of 2 done, 1 left", html)
        self.assertIn("Indexing 1 left", html)
        self.assertIn('hx-swap-oob="true"', html)

    def test_status_stops_polling_once_nothing_is_queued(self) -> None:
        """Contract: V15 (risk R14)"""
        _row(indexed=True)
        html = self.client.get(reverse("search_index_status")).content.decode()
        self.assertNotIn("hx-trigger", html)
        self.assertIn("Search index complete: 1 image searchable", html)
        self.assertIn("nav-job--idle", html)

    def test_status_shows_the_last_report_when_rows_wait_and_nothing_runs(self) -> None:
        """Contract: V14, V15"""
        _row(indexed=False)
        _finished_task(success=True, result={"ok": True, "encoded": 0, "remaining": 1})
        html = self.client.get(reverse("search_index_status")).content.decode()
        self.assertIn("could not be indexed", html)
        self.assertIn("0 of 1 indexed", html)

    def test_config_page_shows_counts_and_disables_the_button_while_queued(self) -> None:
        """Contract: V15"""
        _row(indexed=False)
        _row(indexed=True)
        idle = self.client.get(reverse("config")).content.decode()
        self.assertIn("1 of 2 images indexed, 1 waiting", idle)
        self.assertIn(f'hx-post="{reverse("trigger_search_index")}"', idle)
        self.assertNotIn("Indexing 1 left", idle)

        async_task(search.INDEX_TASK)
        busy = self.client.get(reverse("config")).content.decode()
        self.assertIn("1 of 2 done, 1 left", busy)
        self.assertIn("Indexing 1 left", busy)
        button = busy.split(f'hx-post="{reverse("trigger_search_index")}"')[1].split(">")[0]
        self.assertIn("disabled", button)

    def test_nav_indicator_and_fresh_start_follow_the_queue_not_the_session(self) -> None:
        """Contract: V15"""
        _row(indexed=False)
        async_task(search.INDEX_TASK)
        indicator = self.client.get(reverse("job_indicator")).content.decode()
        self.assertIn("Indexing 1 left", indicator)
        self.assertIn('hx-trigger="every 5s"', indicator)

        response = self.client.post(reverse("fresh_start"), {"confirm": CONFIRM_WORD}, follow=True)
        self.assertIn("index job is running", response.content.decode())
        self.assertEqual(Image.objects.count(), 1)

    def test_context_processor_stays_quiet_when_the_queue_table_is_unavailable(self) -> None:
        """Contract: V15 (first request before django-q's tables exist must still render)"""
        with mock.patch.object(search, "index_job_queued", side_effect=RuntimeError("no such table")):
            self.assertEqual(_index_job_ctx(), {"active_index": False, "index_remaining": 0})

    def test_context_processor_reads_the_queue(self) -> None:
        """Contract: V15"""
        _row(indexed=False)
        self.assertEqual(_index_job_ctx(), {"active_index": False, "index_remaining": 0})
        async_task(search.INDEX_TASK)
        self.assertEqual(_index_job_ctx(), {"active_index": True, "index_remaining": 1})
