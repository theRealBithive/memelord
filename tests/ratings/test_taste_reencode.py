"""
The taste re-encode chain: slices, its start points and its status.

Contract (full list in tests/core/test_taste.py):

V18 Der Encoder sieht jedes Bild in 448×448 Pixeln. Die Vektoren tragen einen
    neuen Stempel; jeder Vektor mit altem Stempel gilt ohne Migration als
    veraltet. (tests/core/test_brain_encoder.py)
V19 Die Neu-Kodierung läuft als Kette von Scheiben zu 500 Bildern, bewertete
    zuerst, dann nach Downloadzeit. Sie endet, wenn nichts mehr veraltet ist
    oder eine Scheibe nichts kodieren konnte, und sagt im zweiten Fall, was zu
    tun ist. Zwischen zwei Scheiben ist die Warteschlange nie leer.
    (the order: tests/ratings/test_embedding_generation.py)
V20 Ein Scrape kodiert höchstens 200 veraltete Bestandsbilder inline und
    überlässt den Rest der Kette. Neue Downloads werden wie bisher beim Scrape
    kodiert. Die Kette startet nach jedem Scrape, wenn etwas veraltet ist, und
    per Knopf auf der Config-Seite. (the cap: test_embedding_generation.py)
V21 Die Config-Seite zeigt, wie viele Bilder einen aktuellen Geschmacksvektor
    haben. Der Job-Indikator zeigt die laufende Kette wie die Suchindex-Kette.
V22 Ein Training nach dem Encoder-Wechsel kodiert die bewerteten Bilder selbst
    neu und schreibt Geschmacks- und NSFW-Modell mit dem neuen Stempel.
    (tests/core/test_trainer.py backfill tests and the stamp assertions in
    tests/core/test_brain_encoder.py run against the 448 stamp)

The module mirrors tests/ratings/test_search_index_job.py on purpose: the two
chains are twins, and a reader who knows one should recognise the other.
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

from core import brain
from core.brain import EncoderUnavailableError
from ratings import embeddings, search, tasks
from ratings.context_processors import _reencode_job_ctx
from ratings.models import Image, LogEntry
from ratings.reset import CONFIRM_WORD

CURRENT = brain.ENCODER_ID


def _row(*, current: bool, purged: bool = False) -> Image:
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.png",
        source_label="t",
        is_purged=purged,
        embedding=b"\x00" * 3072 if current else None,
        embedding_model=CURRENT if current else "dinov3_vitb16",
    )


def _finished_task(*, success: bool, result) -> Task:
    now = timezone.now()
    return Task.objects.create(
        id=uuid.uuid4().hex, name=uuid.uuid4().hex[:8], func=embeddings.REENCODE_TASK,
        started=now, stopped=now, success=success, result=result,
    )


def _clear_queue() -> None:
    OrmQ.objects.all().delete()
    Task.objects.filter(func=embeddings.REENCODE_TASK).delete()


class TasteVectorCountsTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    def test_counts_follow_the_stamp_and_skip_purged_rows(self) -> None:
        """Contract: V18, V21"""
        _row(current=True)
        _row(current=False)
        _row(current=True, purged=True)
        _row(current=False, purged=True)
        self.assertEqual(embeddings.taste_vector_counts(), (1, 2))

    def test_a_current_stamp_without_a_vector_does_not_count(self) -> None:
        """Contract: V21 (the count must agree with stale_images, which treats such a row as stale)"""
        half = _row(current=True)
        Image.objects.filter(pk=half.pk).update(embedding=None)
        self.assertEqual(embeddings.taste_vector_counts(), (0, 1))
        self.assertEqual(embeddings.stale_images().count(), 1)


class RunTasteReencodeTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        LogEntry.objects.all().delete()
        _clear_queue()

    def _run(self, slice_result: dict):
        with mock.patch.object(embeddings, "reencode_stale_embeddings", return_value=slice_result) as encode, \
             mock.patch("django_q.tasks.async_task") as enqueue:
            result = tasks.run_taste_reencode()
        return result, encode, enqueue

    def test_a_slice_with_work_left_queues_the_next_slice(self) -> None:
        """Contract: V19"""
        _row(current=False)
        _row(current=False)
        result, encode, enqueue = self._run({"encoded": 1, "missing_file": 0, "unreadable": 0})
        encode.assert_called_once()
        self.assertEqual(encode.call_args.kwargs["limit"], embeddings.REENCODE_SLICE_SIZE)
        self.assertEqual(embeddings.REENCODE_SLICE_SIZE, 500)
        enqueue.assert_called_once_with(embeddings.REENCODE_TASK)
        self.assertEqual(result, {"ok": True, "encoded": 1, "remaining": 2})

    def test_the_last_slice_does_not_requeue(self) -> None:
        """Contract: V19"""
        _row(current=True)
        result, _, enqueue = self._run({"encoded": 1, "missing_file": 0, "unreadable": 0})
        enqueue.assert_not_called()
        self.assertEqual(result["remaining"], 0)
        self.assertTrue(LogEntry.objects.filter(source="reencode", message__contains="complete").exists())

    def test_a_slice_that_encodes_nothing_stops_the_chain_and_names_repair_orphans(self) -> None:
        """Contract: V19 (risk R11)"""
        _row(current=False)
        result, _, enqueue = self._run({"encoded": 0, "missing_file": 1, "unreadable": 0})
        enqueue.assert_not_called()
        self.assertEqual(result, {"ok": True, "encoded": 0, "remaining": 1})
        warning = LogEntry.objects.filter(source="reencode", level="WARNING").first()
        self.assertIsNotNone(warning)
        self.assertIn("repair_orphans", warning.message)
        self.assertIn("1 image", warning.message)

    def test_missing_weights_end_the_chain_with_the_clear_message(self) -> None:
        """Contract: V19 (mirror of search V10)"""
        _row(current=False)
        with mock.patch.object(embeddings, "reencode_stale_embeddings", side_effect=EncoderUnavailableError("set HF_TOKEN")), \
             mock.patch("django_q.tasks.async_task") as enqueue:
            result = tasks.run_taste_reencode()
        enqueue.assert_not_called()
        self.assertEqual(result, {"ok": False, "error": "set HF_TOKEN"})
        self.assertTrue(LogEntry.objects.filter(source="reencode", message__contains="set HF_TOKEN").exists())

    def test_any_other_failure_is_reported_and_not_requeued(self) -> None:
        """Contract: V19"""
        _row(current=False)
        with mock.patch.object(embeddings, "reencode_stale_embeddings", side_effect=RuntimeError("disk full")), \
             mock.patch("django_q.tasks.async_task") as enqueue:
            result = tasks.run_taste_reencode()
        enqueue.assert_not_called()
        self.assertEqual(result, {"ok": False, "error": "disk full"})


class EnqueueTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        _clear_queue()

    def test_nothing_stale_means_nothing_queued(self) -> None:
        """Contract: V20"""
        _row(current=True)
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertFalse(embeddings.enqueue_reencode_job_if_needed())
        enqueue.assert_not_called()

    def test_stale_rows_start_exactly_one_chain(self) -> None:
        """Contract: V20"""
        _row(current=False)
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertTrue(embeddings.enqueue_reencode_job_if_needed())
        enqueue.assert_called_once_with(embeddings.REENCODE_TASK)

    def test_a_queued_reencode_task_blocks_a_second_chain(self) -> None:
        """Contract: V19 (one chain at a time)"""
        _row(current=False)
        async_task(embeddings.REENCODE_TASK)
        self.assertTrue(embeddings.reencode_job_queued())
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertFalse(embeddings.enqueue_reencode_job_if_needed())
        enqueue.assert_not_called()

    def test_the_search_index_chain_does_not_block_the_reencode_chain(self) -> None:
        """Contract: V19 (the two generations have independent chains)"""
        _row(current=False)
        async_task(search.INDEX_TASK)
        self.assertFalse(embeddings.reencode_job_queued())
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertTrue(embeddings.enqueue_reencode_job_if_needed())
        enqueue.assert_called_once_with(embeddings.REENCODE_TASK)

    def test_background_scrape_starts_the_chain(self) -> None:
        """Contract: V20"""
        with mock.patch("ratings.scraper.run", return_value={"4chan/wg": 3}), \
             mock.patch("ratings.scraper.vision_config_from_settings"), \
             mock.patch.object(search, "enqueue_index_job_if_needed", return_value=False), \
             mock.patch.object(embeddings, "enqueue_reencode_job_if_needed", return_value=True) as enqueue:
            result = tasks.run_scrape()
        self.assertTrue(result["ok"])
        enqueue.assert_called_once_with()

    def test_failed_scrape_does_not_start_the_chain(self) -> None:
        """Contract: V20"""
        with mock.patch("ratings.scraper.run", side_effect=RuntimeError("boom")), \
             mock.patch("ratings.scraper.vision_config_from_settings"), \
             mock.patch.object(search, "enqueue_index_job_if_needed"), \
             mock.patch.object(embeddings, "enqueue_reencode_job_if_needed") as enqueue:
            result = tasks.run_scrape()
        self.assertFalse(result["ok"])
        enqueue.assert_not_called()


class LastReportTests(TestCase):
    def setUp(self) -> None:
        _clear_queue()

    def test_no_task_yet_means_no_report(self) -> None:
        """Contract: V21"""
        self.assertIsNone(embeddings.last_reencode_report())

    def test_a_failed_slice_is_reported_with_its_error(self) -> None:
        """Contract: V19, V21"""
        _finished_task(success=True, result={"ok": False, "error": "set HF_TOKEN"})
        self.assertEqual(embeddings.last_reencode_report(), "Last re-encode run failed: set HF_TOKEN")

    def test_a_crashed_worker_is_reported_too(self) -> None:
        """Contract: V21"""
        _finished_task(success=False, result="Traceback …")
        self.assertEqual(embeddings.last_reencode_report(), "Last re-encode run failed: Traceback …")

    def test_a_worker_that_left_no_result_is_reported_in_plain_words(self) -> None:
        """Contract: V21"""
        _finished_task(success=False, result=None)
        self.assertEqual(
            embeddings.last_reencode_report(),
            "Last re-encode run failed: Task exited without a result.",
        )

    def test_the_stop_rule_is_reported_with_the_count(self) -> None:
        """Contract: V19, V21"""
        _finished_task(success=True, result={"ok": True, "encoded": 0, "remaining": 4})
        self.assertEqual(
            embeddings.last_reencode_report(),
            "4 image(s) could not be re-encoded (file missing or unreadable). "
            "Run `manage.py repair_orphans`, then re-encode again.",
        )

    def test_an_ordinary_slice_has_nothing_to_report(self) -> None:
        """Contract: V21"""
        _finished_task(success=True, result={"ok": True, "encoded": 500, "remaining": 24000})
        self.assertIsNone(embeddings.last_reencode_report())

    def test_the_index_chains_tasks_are_not_mistaken_for_reencode_runs(self) -> None:
        """Contract: V21 (the report reads only this chain's tasks)"""
        now = timezone.now()
        Task.objects.create(
            id=uuid.uuid4().hex, name="idx", func=search.INDEX_TASK,
            started=now, stopped=now, success=False, result="index crashed",
        )
        self.assertIsNone(embeddings.last_reencode_report())


@override_settings(DEBUG=True)
class ReencodeViewsTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        _clear_queue()
        user = get_user_model().objects.create_user(f"rec-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def test_trigger_requires_post_and_login(self) -> None:
        """Contract: V20 (OWASP A01)"""
        self.assertEqual(self.client.get(reverse("trigger_taste_reencode")).status_code, 405)
        self.client.logout()
        self.assertEqual(self.client.post(reverse("trigger_taste_reencode")).status_code, 302)
        self.assertEqual(self.client.get(reverse("taste_reencode_status")).status_code, 302)

    def test_trigger_queues_the_chain_and_returns_the_polling_fragment(self) -> None:
        """Contract: V20, V21"""
        _row(current=False)
        _row(current=True)
        html = self.client.post(reverse("trigger_taste_reencode")).content.decode()
        self.assertTrue(embeddings.reencode_job_queued())
        self.assertIn('hx-trigger="every 3s"', html)
        self.assertIn(reverse("taste_reencode_status"), html)
        self.assertIn("1 of 2 done, 1 left", html)
        self.assertIn("Re-encoding 1 left", html)
        self.assertIn('hx-swap-oob="true"', html)

    def test_a_second_trigger_does_not_start_a_second_chain(self) -> None:
        """Contract: V19"""
        _row(current=False)
        self.client.post(reverse("trigger_taste_reencode"))
        self.client.post(reverse("trigger_taste_reencode"))
        queued = [q for q in OrmQ.objects.all() if q.func() == embeddings.REENCODE_TASK]
        self.assertEqual(len(queued), 1)

    def test_status_stops_polling_once_nothing_is_queued(self) -> None:
        """Contract: V21"""
        _row(current=True)
        html = self.client.get(reverse("taste_reencode_status")).content.decode()
        self.assertNotIn("hx-trigger", html)
        self.assertIn("Taste vectors complete: 1 image current", html)
        self.assertIn("nav-job--idle", html)

    def test_status_shows_the_last_report_when_rows_wait_and_nothing_runs(self) -> None:
        """Contract: V19, V21"""
        _row(current=False)
        _finished_task(success=True, result={"ok": True, "encoded": 0, "remaining": 1})
        html = self.client.get(reverse("taste_reencode_status")).content.decode()
        self.assertIn("could not be re-encoded", html)
        self.assertIn("0 of 1 current", html)

    def test_config_page_shows_counts_and_disables_the_button_while_queued(self) -> None:
        """Contract: V21"""
        _row(current=False)
        _row(current=True)
        idle = self.client.get(reverse("config")).content.decode()
        self.assertIn("1 of 2 images current, 1 waiting", idle)
        self.assertIn(f'hx-post="{reverse("trigger_taste_reencode")}"', idle)
        self.assertNotIn("Re-encoding 1 left", idle)

        async_task(embeddings.REENCODE_TASK)
        busy = self.client.get(reverse("config")).content.decode()
        self.assertIn("1 of 2 done, 1 left", busy)
        self.assertIn("Re-encoding 1 left", busy)
        button = busy.split(f'hx-post="{reverse("trigger_taste_reencode")}"')[1].split(">")[0]
        self.assertIn("disabled", button)

    def test_nav_indicator_and_fresh_start_follow_the_queue_not_the_session(self) -> None:
        """Contract: V21"""
        _row(current=False)
        async_task(embeddings.REENCODE_TASK)
        indicator = self.client.get(reverse("job_indicator")).content.decode()
        self.assertIn("Re-encoding 1 left", indicator)
        self.assertIn('hx-trigger="every 5s"', indicator)

        response = self.client.post(reverse("fresh_start"), {"confirm": CONFIRM_WORD}, follow=True)
        self.assertIn("re-encode job is running", response.content.decode())
        self.assertEqual(Image.objects.count(), 1)

    def test_context_processor_stays_quiet_when_the_queue_table_is_unavailable(self) -> None:
        """Contract: V21 (first request before django-q's tables exist must still render)"""
        with mock.patch.object(embeddings, "reencode_job_queued", side_effect=RuntimeError("no such table")):
            self.assertEqual(_reencode_job_ctx(), {"active_reencode": False, "reencode_remaining": 0})

    def test_context_processor_reads_the_queue(self) -> None:
        """Contract: V21"""
        _row(current=False)
        self.assertEqual(_reencode_job_ctx(), {"active_reencode": False, "reencode_remaining": 0})
        async_task(embeddings.REENCODE_TASK)
        self.assertEqual(_reencode_job_ctx(), {"active_reencode": True, "reencode_remaining": 1})
