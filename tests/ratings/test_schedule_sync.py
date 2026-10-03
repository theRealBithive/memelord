"""
The auto-scrape timer: the django-q Schedule row that ratings/schedule_sync.py
keeps aligned with the ScrapeSchedule singleton.

Contract (confirmed by the operator on 2026-10-03):

V1 Das Speichern des Zeitplans mit Enable startet das Intervall ab jetzt. Der
   nächste automatische Scrape ist genau ein Intervall nach dem Speichern fällig,
   nie sofort.
V2 Disable entfernt den Zeitplan vollständig. Danach ist kein automatischer
   Scrape mehr fällig.
V3 Ein Neustart (jeder `migrate`-Lauf, also jeder Container-Start) ändert einen
   laufenden Zeitplan nicht: Fälligkeit und Intervall bleiben. Fehlt die Zeile
   bei aktivem Zeitplan, wird sie mit Fälligkeit „jetzt + Intervall" angelegt.
   Mehrere Zeilen werden auf eine reduziert. Bei deaktiviertem Zeitplan wird
   jede Zeile entfernt.
V4 Ein Scrape, der während einer Auszeit des Workers fällig wurde, wird nach dem
   Neustart genau einmal nachgeholt, nicht einmal pro versäumtem Intervall.
   Danach läuft der Zeitplan im alten Raster weiter.
V5 Die Config-Seite zeigt zur nächsten Fälligkeit Wochentag, Datum und Uhrzeit
   mit Zeitzone, nicht nur die Uhrzeit.
V6 Der manuelle Knopf „Scrape now" verschiebt die nächste Fälligkeit nicht.

Why these are DB-level tests: django-q's scheduler reads the row, so the row
*is* the timer. V4 is the one item pinned on configuration (catch_up off): the
behaviour lives in django-q's scheduler loop, which cannot run inside a test
transaction (see the test's docstring).
"""

import os
import uuid
from datetime import UTC, datetime, timedelta

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from django_q.models import Schedule as QSchedule

from ratings.apps import _bootstrap_schedule
from ratings.models import ScrapeSchedule
from ratings.schedule_sync import SCHEDULE_FUNC, SCHEDULE_NAME, sync_scrape_q_schedule

# Any wall-clock slack between "now" in the test and "now" in the code under test.
SLACK = timedelta(seconds=30)


def _enable(hours: int) -> None:
    ScrapeSchedule.objects.update_or_create(pk=1, defaults={"interval_hours": hours, "enabled": True})


def _disable() -> None:
    ScrapeSchedule.objects.update_or_create(pk=1, defaults={"interval_hours": 6, "enabled": False})


def _row(**fields) -> QSchedule:
    defaults = {
        "func": SCHEDULE_FUNC,
        "name": SCHEDULE_NAME,
        "schedule_type": QSchedule.MINUTES,
        "minutes": 360,
        "repeats": -1,
    }
    defaults.update(fields)
    return QSchedule.objects.create(**defaults)


def _rows() -> list[QSchedule]:
    return list(QSchedule.objects.filter(func=SCHEDULE_FUNC).order_by("id"))


def _assert_due_in(test: TestCase, row: QSchedule, hours: int, since: datetime) -> None:
    """The row is due one interval after `since`, give or take the clock slack; never now."""
    expected = since + timedelta(hours=hours)
    test.assertGreaterEqual(row.next_run, expected - SLACK)
    test.assertLessEqual(row.next_run, expected + SLACK)
    test.assertGreater(row.next_run, since + SLACK, "a fresh timer must never be due at once")


class SyncScrapeQScheduleTests(TestCase):
    def setUp(self) -> None:
        QSchedule.objects.filter(func=SCHEDULE_FUNC).delete()

    def test_enable_creates_one_row_due_one_interval_from_now(self) -> None:
        """Contract: V1 — a missing row is created due in one interval, not now."""
        _enable(2)
        before = timezone.now()

        sync_scrape_q_schedule(restart_interval=True)

        rows = _rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, SCHEDULE_NAME)
        self.assertEqual(rows[0].minutes, 120)
        self.assertEqual(rows[0].schedule_type, QSchedule.MINUTES)
        _assert_due_in(self, rows[0], 2, before)

    def test_enable_restarts_the_interval_of_an_existing_row(self) -> None:
        """Contract: V1 — the weekly timer counts from the save, whatever the row said before."""
        stale_due = timezone.now() - timedelta(days=3)
        _row(minutes=360, next_run=stale_due)
        _enable(168)
        before = timezone.now()

        sync_scrape_q_schedule(restart_interval=True)

        rows = _rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].minutes, 168 * 60)
        _assert_due_in(self, rows[0], 168, before)

    def test_disable_removes_every_row(self) -> None:
        """Contract: V2 — including duplicates and rows of other names."""
        _row(name="Periodic scrape", schedule_type=QSchedule.HOURLY, minutes=6)
        _row()
        _disable()

        sync_scrape_q_schedule(restart_interval=True)

        self.assertEqual(_rows(), [])

    def test_bootstrap_keeps_the_due_time_and_reconciles_the_interval(self) -> None:
        """Contract: V3 — a restart changes neither the due time nor loses the configured interval."""
        due = datetime(2026, 10, 10, 11, 32, tzinfo=UTC)
        _row(minutes=60, next_run=due)
        _enable(2)

        _bootstrap_schedule(sender=None)

        rows = _rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].next_run, due)
        self.assertEqual(rows[0].minutes, 120)

    def test_bootstrap_creates_a_missing_row_due_one_interval_from_now(self) -> None:
        """Contract: V3 — never immediately, even on a fresh Q table."""
        _enable(6)
        before = timezone.now()

        _bootstrap_schedule(sender=None)

        rows = _rows()
        self.assertEqual(len(rows), 1)
        _assert_due_in(self, rows[0], 6, before)

    def test_bootstrap_collapses_duplicates_onto_the_soonest_due_row(self) -> None:
        """Contract: V3 — one row survives, and it is the one that was due first."""
        later = timezone.now() + timedelta(hours=5)
        sooner = timezone.now() + timedelta(hours=1)
        _row(name="Periodic scrape", schedule_type=QSchedule.HOURLY, minutes=6, next_run=later)
        kept = _row(minutes=60, next_run=sooner)
        _enable(6)

        _bootstrap_schedule(sender=None)

        rows = _rows()
        self.assertEqual([row.pk for row in rows], [kept.pk])
        self.assertEqual(rows[0].next_run, sooner)
        self.assertEqual(rows[0].minutes, 360)
        self.assertEqual(rows[0].name, SCHEDULE_NAME)

    def test_bootstrap_removes_every_row_when_disabled(self) -> None:
        """Contract: V3 (disabled half)."""
        _row(name="Periodic scrape", schedule_type=QSchedule.HOURLY, minutes=6)
        _row()
        _disable()

        _bootstrap_schedule(sender=None)

        self.assertEqual(_rows(), [])

    def test_a_row_without_a_due_time_gets_one(self) -> None:
        """Contract: V3 — a NULL next_run can never fire, so the bootstrap repairs it."""
        _row(next_run=None)
        _enable(4)
        before = timezone.now()

        _bootstrap_schedule(sender=None)

        _assert_due_in(self, _rows()[0], 4, before)

    def test_a_missed_run_is_made_up_once_not_once_per_missed_interval(self) -> None:
        """Contract: V4 — django-q's catch-up is off, so a schedule found overdue fires once and
        then shifts its grid past now (`scheduler()` loops `calculate_next_run` until the due time
        is in the future when CATCH_UP is False; with it on, it shifts once and fires again on
        the next cycle, one task per missed interval).

        Pinned on the configuration rather than by running `scheduler()` in-process: the ORM
        broker closes the database connection, which inside a TestCase transaction takes every
        later test in the class down with "Cannot operate on a closed database".
        """
        from django.conf import settings
        from django_q.conf import Conf

        self.assertIs(settings.Q_CLUSTER["catch_up"], False)
        self.assertIs(Conf.CATCH_UP, False)


@override_settings(DEBUG=True)
class ScheduleViewTests(TestCase):
    def setUp(self) -> None:
        QSchedule.objects.filter(func=SCHEDULE_FUNC).delete()
        user = get_user_model().objects.create_user(f"sched-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def test_pressing_enable_starts_the_interval_from_now(self) -> None:
        """Contract: V1 — through the form, weekly."""
        _row(minutes=360, next_run=timezone.now() - timedelta(days=1))
        before = timezone.now()

        response = self.client.post(
            reverse("set_scrape_schedule"), {"interval_hours": "168", "enabled": "1"}
        )

        self.assertEqual(response.status_code, 200)
        rows = _rows()
        self.assertEqual(len(rows), 1)
        _assert_due_in(self, rows[0], 168, before)

    def test_pressing_disable_leaves_nothing_due(self) -> None:
        """Contract: V2 — through the form."""
        _row()
        _enable(6)

        self.client.post(reverse("set_scrape_schedule"), {"interval_hours": "6", "enabled": "0"})

        self.assertEqual(_rows(), [])
        self.assertFalse(ScrapeSchedule.objects.get(pk=1).enabled)

    def test_status_shows_weekday_date_time_and_zone(self) -> None:
        """Contract: V5"""
        _enable(168)
        _row(minutes=168 * 60, next_run=datetime(2026, 10, 10, 11, 32, tzinfo=UTC))

        html = self.client.get(reverse("config")).content.decode()

        self.assertIn("next Sat 10 Oct 11:32 UTC", html)
        self.assertIn("every 168h", html)

    def test_scrape_now_does_not_move_the_due_time(self) -> None:
        """Contract: V6"""
        due = datetime(2026, 10, 10, 11, 32, tzinfo=UTC)
        _row(minutes=168 * 60, next_run=due)
        _enable(168)

        with mock.patch("django_q.tasks.async_task", return_value="task-1"), \
             mock.patch("django_q.tasks.fetch", return_value=None):
            response = self.client.post(reverse("trigger_scrape"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(_rows()[0].next_run, due)
