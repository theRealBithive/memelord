"""Scrape schedule sync dedupes django-q rows and respects ScrapeSchedule."""

import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.test import TestCase

from ratings.models import ScrapeSchedule
from ratings.schedule_sync import SCHEDULE_FUNC, SCHEDULE_NAME, sync_scrape_q_schedule


class SyncScrapeQScheduleTests(TestCase):
    def test_sync_removes_duplicate_schedules_when_disabled(self) -> None:
        from django_q.models import Schedule as QSchedule

        QSchedule.objects.create(
            func=SCHEDULE_FUNC,
            name="Periodic scrape",
            schedule_type=QSchedule.HOURLY,
            minutes=6,
            repeats=-1,
        )
        QSchedule.objects.create(
            func=SCHEDULE_FUNC,
            name=SCHEDULE_NAME,
            schedule_type=QSchedule.MINUTES,
            minutes=360,
            repeats=-1,
        )

        sync_scrape_q_schedule()

        assert QSchedule.objects.filter(func=SCHEDULE_FUNC).count() == 0

    def test_sync_creates_single_schedule_when_enabled(self) -> None:
        from django_q.models import Schedule as QSchedule

        ScrapeSchedule.objects.update_or_create(
            pk=1,
            defaults={"interval_hours": 2, "enabled": True},
        )

        sync_scrape_q_schedule()

        schedules = list(QSchedule.objects.filter(func=SCHEDULE_FUNC))
        assert len(schedules) == 1
        assert schedules[0].name == SCHEDULE_NAME
        assert schedules[0].minutes == 120

    def test_bootstrap_handles_existing_duplicates(self) -> None:
        from django_q.models import Schedule as QSchedule

        from ratings.apps import _bootstrap_schedule

        QSchedule.objects.create(
            func=SCHEDULE_FUNC,
            name="Periodic scrape",
            schedule_type=QSchedule.HOURLY,
            minutes=6,
            repeats=-1,
        )
        QSchedule.objects.create(
            func=SCHEDULE_FUNC,
            name=SCHEDULE_NAME,
            schedule_type=QSchedule.MINUTES,
            minutes=60,
            repeats=-1,
        )

        _bootstrap_schedule(sender=None)

        assert QSchedule.objects.filter(func=SCHEDULE_FUNC).count() == 0
