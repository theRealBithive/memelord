"""Keep django-q scrape schedules aligned with the ScrapeSchedule singleton."""

from ratings.models import ScrapeSchedule

SCHEDULE_FUNC = "ratings.tasks.run_scrape"
SCHEDULE_NAME = "auto_scrape"


def sync_scrape_q_schedule() -> None:
    """Replace all scrape Q schedules with one row matching ScrapeSchedule, or none."""
    try:
        from django_q.models import Schedule as QSchedule
    except ImportError:
        return

    config, _ = ScrapeSchedule.objects.get_or_create(
        pk=1,
        defaults={"interval_hours": 6, "enabled": False},
    )
    QSchedule.objects.filter(func=SCHEDULE_FUNC).delete()
    if not config.enabled:
        return
    QSchedule.objects.create(
        func=SCHEDULE_FUNC,
        name=SCHEDULE_NAME,
        schedule_type=QSchedule.MINUTES,
        minutes=config.interval_hours * 60,
        repeats=-1,
    )
