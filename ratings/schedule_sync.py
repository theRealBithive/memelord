"""Keep the django-q scrape schedule aligned with the ScrapeSchedule singleton."""

from datetime import timedelta

from django.utils import timezone

from ratings.models import ScrapeSchedule

SCHEDULE_FUNC = "ratings.tasks.run_scrape"
SCHEDULE_NAME = "auto_scrape"


def sync_scrape_q_schedule(*, restart_interval: bool) -> None:
    """
    Make the django-q Schedule rows match the ScrapeSchedule singleton: none
    when disabled, exactly one when enabled (schedule contract V2, V3).

    The row is updated in place, never deleted and recreated: a new django-q
    row defaults to next_run = now, which made every save and every container
    start (post_migrate runs this on each `migrate`) fire a scrape at once, so
    a weekly interval could never run down. `restart_interval` says who calls:
    the Config form, where the operator has just pressed Enable and the
    interval starts from now (V1); or the post_migrate bootstrap, which only
    reconciles the interval and leaves the due time alone (V3). A missing row
    is created due one interval from now in both cases, never immediately.

    Of a duplicate set (left behind by the old delete-and-recreate) the
    soonest-due row survives, so the merge never loses a run that is due.
    A row without a due time can never fire, so it gets one as well.
    """
    try:
        from django_q.models import Schedule as QSchedule
    except ImportError:
        return

    config, _ = ScrapeSchedule.objects.get_or_create(
        pk=1,
        defaults={"interval_hours": 6, "enabled": False},
    )
    rows = QSchedule.objects.filter(func=SCHEDULE_FUNC)
    if not config.enabled:
        rows.delete()
        return

    minutes = config.interval_hours * 60
    due_in_one_interval = timezone.now() + timedelta(minutes=minutes)
    kept = rows.order_by("next_run", "id").first()
    if kept is None:
        QSchedule.objects.create(
            func=SCHEDULE_FUNC,
            name=SCHEDULE_NAME,
            schedule_type=QSchedule.MINUTES,
            minutes=minutes,
            repeats=-1,
            next_run=due_in_one_interval,
        )
        return

    rows.exclude(pk=kept.pk).delete()
    kept.name = SCHEDULE_NAME
    kept.schedule_type = QSchedule.MINUTES
    kept.minutes = minutes
    kept.repeats = -1
    if restart_interval or kept.next_run is None:
        kept.next_run = due_in_one_interval
    kept.save()
