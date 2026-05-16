from django.apps import AppConfig


class RatingsConfig(AppConfig):
    name = "ratings"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        from django.db.models.signals import post_migrate
        post_migrate.connect(_bootstrap_schedule, sender=self)


def _bootstrap_schedule(sender, **kwargs):
    try:
        from django_q.models import Schedule
    except ImportError:
        return
    Schedule.objects.get_or_create(
        func="ratings.tasks.run_scrape",
        defaults={
            "name": "Periodic scrape",
            "schedule_type": Schedule.HOURLY,
            "minutes": 6,
            "repeats": -1,
        },
    )
