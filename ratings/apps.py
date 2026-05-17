from django.apps import AppConfig


class RatingsConfig(AppConfig):
    name = "ratings"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        from django.db.models.signals import post_migrate

        post_migrate.connect(_bootstrap_schedule, sender=self)


def _bootstrap_schedule(sender, **kwargs):
    from ratings.schedule_sync import sync_scrape_q_schedule

    sync_scrape_q_schedule()
