from django.apps import AppConfig


class RatingsConfig(AppConfig):
    name = "ratings"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        from django.db.backends.signals import connection_created
        from django.db.models.signals import post_migrate

        post_migrate.connect(_bootstrap_schedule, sender=self)
        connection_created.connect(_set_sqlite_pragmas)


def _set_sqlite_pragmas(sender, connection, **kwargs):
    """
    Put every new SQLite connection into WAL mode with NORMAL sync.

    The default rollback journal (journal_mode=delete) gives a writer an
    EXCLUSIVE lock that blocks *all* readers. During a scrape/train run the
    background django-q worker writes in tight bursts — per-image saves in
    classify_images / populate_knn_tag_suggestions, plus a LogEntry row per log
    line — so every web request browsing the gallery or review queue stalls
    behind it (up to the busy timeout) and the UI feels frozen.

    WAL lets readers proceed concurrently with the single writer, which is what
    makes the app usable during a background scrape. synchronous=NORMAL is the
    safe WAL pairing (a power loss can drop the last transaction but never
    corrupts the file) and saves an fsync per commit. busy_timeout is already
    set via the connection's `timeout` option in settings.

    Fires on Django's connection_created signal so it applies to connections
    from *both* gunicorn and the django-q cluster, not just a one-off init.

    `PRAGMA journal_mode=WAL` reports the mode actually adopted: on a filesystem
    that can't back WAL's shared-memory mapping (some network mounts, read-only
    volumes) SQLite silently keeps the rollback journal and returns e.g.
    "delete". We read that result back and warn, so a deployment where the
    responsiveness fix didn't take is visible in the logs instead of silently
    re-introducing the UI-freeze stall.
    """
    if connection.vendor != "sqlite":
        return
    with connection.cursor() as cursor:
        # Two statements rather than chaining .fetchone() off .execute(): the
        # latter relies on the cursor returning itself, which Django's
        # CursorWrapper does not guarantee across backends.
        cursor.execute("PRAGMA journal_mode=WAL;")
        adopted = cursor.fetchone()
        cursor.execute("PRAGMA synchronous=NORMAL;")
    if adopted and adopted[0].lower() != "wal":
        from loguru import logger

        logger.warning(
            "SQLite WAL mode not adopted (journal_mode={}); the app may stall "
            "during background scrape/train on this filesystem.",
            adopted[0],
        )


def _bootstrap_schedule(sender, **kwargs):
    from ratings.schedule_sync import sync_scrape_q_schedule

    sync_scrape_q_schedule()
