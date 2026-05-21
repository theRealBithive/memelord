from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from ratings import scraper


def _db_sink(source: str):
    """
    Return a loguru sink function that writes log records to the LogEntry table.

    Used so scrape and train output is visible in the in-app log viewer without
    needing to read server-side log files. source ("scrape" or "train") lets the
    UI filter by operation type.
    """

    def _write(message):
        from ratings.models import LogEntry

        record = message.record
        LogEntry.objects.create(
            level=record["level"].name,
            source=source,
            message=record["message"],
        )

    return _write


def _trim_logs():
    """
    Delete log entries older than 48 hours to keep the table small.

    Log entries accumulate on every scrape and train run; without trimming they
    grow unbounded in the SQLite file. 48 hours keeps recent runs visible while
    preventing the table from becoming a significant fraction of the DB size.
    """
    from ratings.models import LogEntry

    cutoff = timezone.now() - timedelta(hours=48)
    LogEntry.objects.filter(timestamp__lt=cutoff).delete()


def run_scrape():
    """
    Run a full scrape in the background worker context and log output to the DB.

    Called by django-q on the auto_scrape schedule. Wraps scraper.run() with
    a DB sink so results appear in the in-app log viewer without any additional
    configuration.
    """
    from loguru import logger

    _trim_logs()
    sink_id = logger.add(_db_sink("scrape"), format="{message}")
    try:
        scraper.run(
            config_path=Path(settings.CONFIG_PATH),
            data_dir=Path(settings.DATA_DIR),
            vision=scraper.vision_config_from_settings(),
        )
    finally:
        logger.remove(sink_id)


def run_train():
    """
    Run trainer.run() in the background worker and return a result dict.

    Returns {"ok": True} on success or {"ok": False, "error": "..."} on failure.
    The dict is stored by django-q as task.result and read by train_status to
    render the success/failure fragment without keeping state elsewhere.
    """
    from loguru import logger
    from core import trainer

    _trim_logs()
    sink_id = logger.add(_db_sink("train"), format="{message}")
    try:
        trainer.run(
            data_dir=Path(settings.DATA_DIR),
            weights_path=Path(settings.WEIGHTS_PATH),
            nsfw_weights_path=Path(settings.NSFW_WEIGHTS_PATH),
            nsfw_threshold=settings.NSFW_THRESHOLD,
        )
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        logger.remove(sink_id)
