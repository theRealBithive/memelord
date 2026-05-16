from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from ratings import scraper


def _db_sink(source: str):
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
    from ratings.models import LogEntry
    cutoff = timezone.now() - timedelta(hours=48)
    LogEntry.objects.filter(timestamp__lt=cutoff).delete()


def run_scrape():
    from loguru import logger
    _trim_logs()
    sink_id = logger.add(_db_sink("scrape"), format="{message}")
    try:
        scraper.run(
            config_path=Path(settings.CONFIG_PATH),
            data_dir=Path(settings.DATA_DIR),
            weights_path=Path(settings.WEIGHTS_PATH),
        )
    finally:
        logger.remove(sink_id)


def run_train():
    from loguru import logger
    from core import trainer
    _trim_logs()
    sink_id = logger.add(_db_sink("train"), format="{message}")
    try:
        trainer.run(
            data_dir=Path(settings.DATA_DIR),
            weights_path=Path(settings.WEIGHTS_PATH),
        )
        return {"ok": True}
    except SystemExit:
        return {"ok": False, "error": "Need both corpus and void images to train."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        logger.remove(sink_id)
