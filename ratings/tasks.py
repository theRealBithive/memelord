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
    Run a full scrape in the background worker and return a result dict.

    Called by django-q both on the auto_scrape schedule and from the manual
    trigger_scrape button (the latter polls scrape_status, which reads this
    return value). Returns {"ok": True, "total": N, "counts": {...}} on success
    or {"ok": False, "error": "..."} on failure — mirroring run_train so the
    polling view can render a result fragment without storing state elsewhere.
    Wraps scraper.run() with a DB sink so output shows in the in-app log viewer.
    """
    from loguru import logger

    from ratings import search

    _trim_logs()
    sink_id = logger.add(_db_sink("scrape"), format="{message}")
    try:
        counts = scraper.run(
            config_path=Path(settings.CONFIG_PATH),
            data_dir=Path(settings.DATA_DIR),
            vision=scraper.vision_config_from_settings(),
        )
        # New images get their search vector from the index chain, not inline:
        # a 25k-image scrape would otherwise run 3.5 h longer and hit the
        # cluster timeout. The chain starts here, in the background wrapper,
        # so the CLI scrape stays pure and the auto_scrape schedule is covered.
        if search.enqueue_index_job_if_needed():
            logger.info("Search index job queued for the new images.")
        return {"ok": True, "total": sum(counts.values()), "counts": counts}
    except Exception as exc:
        logger.error("Scrape failed: {}", exc)
        return {"ok": False, "error": str(exc)}
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
    from ratings.scraper import classify_images, populate_knn_tag_suggestions

    _trim_logs()
    sink_id = logger.add(_db_sink("train"), format="{message}")
    try:
        trainer.run(
            data_dir=Path(settings.DATA_DIR),
            weights_path=Path(settings.WEIGHTS_PATH),
            nsfw_weights_path=Path(settings.NSFW_WEIGHTS_PATH),
            nsfw_threshold=settings.NSFW_THRESHOLD,
        )
        # Re-run classification on unscored images with the freshly-trained
        # model so images scraped under an older classifier get a fresh score.
        classify_images(
            data_dir=Path(settings.DATA_DIR),
            vision=scraper.vision_config_from_settings(),
        )
        # Refresh kNN tag suggestions so newly-tagged anchors propagate
        # immediately and existing rows recompute against the latest threshold.
        populate_knn_tag_suggestions(refill=True)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        logger.remove(sink_id)


def run_search_index():
    """
    Encode one slice of the search index and queue the next one (V3, V13, V14).

    Why slices: the single django-q worker also runs scrapes and training. A
    25k backlog is 3.5 hours on this CPU; as one task it would hold the worker
    for that long and sit right at the 4 h cluster timeout. One slice of 1000
    images is about 10 minutes, after which a waiting scrape gets its turn,
    then the chain continues with the slice this function enqueues before it
    returns (so OrmQ is never empty between slices and the UI never flickers
    to "idle").

    Why the stop rule: a slice that encoded nothing although rows are still
    stale can only be looking at missing or unreadable files; re-queueing
    would loop forever. Missing weights (EncoderUnavailableError) end the
    chain the same way with the message that says what to do (V10).
    """
    from loguru import logger

    from core.brain import EncoderUnavailableError
    from ratings import search

    _trim_logs()
    sink_id = logger.add(_db_sink("index"), format="{message}")
    try:
        try:
            result = search.encode_stale_search_embeddings(
                Path(settings.DATA_DIR), limit=search.INDEX_SLICE_SIZE
            )
        except EncoderUnavailableError as exc:
            logger.error("Search index stopped: {}", exc)
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            logger.error("Search index failed: {}", exc)
            return {"ok": False, "error": str(exc)}

        remaining = search.stale_search_images().count()
        if remaining == 0:
            logger.info("Search index complete ({} encoded in the last slice).", result["encoded"])
        elif result["encoded"] > 0:
            from django_q.tasks import async_task

            logger.info("Search index: {} images left, next slice queued.", remaining)
            async_task(search.INDEX_TASK)
        else:
            logger.warning(
                "Search index stopped: {} image(s) cannot be encoded (file missing or "
                "unreadable). Run `manage.py repair_orphans`, then index again.",
                remaining,
            )
        return {"ok": True, "encoded": result["encoded"], "remaining": remaining}
    finally:
        logger.remove(sink_id)
