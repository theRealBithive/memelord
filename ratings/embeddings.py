"""Embedding generations: which stored vectors are current, re-encoding the rest, and the chain that does it in the background."""

from pathlib import Path

from django.db.models import F
from loguru import logger

from core import brain
from ratings.models import Image

# The re-encode chain: one django-q task per slice, re-enqueued while rows
# remain (taste contract V19). 500 rows at ~1.7 s each on the CPU host is
# about 15 minutes, short enough that a waiting scrape gets its turn between
# two slices and far from the 4 h cluster timeout.
REENCODE_SLICE_SIZE = 500
REENCODE_TASK = "ratings.tasks.run_taste_reencode"


def has_current_embedding(image: Image) -> bool:
    """
    The single definition of "this row's vector may be compared with others".
    Both halves matter: a NULL blob has nothing to compare, and a blob stamped
    with another encoder lives in a different space (contracts V1, V2).
    """
    return image.embedding is not None and image.embedding_model == brain.ENCODER_ID


def stale_images():
    """
    Non-purged rows whose vector is missing or from another encoder.

    Purged rows are excluded on purpose: their file is gone, so there is
    nothing to encode, and they only need to keep blocking re-downloads by
    content hash (contract V4).
    """
    return Image.objects.filter(is_purged=False).exclude(
        embedding__isnull=False, embedding_model=brain.ENCODER_ID
    )


def taste_vector_counts() -> tuple[int, int]:
    """(current, total) over non-purged images, for the config page and the nav indicator (taste V21)."""
    live = Image.objects.filter(is_purged=False)
    total = live.count()
    current = live.filter(embedding_model=brain.ENCODER_ID, embedding__isnull=False).count()
    return current, total


def reencode_stale_embeddings(
    data_dir: Path,
    *,
    encoder=None,
    transform=None,
    chunk_size: int = 256,
    batch_size: int = 16,
    limit: int | None = None,
    progress_label: str = "reencode",
) -> dict[str, int]:
    """
    Bring every stale row up to the current encoder (contract V3), rated rows
    first (taste contract V19).

    Work is cut into chunks and each chunk's rows are saved as soon as it is
    encoded, so an interrupted run resumes where it stopped instead of
    repeating hours of GPU time: brain.encode() only returns at the very end of
    its list, so handing it the whole library would hold all progress in memory
    with nothing durable. chunk_size is that durability granularity; batch_size
    is the GPU batch inside brain.encode (16 since the 448 px input, four times
    the activations per image of 224 px).

    Order: most recently rated first, then unrated by download time, the same
    order the search index uses. The rated rows are what the next Train run
    needs; after the first slice of a 25k backlog the trainer's own inline
    backfill has little left to do while the rest fills in.

    Rows whose file is missing are skipped and counted: marking them purged is
    repair_orphans' job. Returns counts so the management command and the tests
    can report what happened.
    """
    queryset = stale_images().order_by(F("rated_at").desc(nulls_last=True), "downloaded_at")
    if limit:
        queryset = queryset[:limit]
    images = list(queryset)
    present = [img for img in images if (data_dir / img.file_path).exists()]
    missing_file = len(images) - len(present)
    result = {"encoded": 0, "missing_file": missing_file, "unreadable": 0}
    if not present:
        return result

    if encoder is None:
        encoder = brain.get_encoder()
    if transform is None:
        transform = brain.get_transform()

    total = len(present)
    for start in range(0, total, chunk_size):
        chunk = present[start : start + chunk_size]
        paths = [data_dir / img.file_path for img in chunk]
        embeddings, valid_paths = brain.encode(
            encoder,
            paths,
            transform=transform,
            batch_size=batch_size,
            progress_label=progress_label,
        )
        path_to_emb = dict(zip(valid_paths, embeddings, strict=True))
        for img, path in zip(chunk, paths, strict=True):
            emb = path_to_emb.get(path)
            if emb is None:
                result["unreadable"] += 1
                continue
            img.embedding = brain.embedding_to_bytes(emb)
            img.embedding_model = brain.ENCODER_ID
            img.save(update_fields=["embedding", "embedding_model"])
            result["encoded"] += 1
        done = min(start + chunk_size, total)
        logger.info("{}: {}/{} images re-encoded with {}", progress_label, done, total, brain.ENCODER_ID)
    return result


def reencode_job_queued() -> bool:
    """
    True while a re-encode slice is queued or running (taste contract V19, V21).

    OrmQ is django-q's queue table; a task's row stays there until the worker
    acknowledges it after completion, so "queued or running" is one question.
    The chain enqueues the next slice before the current one returns, so the
    table is never empty between two slices. A deliberate twin of
    ratings.search.index_job_queued for the other generation.
    """
    from django_q.models import OrmQ

    return any(queued.func() == REENCODE_TASK for queued in OrmQ.objects.all())


def enqueue_reencode_job_if_needed() -> bool:
    """
    Start the re-encode chain unless there is nothing to do or it already runs
    (taste contract V20). One chain at a time: a scrape that ends while the
    chain runs would otherwise start a second one over the same stale rows.
    """
    if not stale_images().exists():
        return False
    if reencode_job_queued():
        return False
    from django_q.tasks import async_task

    async_task(REENCODE_TASK)
    return True


def last_reencode_report() -> str | None:
    """
    A sentence about the last finished slice when it ended with a problem,
    else None; shown on the config page when no slice is queued (V21).

    Two endings count as a problem: the worker raised (missing weights), or
    the slice could not encode a single image although rows were still stale,
    which is the stop rule for unreadable files (V19).
    """
    from django_q.models import Task

    task = Task.objects.filter(func=REENCODE_TASK).order_by("-stopped").first()
    if task is None:
        return None
    result = task.result if isinstance(task.result, dict) else {}
    if not task.success or result.get("ok") is False:
        error = result.get("error") or str(task.result or "Task exited without a result.")
        return f"Last re-encode run failed: {error}"
    remaining = result.get("remaining", 0)
    if remaining and result.get("encoded", 0) == 0:
        return (
            f"{remaining} image(s) could not be re-encoded (file missing or unreadable). "
            "Run `manage.py repair_orphans`, then re-encode again."
        )
    return None
