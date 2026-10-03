"""Text search: the SigLIP2 vector generation, the background index job, the ranking."""

from pathlib import Path

from django.db.models import F, Q
from loguru import logger

from core import brain, siglip
from ratings.models import Image
from ratings.vector_bank import VectorBank, first_allowed

MAX_QUERY_LENGTH = 200
RESULT_LIMIT = 100
INDEX_SLICE_SIZE = 1000
INDEX_TASK = "ratings.tasks.run_search_index"
SCOPES = ("rated", "all")
DEFAULT_SCOPE = "rated"

SEARCH_BANK = VectorBank(
    "search_embedding", "search_embedding_model", siglip.SEARCH_ENCODER_ID
)


def has_search_embedding(image: Image) -> bool:
    """Same two-part rule as ratings.embeddings.has_current_embedding, for the search vector (V1)."""
    return (
        image.search_embedding is not None
        and image.search_embedding_model == siglip.SEARCH_ENCODER_ID
    )


def stale_search_images():
    """Non-purged rows whose search vector is missing or from another encoder (V3, V4)."""
    return Image.objects.filter(is_purged=False).exclude(
        search_embedding__isnull=False, search_embedding_model=siglip.SEARCH_ENCODER_ID
    )


def index_counts() -> tuple[int, int]:
    """(indexed, total) over non-purged images, for the config page and the gallery hint (V7, V15)."""
    live = Image.objects.filter(is_purged=False)
    total = live.count()
    indexed = (
        live.filter(search_embedding_model=siglip.SEARCH_ENCODER_ID)
        .exclude(search_embedding=None)
        .count()
    )
    return indexed, total


def encode_stale_search_embeddings(
    data_dir: Path,
    *,
    encoder=None,
    transform=None,
    chunk_size: int = 256,
    batch_size: int = 32,
    limit: int | None = None,
    progress_label: str = "index",
) -> dict[str, int]:
    """
    Give stale rows a search vector, rated images first (contract V3).

    This is a deliberate copy of ratings.embeddings.reencode_stale_embeddings
    rather than a parameterised version of it: the DINOv3 loop is covered by
    the mutation suite and must stay untouched, and a reader of either loop
    should not have to hold the other file in their head.

    Order: most recently rated first, then unrated by download time. The
    gallery searches rated images by default, so after the first slice of a
    25k backlog the gallery is already searchable while the rest fills in.
    `only()` keeps the 3 KB DINOv3 blob of every row out of memory; `limit`
    is the slice size of the background job. Each row is saved as soon as it
    is encoded, so an interrupted slice keeps its progress.

    Rows whose file is missing are skipped and counted: marking them purged
    is repair_orphans' job.
    """
    queryset = (
        stale_search_images()
        .order_by(F("rated_at").desc(nulls_last=True), "downloaded_at")
        .only("content_hash", "file_path")
    )
    if limit:
        queryset = queryset[:limit]
    images = list(queryset)
    present = [img for img in images if (data_dir / img.file_path).exists()]
    missing_file = len(images) - len(present)
    result = {"encoded": 0, "missing_file": missing_file, "unreadable": 0}
    if not present:
        return result

    if encoder is None:
        encoder = siglip.get_image_encoder()
    if transform is None:
        transform = siglip.get_image_transform()

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
            img.search_embedding = brain.embedding_to_bytes(emb)
            img.search_embedding_model = siglip.SEARCH_ENCODER_ID
            img.save(update_fields=["search_embedding", "search_embedding_model"])
            result["encoded"] += 1
        done = min(start + chunk_size, total)
        logger.info(
            "{}: {}/{} images encoded with {}",
            progress_label, done, total, siglip.SEARCH_ENCODER_ID,
        )
    return result


def index_job_queued() -> bool:
    """
    True while an index slice is queued or running (contract V13, V15).

    OrmQ is django-q's queue table; a task's row stays there until the
    worker acknowledges it after completion, so "queued or running" is one
    question. The chain enqueues the next slice before the current one
    returns, so the table is never empty between two slices (risk R14).
    """
    from django_q.models import OrmQ

    return any(queued.func() == INDEX_TASK for queued in OrmQ.objects.all())


def enqueue_index_job_if_needed() -> bool:
    """
    Start the index chain unless there is nothing to do or it already runs.

    One chain at a time (V13): a scrape that ends while the chain runs would
    otherwise start a second one, and both would walk the same stale rows.
    """
    if not stale_search_images().exists():
        return False
    if index_job_queued():
        return False
    from django_q.tasks import async_task

    async_task(INDEX_TASK)
    return True


def last_index_report() -> str | None:
    """
    A sentence about the last finished slice when it ended with a problem,
    else None; shown on the config page when no slice is queued.

    Two endings count as a problem: the worker raised (missing weights, V10),
    or the slice could not encode a single image although rows were still
    stale, which is the stop rule for unreadable files (V14).
    """
    from django_q.models import Task

    task = Task.objects.filter(func=INDEX_TASK).order_by("-stopped").first()
    if task is None:
        return None
    result = task.result if isinstance(task.result, dict) else {}
    if not task.success or result.get("ok") is False:
        error = result.get("error") or str(task.result or "Task exited without a result.")
        return f"Last index run failed: {error}"
    remaining = result.get("remaining", 0)
    if remaining and result.get("encoded", 0) == 0:
        return (
            f"{remaining} image(s) could not be indexed (file missing or unreadable). "
            "Run `manage.py repair_orphans`, then index again."
        )
    return None


def normalize_query(raw: str | None) -> str:
    """Collapse whitespace and cap the length; the result goes to the tokenizer only (V8)."""
    if not raw:
        return ""
    collapsed = " ".join(str(raw).split())
    return collapsed[:MAX_QUERY_LENGTH]


def normalize_scope(raw: str | None) -> str:
    """Whitelist for the ?scope= parameter (V8)."""
    if raw in SCOPES:
        return raw
    return DEFAULT_SCOPE


def candidate_images(scope: str, min_score: int, tag: str, show_nsfw: bool):
    """
    The set a search or a similarity ranking may draw from (contract V5).

    `rated` is exactly the gallery: scored at least min_score. `all` adds the
    unrated backlog; min_score still applies to the rated part, so trash
    (score 0) never comes back through a search. The NSFW switch and the tag
    filter apply in both scopes. The normal gallery uses this function too,
    with `rated`, so "result is a subset of the gallery" holds by construction.
    """
    queryset = Image.objects.filter(is_purged=False)
    if scope == "all":
        queryset = queryset.filter(Q(score__isnull=True) | Q(score__gte=min_score))
    else:
        queryset = queryset.filter(score__isnull=False, score__gte=min_score)
    if not show_nsfw:
        queryset = queryset.filter(is_nsfw=False)
    if tag:
        queryset = queryset.filter(tags__name=tag)
    return queryset


def rank_by_text(query: str, candidates, limit: int) -> list[tuple[str, float]]:
    """
    The best `limit` candidates for a text query, highest similarity first
    (V5, V6). An empty candidate set returns before the text model is touched,
    so a filter that matches nothing never pays the model load (V11, R10).
    """
    allowed = set(candidates.values_list("content_hash", flat=True))
    if not allowed:
        return []
    query_vector = siglip.encode_text(query)
    ranked_hashes, similarities = SEARCH_BANK.rank(query_vector)
    return first_allowed(ranked_hashes, similarities, allowed, limit)


def unindexed_count(candidates) -> int:
    """Candidates a search cannot see yet, for the hint under the result count (V7)."""
    return candidates.exclude(search_embedding_model=siglip.SEARCH_ENCODER_ID).count()
