from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from ratings.models import (
    Image,
    LogEntry,
    NotificationChannel,
    ReviewThresholds,
    ScrapeSchedule,
    Source,
    Tag,
)
from ratings.queue_rules import (
    below_cutoff_q,
    bucket_to_cutoff,
    get_review_thresholds,
    pred_score_visible,
    visibility_q,
)
from ratings.utils import purge_image as _purge_image_util
import ratings.notifiers as notifiers

_INTERVAL_CHOICES = [1, 2, 4, 6, 12, 24, 48, 72, 168]

WEIGHTS_PATH = Path(settings.WEIGHTS_PATH)
DATA_DIR = Path(settings.DATA_DIR)

_taste_clf_cache = None
_taste_clf_mtime: float | None = None

_similar_index_cache: dict | None = None
_similar_index_built_at: float = 0.0
_SIMILAR_INDEX_TTL = 300.0


def _get_taste_clf():
    """Load (and cache) the taste classifier, reloading if weights change on disk."""
    global _taste_clf_cache, _taste_clf_mtime
    if not WEIGHTS_PATH.exists():
        return None
    mtime = WEIGHTS_PATH.stat().st_mtime
    if _taste_clf_cache is None or mtime != _taste_clf_mtime:
        from core import brain

        _taste_clf_cache = brain.load_classifier(WEIGHTS_PATH)
        _taste_clf_mtime = mtime
    return _taste_clf_cache


def _taste_prediction(image) -> int | None:
    """Return P(corpus) as integer percentage 0–100, or None if unavailable."""
    if not image or not image.embedding:
        return None
    clf = _get_taste_clf()
    if clf is None:
        return None
    from core import brain

    emb = brain.bytes_to_embedding(bytes(image.embedding))
    return round(float(brain.predict_proba(clf, emb)) * 100)


def _get_similar_index() -> dict | None:
    """
    Cache (hashes, embeddings) of every rated image for fast kNN at view time.

    Held at module scope with a 5-minute TTL plus explicit invalidation on
    rating writes — the matrix multiplication itself is sub-millisecond but
    rebuilding the matrix from a DB scan of binary blobs would dominate the
    per-request cost without this cache.
    """
    global _similar_index_cache, _similar_index_built_at
    import time

    now = time.monotonic()
    if (
        _similar_index_cache is not None
        and now - _similar_index_built_at < _SIMILAR_INDEX_TTL
    ):
        return _similar_index_cache

    rows = list(
        Image.objects.filter(is_purged=False, score__isnull=False)
        .exclude(embedding=None)
        .values_list("content_hash", "embedding")
    )
    if not rows:
        _similar_index_cache = None
    else:
        import numpy as np
        from loguru import logger

        from core.brain import EMBEDDING_DIM

        # Guard against corrupt blobs (truncated writes, schema drift): a single
        # wrong-size row would raise ValueError inside np.stack and 500 every
        # review/gallery render until the offending row was hunted down and
        # deleted. Drop the bad rows here, log them so they can be repaired,
        # and stack the survivors.
        expected_bytes = EMBEDDING_DIM * 4
        hashes: list[str] = []
        vectors: list = []
        for content_hash, blob in rows:
            buf = bytes(blob)
            if len(buf) != expected_bytes:
                logger.warning(
                    "similar_index: skipping {} — embedding blob is {} bytes, expected {}",
                    content_hash, len(buf), expected_bytes,
                )
                continue
            hashes.append(content_hash)
            vectors.append(np.frombuffer(buf, dtype=np.float32))
        if not vectors:
            _similar_index_cache = None
        else:
            _similar_index_cache = {
                "hashes": hashes,
                "embeddings": np.stack(vectors),
            }
    _similar_index_built_at = now
    return _similar_index_cache


def _invalidate_similar_index() -> None:
    """Drop the kNN cache so the next request rebuilds it with fresh ratings."""
    global _similar_index_cache
    _similar_index_cache = None


def _get_similar_rated(image, k: int = 6) -> list[dict]:
    """
    Return the top-K most cosine-similar already-rated images.

    Shown alongside the rate card so the user can see how they (or the model)
    treated visually-comparable images before — a consistency aid, and a
    live read on whether the embedding neighbourhood reflects actual taste.
    """
    if not image or not image.embedding:
        return []
    idx = _get_similar_index()
    if not idx:
        return []
    import numpy as np

    from core import brain

    q = brain.bytes_to_embedding(bytes(image.embedding))
    sims = brain.cosine_similarity_matrix(q, idx["embeddings"])
    own_hash = image.content_hash
    order = np.argsort(-sims)
    picks: list[tuple[str, float]] = []
    for i in order:
        h = idx["hashes"][int(i)]
        if h == own_hash:
            continue
        picks.append((h, float(sims[int(i)])))
        if len(picks) >= k:
            break
    if not picks:
        return []
    hashes = [h for h, _ in picks]
    images = {
        img.content_hash: img for img in Image.objects.filter(content_hash__in=hashes)
    }
    return [
        {"image": images[h], "similarity": round(s * 100)}
        for h, s in picks
        if h in images
    ]


@login_required
def index(request):
    return redirect("review_corpus")


def _counts(show_nsfw: bool = False) -> dict:
    """
    Aggregate image counts across all queues and locations in a single DB query.

    Used by every page for nav badges; a separate query per badge would be 6×
    the DB round-trips per request. show_nsfw controls whether NSFW images are
    folded into the main counts or kept separate so the user can see SFW and
    NSFW numbers independently. The queue counts respect the [vision]
    threshold so the badge matches what the user will actually see in the
    review queue — a stale "12 to review" badge that opens onto an empty
    page would be worse than no badge at all.
    """
    sfw_bucket, nsfw_bucket = get_review_thresholds()
    sfw_pred_visible = pred_score_visible(bucket_to_cutoff(sfw_bucket))
    nsfw_pred_visible = pred_score_visible(bucket_to_cutoff(nsfw_bucket))

    qs = Image.objects.filter(is_purged=False)
    sfw_queue = Q(score__isnull=True, is_nsfw=False) & sfw_pred_visible
    nsfw_queue = Q(score__isnull=True, is_nsfw=True) & nsfw_pred_visible
    below_q = below_cutoff_q(sfw_bucket, nsfw_bucket, show_nsfw=True)

    if show_nsfw:
        return qs.aggregate(
            queue_count=Count("pk", filter=sfw_queue | nsfw_queue),
            below_cutoff_count=Count("pk", filter=below_q),
            nsfw_queue_count=Count("pk", filter=nsfw_queue),
        )
    return qs.aggregate(
        queue_count=Count("pk", filter=sfw_queue),
        below_cutoff_count=Count("pk", filter=below_q & Q(is_nsfw=False)),
        nsfw_queue_count=Count("pk", filter=nsfw_queue),
    )


def _fmt_elapsed(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


def _elapsed_from_session(request, kind: str = "training") -> int | None:
    """Return seconds since the {kind}_started_at stamp in the session, or None.

    ``kind`` namespaces the session keys ("training" / "scrape") so the same
    elapsed-time logic backs both background jobs; the default keeps the
    original training call sites unchanged.
    """
    started_at_str = request.session.get(f"{kind}_started_at")
    if not started_at_str:
        return None
    started_at = datetime.fromisoformat(started_at_str)
    return int((datetime.now(dt_timezone.utc) - started_at).total_seconds())


def _clear_training_session(request, task_id: str | None = None) -> None:
    """Drop training session keys only when ``task_id`` matches the stored job (or is omitted)."""
    session_task = request.session.get("training_task_id")
    if task_id is not None and session_task != task_id:
        return
    request.session.pop("training_task_id", None)
    request.session.pop("training_started_at", None)


def _training_task_stale(request, task_id: str) -> bool:
    """
    Return True when the session's training task is no longer running in django-q.

    Used on ordinary page loads so a finished or lost worker job does not leave
    the nav spinner stuck until someone happens to poll train_status.

    fetch() only returns completed tasks (written to django_q_task after the
    worker finishes). While the task is queued or running it lives in django_q_ormq
    and fetch() returns None — that does NOT mean the task is lost. We use elapsed
    time as the discriminator: None from fetch() is only stale once the elapsed
    time exceeds the cluster timeout (14400s), at which point the worker would have
    killed it anyway.
    """
    from django_q.tasks import fetch

    task = fetch(task_id)
    if task is None:
        elapsed = _elapsed_from_session(request)
        return elapsed is None or elapsed > 14400
    if task.stopped is not None:
        return True
    elapsed = _elapsed_from_session(request)
    return elapsed is not None and elapsed > 14400


def _training_ctx(request) -> dict:
    """
    Build the training-progress context fragment for templates.

    django-q doesn't expose task progress over HTTP, so elapsed time is tracked
    via session storage on the web process side. The 14400s (4h) cutoff matches
    Q_CLUSTER["timeout"] so the UI gives up at the same moment the worker would
    have killed the task — without it a crashed worker would leave the UI
    permanently showing "Training…".
    """
    task_id = request.session.get("training_task_id")
    if not task_id:
        return {"active_task_id": None, "training_elapsed": None}
    if _training_task_stale(request, task_id):
        _clear_training_session(request, task_id)
        return {"active_task_id": None, "training_elapsed": None}
    elapsed = _elapsed_from_session(request)
    return {"active_task_id": task_id, "training_elapsed": _fmt_elapsed(elapsed)}


def _clear_scrape_session(request, task_id: str | None = None) -> None:
    """Drop scrape session keys only when ``task_id`` matches the stored job (or is omitted)."""
    session_task = request.session.get("scrape_task_id")
    if task_id is not None and session_task != task_id:
        return
    request.session.pop("scrape_task_id", None)
    request.session.pop("scrape_started_at", None)


def _scrape_task_stale(request, task_id: str) -> bool:
    """Return True when the session's scrape task is no longer running in django-q.

    Same reasoning as _training_task_stale: fetch() returns None while the task
    is still queued/running in OrmQ, so a None result is only "lost" once elapsed
    exceeds the cluster timeout (14400s), at which point the worker would have
    been killed anyway.
    """
    from django_q.tasks import fetch

    task = fetch(task_id)
    if task is None:
        elapsed = _elapsed_from_session(request, "scrape")
        return elapsed is None or elapsed > 14400
    if task.stopped is not None:
        return True
    elapsed = _elapsed_from_session(request, "scrape")
    return elapsed is not None and elapsed > 14400


def _scrape_ctx(request) -> dict:
    """Build the scrape-progress context fragment, mirroring _training_ctx.

    Lets a page reload during an active scrape resume the polling fragment
    instead of showing an empty result box while the worker keeps running.
    """
    task_id = request.session.get("scrape_task_id")
    if not task_id:
        return {"active_scrape_task_id": None, "scrape_elapsed": None}
    if _scrape_task_stale(request, task_id):
        _clear_scrape_session(request, task_id)
        return {"active_scrape_task_id": None, "scrape_elapsed": None}
    elapsed = _elapsed_from_session(request, "scrape")
    return {"active_scrape_task_id": task_id, "scrape_elapsed": _fmt_elapsed(elapsed)}


def _purge_image(image: Image) -> None:
    """Local wrapper around the utility purge so cache invalidation stays centralised."""
    _purge_image_util(image)
    _invalidate_similar_index()


def _neighbor_hash(all_hashes: list[str], content_hash: str) -> str | None:
    """Next hash in list, or previous if last, or None if single item.

    When content_hash isn't in the queue at all (another tab rated it, or it
    was filtered out between the GET and POST), fall back to the head of the
    queue rather than treating the missing item as if it were at index 0 — the
    previous behaviour silently teleported the user one position in (returning
    `all_hashes[1]`) instead of landing them somewhere predictable.
    """
    if not all_hashes:
        return None
    idx = {h: i for i, h in enumerate(all_hashes)}.get(content_hash)
    if idx is None:
        return all_hashes[0]
    if idx < len(all_hashes) - 1:
        return all_hashes[idx + 1]
    return all_hashes[idx - 1] if idx > 0 else None


@login_required
def rate_nsfw_corpus(request, content_hash: str | None = None):
    """
    Entry point for the NSFW corpus review queue — same layout as review_corpus
    but filtered to is_nsfw=True images.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    ctx = _review_nsfw_ctx(content_hash, show_nsfw, request)
    if request.htmx:
        return render(request, "ratings/_review_htmx.html", ctx)
    return render(request, "ratings/review.html", ctx)


@login_required
@require_POST
def nsfw_toggle(request):
    """Session toggle for the show-NSFW preference; redirects back to the previous page."""
    request.session["show_nsfw"] = not request.session.get("show_nsfw", False)
    return redirect(request.META.get("HTTP_REFERER") or "index")


@login_required
def stats(request):
    """
    Render the stats dashboard.

    last_trained is derived from the weights file mtime — the weights file is
    the ground truth for "last successful save" (a DB timestamp could drift if
    the file was replaced out-of-band). last_train_task surfaces the most
    recent django-q run regardless of outcome so a user can tell the
    difference between "trained successfully yesterday" and "tried this
    morning and crashed" — without it, a failed retrain shows the same stale
    mtime as before the attempt.
    """
    from django_q.models import Task

    show_nsfw = request.session.get("show_nsfw", False)
    counts = _counts(show_nsfw)

    last_trained = None
    try:
        mtime = WEIGHTS_PATH.stat().st_mtime
        last_trained = datetime.fromtimestamp(mtime, tz=dt_timezone.utc)
    except FileNotFoundError:
        pass

    last_train_task = (
        Task.objects.filter(func="ratings.tasks.run_train").order_by("-stopped").first()
    )
    last_train_info: dict | None = None
    if last_train_task is not None:
        result = (
            last_train_task.result if isinstance(last_train_task.result, dict) else {}
        )
        # success=True from django-q only means the worker returned without raising —
        # run_train catches its own exceptions and returns {"ok": False, "error": ...},
        # so the in-app notion of "succeeded" needs both flags.
        ok = bool(last_train_task.success and result.get("ok", True))
        error = None
        if not ok:
            if isinstance(result, dict) and result.get("error"):
                error = str(result["error"])
            elif last_train_task.result:
                error = str(last_train_task.result)[:500]
            else:
                error = "Task exited without a result."
        last_train_info = {
            "ok": ok,
            "stopped": last_train_task.stopped,
            "started": last_train_task.started,
            "error": error,
        }

    # Exclude purged rows, matching every other list view (gallery, below_cutoff,
    # _review_qs, _counts) so purged images don't skew the charts and counters.
    scored_qs = Image.objects.filter(score__isnull=False, is_purged=False)
    if not show_nsfw:
        scored_qs = scored_qs.filter(is_nsfw=False)

    # The "Gallery" headline and tagging stats mirror the gallery page (score >= 1,
    # i.e. trash excluded); the distribution chart below still shows the 0 bucket.
    gallery_qs = scored_qs.filter(score__gte=1)
    gallery_total = gallery_qs.count()
    # Disjoint from below_cutoff_count (score <= 2) and matches the trainer's split,
    # so the "Training data" line on the page doesn't double-count scores 0-2.
    liked_count = scored_qs.filter(score__gte=3).count()

    score_dist = list(
        scored_qs.values("score").annotate(n=Count("content_hash")).order_by("-score")
    )
    score_dist_max = max((row["n"] for row in score_dist), default=1)

    source_breakdown = list(
        scored_qs.values("source_label")
        .annotate(n=Count("content_hash"))
        .order_by("-n")
    )

    seven_days_ago = timezone.now() - timedelta(days=7)
    scraped_7d = Image.objects.filter(
        downloaded_at__gte=seven_days_ago, is_purged=False
    ).count()
    rated_7d = Image.objects.filter(
        rated_at__gte=seven_days_ago, score__isnull=False, is_purged=False
    ).count()

    tag_breakdown = list(
        Tag.objects.annotate(n=Count("images")).filter(n__gt=0).order_by("-n")[:20]
    )

    tagged_count = gallery_qs.filter(tags__isnull=False).distinct().count()
    untagged_count = gallery_total - tagged_count

    # Compute average inbox dwell time in Python — SQLite doesn't aggregate
    # timedeltas natively and fetching (downloaded_at, rated_at) pairs is
    # cheap at corpus scale.
    inbox_durations = list(
        Image.objects.filter(
            score__isnull=False, rated_at__isnull=False, is_purged=False
        ).values_list("downloaded_at", "rated_at")
    )
    avg_inbox_hours: int | None = None
    valid_durations = [(r - d).total_seconds() for d, r in inbox_durations if r > d]
    if valid_durations:
        avg_inbox_hours = round(sum(valid_durations) / len(valid_durations) / 3600)

    return render(
        request,
        "ratings/stats.html",
        {
            **counts,
            **_training_ctx(request),
            **_scrape_ctx(request),
            "show_nsfw": show_nsfw,
            "last_trained": last_trained,
            "last_train": last_train_info,
            "gallery_total": gallery_total,
            "liked_count": liked_count,
            "score_dist": score_dist,
            "score_dist_max": score_dist_max,
            "source_breakdown": source_breakdown,
            "scraped_7d": scraped_7d,
            "rated_7d": rated_7d,
            "tag_breakdown": tag_breakdown,
            "tagged_count": tagged_count,
            "untagged_count": untagged_count,
            "avg_inbox_hours": avg_inbox_hours,
        },
    )


@login_required
@require_POST
def trigger_scrape(request):
    """
    Enqueue a scrape via django-q and return a polling fragment.

    Scraping used to run synchronously in the request, on the assumption it was
    "fast enough for a normal request timeout." That stopped being true once the
    4chan scraper switched from reading two index pages to fetching every live
    thread — one rate-limited request per thread (~1s each), so a multi-board
    scrape now runs for many minutes. A synchronous request would blow past
    gunicorn's --timeout (300s) and the worker would be SIGKILLed mid-scrape. So,
    like training, it now runs in a background worker and the UI polls
    scrape_status. The session stores the task ID for the poller to watch.
    """
    from django_q.tasks import async_task, fetch

    existing_id = request.session.get("scrape_task_id")
    if existing_id:
        task = fetch(existing_id)
        if task is not None and task.stopped is None:
            ctx = _scrape_ctx(request)
            return render(
                request,
                "ratings/_scrape_pending.html",
                {
                    "task_id": ctx["active_scrape_task_id"],
                    "elapsed": ctx["scrape_elapsed"],
                },
            )
        _clear_scrape_session(request, existing_id)

    task_id = async_task("ratings.tasks.run_scrape")
    request.session["scrape_task_id"] = task_id
    request.session["scrape_started_at"] = timezone.now().isoformat()
    return render(
        request, "ratings/_scrape_pending.html", {"task_id": task_id, "elapsed": "0s"}
    )


@login_required
def scrape_status(request, task_id: str):
    """
    Polling endpoint for the active scrape job (mirror of train_status).

    Returns a "pending" fragment while the worker runs and a "result" fragment
    once it completes; the session entry is cleared on completion. fetch()
    returns None while the task is still queued/running in OrmQ, so a None result
    is only treated as a lost worker after the cluster timeout (14400s).
    """
    from django_q.tasks import fetch

    session_task = request.session.get("scrape_task_id")
    task = fetch(task_id)

    if task is None:
        if session_task == task_id:
            elapsed = _elapsed_from_session(request, "scrape")
            if elapsed is None or elapsed > 14400:
                _clear_scrape_session(request, task_id)
                return render(
                    request,
                    "ratings/_scrape_result.html",
                    {
                        "ok": False,
                        "error": "Scrape task not found (worker may have restarted).",
                    },
                )
            return render(
                request,
                "ratings/_scrape_pending.html",
                {"task_id": task_id, "elapsed": _fmt_elapsed(elapsed)},
            )
        elapsed = _elapsed_from_session(request, "scrape") if session_task else None
        return render(
            request,
            "ratings/_scrape_pending.html",
            {"task_id": session_task or task_id, "elapsed": _fmt_elapsed(elapsed)},
        )

    if task.stopped is None:
        elapsed = (
            _elapsed_from_session(request, "scrape")
            if session_task == task_id
            else None
        )
        return render(
            request,
            "ratings/_scrape_pending.html",
            {"task_id": task_id, "elapsed": _fmt_elapsed(elapsed)},
        )

    if session_task == task_id:
        _clear_scrape_session(request, task_id)

    result = task.result if isinstance(task.result, dict) else {}
    # success=True from django-q only means the worker returned without raising;
    # run_scrape catches its own exceptions and returns {"ok": False, ...}, so a
    # genuine success needs both flags (matches the stats-page train logic).
    ok = bool(task.success and result.get("ok", True))
    if ok:
        return render(
            request,
            "ratings/_scrape_result.html",
            {
                "ok": True,
                "total": result.get("total", 0),
                "counts": result.get("counts", {}),
            },
        )
    if result.get("error"):
        error = str(result["error"])
    elif task.result:
        error = str(task.result)[:500]
    else:
        error = "Task exited without a result."
    return render(
        request, "ratings/_scrape_result.html", {"ok": False, "error": error}
    )


@login_required
@require_POST
def trigger_train(request):
    """
    Enqueue a training job via django-q and return a polling fragment.

    Training blocks for several minutes (DINOv2 encoding + LogReg fit), so it
    runs in a background worker. The session stores the task ID so the polling
    template knows which job to watch via train_status.
    """
    from django_q.tasks import async_task, fetch

    existing_id = request.session.get("training_task_id")
    if existing_id:
        task = fetch(existing_id)
        if task is not None and task.stopped is None:
            ctx = _training_ctx(request)
            return render(
                request,
                "ratings/_train_pending.html",
                {
                    "task_id": ctx["active_task_id"],
                    "elapsed": ctx["training_elapsed"],
                },
            )
        _clear_training_session(request, existing_id)

    task_id = async_task("ratings.tasks.run_train")
    request.session["training_task_id"] = task_id
    request.session["training_started_at"] = timezone.now().isoformat()
    return render(
        request, "ratings/_train_pending.html", {"task_id": task_id, "elapsed": "0s"}
    )


@login_required
def train_status(request, task_id: str):
    """
    Polling endpoint for the active training job.

    Returns a "pending" fragment while the worker is running, and a "result"
    fragment once it completes. The session entry is cleared on completion so
    a subsequent visit to stats doesn't show a stale training indicator.
    """
    from django_q.tasks import fetch

    session_task = request.session.get("training_task_id")
    task = fetch(task_id)

    if task is None:
        if session_task == task_id:
            elapsed = _elapsed_from_session(request)
            if elapsed is None or elapsed > 14400:
                # Elapsed time meets or exceeds the cluster timeout — worker is gone.
                _clear_training_session(request, task_id)
                return render(
                    request,
                    "ratings/_train_result.html",
                    {
                        "ok": False,
                        "error": "Training task not found (worker may have restarted).",
                    },
                )
            # Task not yet in django_q_task → still queued or running in OrmQ.
            return render(
                request,
                "ratings/_train_pending.html",
                {
                    "task_id": task_id,
                    "elapsed": _fmt_elapsed(elapsed),
                },
            )
        elapsed = _elapsed_from_session(request) if session_task else None
        return render(
            request,
            "ratings/_train_pending.html",
            {
                "task_id": session_task or task_id,
                "elapsed": _fmt_elapsed(elapsed),
            },
        )

    if task.stopped is None:
        elapsed = _elapsed_from_session(request) if session_task == task_id else None
        return render(
            request,
            "ratings/_train_pending.html",
            {
                "task_id": task_id,
                "elapsed": _fmt_elapsed(elapsed),
            },
        )

    if session_task == task_id:
        _clear_training_session(request, task_id)

    result = task.result or {}
    return render(
        request,
        "ratings/_train_result.html",
        {
            "ok": result.get("ok", False),
            "error": result.get("error", "Unknown error."),
            # Disjoint counts matching the trainer's split: positives (score>=3)
            # vs below-cutoff negatives (score<=2). Together they're the full
            # training set, so the "+" in the message doesn't double-count.
            "liked_n": Image.objects.filter(score__gte=3).count(),
            "below_cutoff_n": Image.objects.filter(score__lte=2).count(),
        },
    )


def _vision_ctx() -> dict:
    """Threshold dial state for the /config page (DB singleton via get_or_create)."""
    sfw, nsfw = get_review_thresholds()
    return {
        "vision_sfw": sfw,
        "vision_nsfw": nsfw,
        "vision_buckets": range(1, 7),
    }


@login_required
@require_POST
def set_vision_thresholds(request):
    """
    Persist the SFW/NSFW review-queue hide thresholds from the config page.

    Both values are clamped to [1, 6] so a malformed POST can't disable the
    review queue with an out-of-range value. update_or_create writes the
    singleton in one statement.
    """

    def _clamp(name: str) -> int:
        try:
            return max(1, min(6, int(request.POST.get(name, 1))))
        except (ValueError, TypeError):
            return 1

    sfw = _clamp("sfw_threshold")
    nsfw = _clamp("nsfw_threshold")
    ReviewThresholds.objects.update_or_create(
        pk=1, defaults={"sfw_threshold": sfw, "nsfw_threshold": nsfw}
    )
    show_nsfw = request.session.get("show_nsfw", False)
    return render(
        request,
        "ratings/_vision_thresholds_htmx.html",
        {
            **_vision_ctx(),
            **_counts(show_nsfw),
            "show_nsfw": show_nsfw,
            "mode": "config",
            **_training_ctx(request),
        },
    )


def _schedule_ctx() -> dict:
    """
    Build schedule context by joining the user-facing ScrapeSchedule row with the
    live django-q Schedule entry. They are kept as separate records because
    ScrapeSchedule stores user intent while the q-schedule stores the actual
    next_run timestamp and worker state.
    """
    from django_q.models import Schedule as QSchedule

    schedule = ScrapeSchedule.objects.filter(pk=1).first()
    q = QSchedule.objects.filter(name="auto_scrape").first()
    return {
        "schedule": schedule,
        "next_run": q.next_run if q else None,
        "interval_choices": _INTERVAL_CHOICES,
    }


@login_required
def config_view(request):
    """Render the configuration page combining sources, schedule, channels, and training status."""
    show_nsfw = request.session.get("show_nsfw", False)
    counts = _counts(show_nsfw)
    return render(
        request,
        "ratings/config.html",
        {
            "sources": Source.objects.all(),
            "show_nsfw": show_nsfw,
            **_training_ctx(request),
            **counts,
            **_schedule_ctx(),
            **_vision_ctx(),
            **_channel_list_ctx(),
        },
    )


@login_required
@require_POST
def set_scrape_schedule(request):
    """
    Persist scrape schedule settings and sync the django-q cron entry.

    ScrapeSchedule (pk=1 singleton) stores user intent; sync_scrape_q_schedule
    then creates or updates the actual django-q Schedule record so the worker
    picks up the new interval without a restart.
    """
    try:
        interval_hours = max(1, min(168, int(request.POST.get("interval_hours", 6))))
    except (ValueError, TypeError):
        interval_hours = 6
    enabled = request.POST.get("enabled") == "1"

    ScrapeSchedule.objects.update_or_create(
        pk=1,
        defaults={"interval_hours": interval_hours, "enabled": enabled},
    )

    from ratings.schedule_sync import sync_scrape_q_schedule

    sync_scrape_q_schedule()

    return render(request, "ratings/_schedule_status.html", _schedule_ctx())


@login_required
@require_POST
def source_add(request):
    """Validate and create a new scrape source, re-enabling it if previously disabled."""
    stype = request.POST.get("type", "").strip()
    name = request.POST.get("name", "").strip()

    if stype not in dict(Source.TYPE_CHOICES):
        return render(
            request, "ratings/_source_error.html", {"error": "Invalid source type."}
        )
    if not name:
        return render(
            request, "ratings/_source_error.html", {"error": "Name is required."}
        )
    # Mastodon and Pixelfed both target an account via the same @user@instance
    # handle (both speak the Mastodon-compatible API), so they validate alike.
    if stype in (Source.MASTODON, Source.PIXELFED) and "@" not in name.lstrip("@"):
        service = dict(Source.TYPE_CHOICES)[stype]
        host = "pixelfed.social" if stype == Source.PIXELFED else "mastodon.social"
        return render(
            request,
            "ratings/_source_error.html",
            {"error": f"{service} handle must include an instance, e.g. @user@{host}"},
        )

    source, created = Source.objects.get_or_create(type=stype, name=name)
    if not created:
        source.enabled = True
        source.save(update_fields=["enabled"])
    return render(request, "ratings/_source_row.html", {"source": source})


@login_required
@require_POST
def source_toggle(request, pk):
    """Toggle the enabled flag on a source without removing its history."""
    source = get_object_or_404(Source, pk=pk)
    source.enabled = not source.enabled
    source.save(update_fields=["enabled"])
    return render(request, "ratings/_source_row.html", {"source": source})


@login_required
@require_POST
def source_delete(request, pk):
    """Permanently remove a source; past scraped images are unaffected."""
    get_object_or_404(Source, pk=pk).delete()
    return HttpResponse("")


@login_required
@require_POST
def source_import(request):
    """Bulk-import sources from config.toml, creating only records not already in the DB."""
    from ratings.scraper import import_from_config

    n = import_from_config(Path(settings.CONFIG_PATH))
    return render(
        request,
        "ratings/_source_list.html",
        {"sources": Source.objects.all(), "imported": n},
    )


_LOG_SOURCES = {"scrape", "train"}


def _log_source_filter(request) -> str | None:
    """Return the ?source= filter if it matches a known source, else None for 'all'."""
    src = request.GET.get("source", "").strip().lower()
    return src if src in _LOG_SOURCES else None


@login_required
def logs_page(request):
    """
    Show the 500 most recent log entries, optionally filtered by source.

    Source filter is a query param (?source=train|scrape) so it survives an
    HTMX swap of the entries fragment — the polling endpoint reads the same
    value and only returns entries matching the active source.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    source = _log_source_filter(request)
    qs = LogEntry.objects.all()
    if source:
        qs = qs.filter(source=source)
    entries = list(qs.order_by("-pk")[:500])
    next_since = entries[0].pk if entries else 0
    return render(
        request,
        "ratings/logs.html",
        {
            **_counts(show_nsfw),
            **_training_ctx(request),
            "show_nsfw": show_nsfw,
            "entries": entries,
            "next_since": next_since,
            "active_source": source or "all",
        },
    )


@login_required
def log_entries(request):
    """Polling endpoint for log updates; returns only entries newer than since_id."""
    try:
        since_id = int(request.GET.get("since", 0))
    except (ValueError, TypeError):
        since_id = 0
    source = _log_source_filter(request)
    qs = LogEntry.objects.filter(pk__gt=since_id)
    if source:
        qs = qs.filter(source=source)
    entries = list(qs.order_by("-pk")[:100])
    next_since = entries[-1].pk if entries else since_id
    return render(
        request,
        "ratings/_log_entries.html",
        {
            "entries": entries,
            "next_since": next_since,
            "active_source": source or "all",
        },
    )


@login_required
@require_POST
def log_clear(request):
    """Truncate all log entries — useful before a scrape to keep the log view clean."""
    LogEntry.objects.all().delete()
    return redirect("logs")


# ── Browse context helper ─────────────────────────────────────────────────────


def _browse_ctx(
    qs,
    content_hash: str | None,
    mode: str,
    show_nsfw: bool,
    request,
    extra: dict | None = None,
) -> dict:
    """
    Build browse context for a single image, with prev/next navigation when the
    image is part of the queue.

    The full hash list is materialised once so prev/next positions are computed
    from the same snapshot. Fetching prev/next lazily with separate queries
    risks a race condition where an image is rated (and removed from the queue)
    between calls, shifting the navigation offsets.

    A requested content_hash that is NOT in the queue (e.g. an already-scored
    image opened from the gallery for re-review) is still shown — standalone,
    with no queue position and prev/next disabled — instead of silently
    falling back to the head of the queue and showing the wrong picture.
    """
    all_hashes = list(qs.values_list("content_hash", flat=True))
    base = {"show_nsfw": show_nsfw, **_counts(show_nsfw)}
    if extra:
        base.update(extra)

    if content_hash and content_hash not in set(all_hashes):
        image = (
            Image.objects.filter(content_hash=content_hash)
            .prefetch_related("tags")
            .first()
        )
        if image is not None:
            ctx: dict = {
                "image": image,
                "prev_hash": None,
                "next_hash": None,
                "position": None,
                "total": None,
                "mode": mode,
                "prediction": _taste_prediction(image),
                **base,
            }
            if request is not None:
                ctx.update(_training_ctx(request))
            return ctx
        # Requested image is gone (purged/deleted) — fall through to the queue.

    if not all_hashes:
        ctx = {"image": None, "mode": mode, **base}
        if request is not None:
            ctx.update(_training_ctx(request))
        return ctx

    hash_index = {h: i for i, h in enumerate(all_hashes)}
    idx = hash_index.get(content_hash, 0) if content_hash else 0

    image = (
        Image.objects.prefetch_related("tags").get(content_hash=all_hashes[idx])
    )
    ctx = {
        "image": image,
        "prev_hash": all_hashes[idx - 1] if idx > 0 else None,
        "next_hash": all_hashes[idx + 1] if idx < len(all_hashes) - 1 else None,
        "position": idx + 1,
        "total": len(all_hashes),
        "mode": mode,
        "prediction": _taste_prediction(image),
        **base,
    }
    if request is not None:
        ctx.update(_training_ctx(request))
    return ctx


# ── Corpus review ─────────────────────────────────────────────────────────────


def _review_qs(show_nsfw: bool = False):
    """
    Build the ordered queue for the primary review flow.

    Unscored images (score IS NULL) feed the queue. Unseen images
    (queue_seen_at IS NULL) sort first in SQLite ASC; then oldest-downloaded.
    The [vision] threshold further hides low-confidence images so the user
    only reviews things the model thinks they'll like.
    """
    sfw_bucket, nsfw_bucket = get_review_thresholds()
    return (
        Image.objects.filter(score__isnull=True, is_purged=False)
        .filter(visibility_q(sfw_bucket, nsfw_bucket, show_nsfw))
        .order_by("queue_seen_at", "downloaded_at")
    )


def _review_nsfw_qs(show_nsfw: bool = False):
    """
    Same queue shape as _review_qs but filtered to NSFW images only.

    show_nsfw is accepted but ignored — this queue is always NSFW-only by
    definition. The parameter exists so qs_fn callers can treat both queues
    with the same (show_nsfw: bool) → QuerySet signature.
    """
    _, nsfw_bucket = get_review_thresholds()
    return (
        Image.objects.filter(is_nsfw=True, score__isnull=True, is_purged=False)
        .filter(pred_score_visible(bucket_to_cutoff(nsfw_bucket)))
        .order_by("queue_seen_at", "downloaded_at")
    )


def _review_ctx(
    content_hash: str | None, show_nsfw: bool = False, request=None
) -> dict:
    """Build browse context for the primary corpus review queue."""
    return _browse_ctx(
        _review_qs(show_nsfw),
        content_hash,
        "corpus",
        show_nsfw,
        request,
        extra={
            "scores": range(1, 7),
            "score_url": "score_corpus",
            "purge_url": "purge_corpus",
            "image_url": "review_corpus_image",
        },
    )


def _review_nsfw_ctx(
    content_hash: str | None, show_nsfw: bool = False, request=None
) -> dict:
    """Build browse context for the NSFW corpus review queue."""
    return _browse_ctx(
        _review_nsfw_qs(),
        content_hash,
        "nsfw_corpus",
        show_nsfw,
        request,
        extra={
            "scores": range(1, 7),
            "score_url": "score_nsfw_corpus",
            "purge_url": "purge_nsfw_corpus",
            "image_url": "review_nsfw_corpus_image",
        },
    )


def _mark_queue_seen(image: Image | None) -> None:
    """Stamp queue_seen_at once; unseen images sort first in the review queue."""
    if image is not None and image.queue_seen_at is None:
        image.queue_seen_at = timezone.now()
        image.save(update_fields=["queue_seen_at"])


@login_required
def review_corpus(request, content_hash: str | None = None):
    """
    Main corpus review page — full render on first visit, HTMX partial on navigation.

    queue_seen_at is stamped here (not in the queryset) so the unseen-first
    ordering persists across page loads: once you've seen an image it drops to
    the back of the queue only after you move away from it.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    ctx = _review_ctx(content_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    if request.htmx:
        return render(request, "ratings/_review_htmx.html", ctx)
    return render(request, "ratings/review.html", ctx)


# ── Shared corpus action helpers ──────────────────────────────────────────────


def _score_impl(request, content_hash: str, qs_fn, ctx_fn):
    """
    Shared score logic for both the normal and NSFW review queues.

    Scoring is allowed on any image, not just unscored ones, so an already-rated
    image opened from the gallery for re-review can be re-scored here. Navigation
    differs by origin: an in-queue image advances to its neighbour (rate-and-
    advance); an out-of-queue image (re-review) stays put so the user sees the
    updated score instead of being thrown into the rate queue. next_hash is
    captured before scoring because scoring changes which images are in-queue.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    queue_hashes = list(qs_fn(show_nsfw).values_list("content_hash", flat=True))
    if content_hash in queue_hashes:
        next_hash = _neighbor_hash(queue_hashes, content_hash)
    else:
        next_hash = content_hash
    try:
        # 0 is "trash" — below the 1-6 scale; still a real rating, so it leaves
        # the queue and counts as the strongest negative training sample.
        score_val = int(request.POST.get("score", -1))
        if 0 <= score_val <= 6:
            image.score = score_val
            image.rated_at = timezone.now()
            image.save(update_fields=["score", "rated_at"])
            _invalidate_similar_index()
    except (ValueError, TypeError):
        pass
    ctx = ctx_fn(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


def _purge_impl(request, content_hash: str, qs_fn, ctx_fn):
    """
    Shared purge logic for both the normal and NSFW review queues.

    The neighbour is captured first so we know where to navigate after the
    image is hard-deleted from disk.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    next_hash = _neighbor_hash(
        list(qs_fn(show_nsfw).values_list("content_hash", flat=True)),
        content_hash,
    )
    _purge_image(image)
    ctx = ctx_fn(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


@login_required
@require_POST
def score_corpus(request, content_hash: str):
    return _score_impl(request, content_hash, _review_qs, _review_ctx)


@login_required
@require_POST
def purge_corpus(request, content_hash: str):
    return _purge_impl(request, content_hash, _review_qs, _review_ctx)


@login_required
@require_POST
def score_nsfw_corpus(request, content_hash: str):
    return _score_impl(request, content_hash, _review_nsfw_qs, _review_nsfw_ctx)


@login_required
@require_POST
def purge_nsfw_corpus(request, content_hash: str):
    return _purge_impl(request, content_hash, _review_nsfw_qs, _review_nsfw_ctx)


@login_required
@require_POST
def toggle_nsfw(request, content_hash: str):
    """
    Toggle is_nsfw on an image, then navigate appropriately for the current mode.

    Navigation mirrors _score_impl: an in-queue image advances to its neighbour
    when the toggle removes it from the current view; an out-of-queue image (an
    already-scored picture opened from the gallery for re-review) stays put so
    the user keeps seeing it instead of being teleported to the head of the rate
    queue by _neighbor_hash's not-in-queue fallback.

    - Normal queue: marking NSFW only removes the image when show_nsfw is False
      (otherwise it stays visible in the queue).
    - NSFW queue: marking safe removes the image — it's by definition no longer
      in the NSFW queue regardless of show_nsfw.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    mode = request.POST.get("mode", "corpus")

    if mode == "nsfw_corpus":
        queue_hashes = list(_review_nsfw_qs().values_list("content_hash", flat=True))
        in_queue = content_hash in queue_hashes
        neighbor = _neighbor_hash(queue_hashes, content_hash)
        image.is_nsfw = not image.is_nsfw
        image.save(update_fields=["is_nsfw"])
        # In-queue + marked safe → it left the NSFW queue, so advance.
        # Out-of-queue (re-review) or still NSFW → stay on the image.
        target = neighbor if (in_queue and not image.is_nsfw) else content_hash
        ctx = _review_nsfw_ctx(target, show_nsfw, request)
    else:
        queue_hashes = list(
            _review_qs(show_nsfw).values_list("content_hash", flat=True)
        )
        in_queue = content_hash in queue_hashes
        neighbor = _neighbor_hash(queue_hashes, content_hash)
        image.is_nsfw = not image.is_nsfw
        image.save(update_fields=["is_nsfw"])
        # In-queue + marked NSFW while NSFW is hidden → it left the queue, advance.
        # Out-of-queue (re-review) or still visible → stay on the image.
        if in_queue and not show_nsfw and image.is_nsfw:
            target = neighbor
        else:
            target = content_hash
        ctx = _review_ctx(target, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


@login_required
def below_cutoff(request):
    """
    Safety-net grid for images the model rejected or the user scored ≤ 2.

    Unrated rows with predicted_score below the /config/ dial land here instead
    of vanishing from review. User-rated trash (0) and scores 1–2 stay here too.
    Re-score upward or purge from the gallery lightbox.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    sort = request.GET.get("sort", "newest")
    if sort not in ("newest", "oldest", "random"):
        sort = "newest"
    active_tag = request.GET.get("tag", "").strip().lower()

    sfw_bucket, nsfw_bucket = get_review_thresholds()
    qs = Image.objects.filter(is_purged=False).filter(
        below_cutoff_q(sfw_bucket, nsfw_bucket, show_nsfw)
    )
    if not show_nsfw:
        qs = qs.filter(is_nsfw=False)
    if active_tag:
        qs = qs.filter(tags__name=active_tag)

    if sort == "random":
        qs = qs.order_by("?")
    elif sort == "oldest":
        qs = qs.order_by("downloaded_at")
    else:
        qs = qs.order_by("-downloaded_at")

    images = list(qs.prefetch_related("tags")[:500])
    all_tags = list(Tag.objects.values_list("name", flat=True))

    return render(
        request,
        "ratings/gallery.html",
        {
            **_counts(show_nsfw),
            **_training_ctx(request),
            "images": images,
            "total": len(images),
            "min_score": 1,
            "sort": sort,
            "scores": range(1, 7),
            "show_nsfw": show_nsfw,
            "mode": "below_cutoff",
            "is_below_cutoff": True,
            "all_tags": all_tags,
            "active_tag": active_tag,
        },
    )


@login_required
def gallery(request):
    """
    Scored-image gallery with filter, sort, and tag controls.

    The 500-item cap prevents memory pressure on large collections — the gallery
    renders all images into the DOM at once (no pagination) so an unbounded
    query would cause slow page loads and excessive memory use.
    """
    show_nsfw = request.session.get("show_nsfw", False)

    try:
        min_score = max(1, min(6, int(request.GET.get("min_score", 1))))
    except (ValueError, TypeError):
        min_score = 1

    sort = request.GET.get("sort", "newest")
    if sort not in ("newest", "oldest", "random"):
        sort = "newest"

    active_tag = request.GET.get("tag", "").strip().lower()

    qs = Image.objects.filter(
        score__isnull=False, score__gte=min_score, is_purged=False
    )
    if not show_nsfw:
        qs = qs.filter(is_nsfw=False)
    if active_tag:
        qs = qs.filter(tags__name=active_tag)

    if sort == "random":
        qs = qs.order_by("?")
    elif sort == "oldest":
        qs = qs.order_by("downloaded_at")
    else:
        qs = qs.order_by("-downloaded_at")

    images = list(qs.prefetch_related("tags")[:500])
    all_tags = list(Tag.objects.values_list("name", flat=True))

    return render(
        request,
        "ratings/gallery.html",
        {
            **_counts(show_nsfw),
            **_training_ctx(request),
            "images": images,
            "total": len(images),
            "min_score": min_score,
            "sort": sort,
            "scores": range(1, 7),
            "show_nsfw": show_nsfw,
            "mode": "gallery",
            "all_tags": all_tags,
            "active_tag": active_tag,
        },
    )


@login_required
def tag_autocomplete(request):
    """JSON endpoint for tag name suggestions; filtered by prefix when ?q= is given."""
    q = request.GET.get("q", "").strip().lower()
    qs = Tag.objects.all()
    if q:
        qs = qs.filter(name__startswith=q)
    return JsonResponse({"tags": list(qs.values_list("name", flat=True)[:20])})


@login_required
@require_POST
def update_image_tags(request, content_hash: str):
    """
    Replace all tags on an image with the submitted comma-separated list.

    M2M .set() does a diff internally (removes old, adds new) rather than
    clearing and re-inserting, so this is safe to call repeatedly without
    accumulating duplicates or racing against other requests.
    """
    image = get_object_or_404(Image, content_hash=content_hash)
    tag_str = request.POST.get("tags", "")
    tag_names = [t.strip().lower() for t in tag_str.split(",") if t.strip()]
    tags = [Tag.objects.get_or_create(name=name)[0] for name in tag_names]
    image.tags.set(tags)
    return JsonResponse({"tags": sorted(image.tags.values_list("name", flat=True))})


@login_required
def tag_list(request):
    """List all tags with image counts for management (rename / delete)."""
    tags = list(Tag.objects.annotate(n=Count("images")).order_by("-n", "name"))
    return render(request, "ratings/tags.html", {"tags": tags, **_counts()})


@login_required
def tag_rename(request, pk: int):
    """
    Rename a tag in-place.

    GET ?edit=1 swaps the row for an inline edit form.
    POST applies the rename; if the new name already exists the two tags are
    merged — all images from the old tag move to the existing one and the old
    record is deleted — so the user can consolidate typos without manual cleanup.
    """
    tag = get_object_or_404(Tag, pk=pk)
    if request.method == "GET":
        tag.n = tag.images.count()
        tmpl = "_tag_row_edit.html" if request.GET.get("edit") else "_tag_row.html"
        return render(request, f"ratings/{tmpl}", {"tag": tag})
    new_name = request.POST.get("name", "").strip().lower()
    if not new_name or new_name == tag.name:
        tag.n = tag.images.count()
        return render(request, "ratings/_tag_row.html", {"tag": tag})
    existing = Tag.objects.filter(name=new_name).exclude(pk=pk).first()
    if existing:
        # Merge: move all images from the old tag to the existing one, then
        # delete the old tag so there's no duplicate entry in the tag list.
        for image in tag.images.all():
            image.tags.add(existing)
        tag.delete()
        existing.n = existing.images.count()
        return render(request, "ratings/_tag_row.html", {"tag": existing})
    tag.name = new_name
    tag.save(update_fields=["name"])
    tag.n = tag.images.count()
    return render(request, "ratings/_tag_row.html", {"tag": tag})


@login_required
@require_POST
def tag_delete(request, pk: int):
    """Delete a tag and remove it from all images that carry it."""
    get_object_or_404(Tag, pk=pk).delete()
    return HttpResponse("")


@login_required
@require_POST
def gallery_action(request, content_hash: str):
    """
    Handle inline score/nsfw/purge actions from the gallery grid.

    Returns JSON so the gallery JS can update the card in-place without a full
    page reload. Purge returns {"deleted": True} as a signal to remove the card
    from the DOM. Purge is a hard-delete — distinct from review's score-0
    "trash", which keeps the file as a strong negative training example.
    """
    action = request.POST.get("action", "")
    image = get_object_or_404(Image, content_hash=content_hash)
    now = timezone.now()

    if action == "purge":
        _purge_image(image)
        return JsonResponse({"deleted": True})

    if action == "score":
        try:
            # 0 = trash (strongest negative); -1 default keeps a missing param a no-op.
            score_val = int(request.POST.get("score", -1))
            if 0 <= score_val <= 6:
                image.score = score_val
                image.rated_at = now
                image.save(update_fields=["score", "rated_at"])
        except (ValueError, TypeError):
            pass
    elif action == "nsfw":
        image.is_nsfw = not image.is_nsfw
        image.save(update_fields=["is_nsfw"])

    return JsonResponse({"score": image.score, "nsfw": image.is_nsfw})


@login_required
@require_POST
def share_image(request, content_hash):
    """
    Share an image to one or more named NotificationChannels.

    The caller POSTs a list of channel PKs (channels[]=1&channels[]=3).
    Sending is synchronous — both APIs are expected to be on the same LAN so
    latency is negligible. The image bytes are uploaded directly because
    /media/ is @login_required and therefore unreachable for message recipients.
    """
    image = get_object_or_404(Image, content_hash=content_hash)
    image_path = DATA_DIR / image.file_path
    selected_pks = request.POST.getlist("channels")
    channels = NotificationChannel.objects.filter(pk__in=selected_pks, enabled=True)
    sent, errors = [], []
    for ch in channels:
        try:
            if ch.service == NotificationChannel.MATTERMOST:
                notifiers.send_to_mattermost(ch, image_path, image.source_label or "")
            elif ch.service == NotificationChannel.SIGNAL:
                notifiers.send_to_signal(ch, image_path, image.source_label or "")
            sent.append(ch.name)
        except Exception as exc:
            errors.append(f"{ch.name}: {exc}")
    return render(
        request, "ratings/_share_toast.html", {"sent": sent, "errors": errors}
    )


def _channel_list_ctx() -> dict:
    """Channel list context for the config page sharing section."""
    return {"channels": list(NotificationChannel.objects.all())}


@login_required
@require_POST
def channel_add(request):
    """Create a new NotificationChannel from the config page form.

    The form targets the inline #channel-add-error slot. On success the refreshed
    list is returned as an out-of-band swap (so #channel-list updates while the
    empty main body clears any prior error); validation failures render into the
    slot without disturbing the existing list. Names are unique, so a duplicate is
    rejected outright rather than silently returning a channel of the wrong service
    (get_or_create would ignore the chosen service for an existing name).
    """
    name = request.POST.get("name", "").strip()
    service = request.POST.get("service", "").strip()
    if not name:
        return render(
            request, "ratings/_channel_error.html", {"error": "Name is required."}
        )
    if service not in dict(NotificationChannel.SERVICE_CHOICES):
        return render(
            request, "ratings/_channel_error.html", {"error": "Invalid service."}
        )
    if NotificationChannel.objects.filter(name=name).exists():
        return render(
            request,
            "ratings/_channel_error.html",
            {"error": f"A channel named “{name}” already exists."},
        )
    NotificationChannel.objects.create(name=name, service=service)
    return render(
        request,
        "ratings/_channel_list.html",
        {**_channel_list_ctx(), "oob": True},
    )


@login_required
@require_POST
def channel_save(request, pk: int):
    """Persist credential fields for an existing channel."""
    ch = get_object_or_404(NotificationChannel, pk=pk)
    ch.name = request.POST.get("name", ch.name).strip() or ch.name
    if ch.service == NotificationChannel.MATTERMOST:
        ch.mm_base_url = request.POST.get("mm_base_url", "").strip()
        ch.mm_token = request.POST.get("mm_token", "").strip()
        ch.mm_channel_id = request.POST.get("mm_channel_id", "").strip()
        ch.mm_message_prefix = request.POST.get("mm_message_prefix", "").strip()
    elif ch.service == NotificationChannel.SIGNAL:
        ch.signal_api_url = request.POST.get("signal_api_url", "").strip()
        ch.signal_sender = request.POST.get("signal_sender", "").strip()
        ch.signal_recipients = request.POST.get("signal_recipients", "").strip()
        ch.signal_message_prefix = request.POST.get("signal_message_prefix", "").strip()
    ch.save()
    return render(request, "ratings/_channel_row.html", {"channel": ch})


@login_required
@require_POST
def channel_toggle(request, pk: int):
    """Toggle enabled on a channel without removing its credentials."""
    ch = get_object_or_404(NotificationChannel, pk=pk)
    ch.enabled = not ch.enabled
    ch.save(update_fields=["enabled"])
    return render(request, "ratings/_channel_row.html", {"channel": ch})


@login_required
@require_POST
def channel_delete(request, pk: int):
    """Permanently delete a notification channel."""
    get_object_or_404(NotificationChannel, pk=pk).delete()
    return HttpResponse("")
