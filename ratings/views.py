import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

import ratings.notifiers as notifiers
from core.brain import EncoderUnavailableError
from ratings import features, reset, search, similar
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
    QueueOrder,
    below_cutoff_q,
    bucket_to_cutoff,
    get_queue_order,
    get_review_thresholds,
    normalize_queue_order,
    pred_score_visible,
    review_settings_row,
    visibility_q,
)
from ratings.toast import with_toast
from ratings.utils import purge_image as _purge_image_util

_INTERVAL_CHOICES = [1, 2, 4, 6, 12, 24, 48, 72, 168]

WEIGHTS_PATH = Path(settings.WEIGHTS_PATH)
DATA_DIR = Path(settings.DATA_DIR)

_taste_model_cache = None
_taste_model_mtime: float | None = None

def _get_taste_model():
    """
    Load (and cache) the taste model, reloading if weights change on disk.

    The cached value may legitimately be None: load_taste_model returns None
    for a file trained on another encoder (V5). Keying the reload on the file's
    mtime alone, rather than on "cache is None", keeps that None from being
    unpickled again on every render. A retrain writes a new file, so its new
    per-source dict arrives with the next mtime change.
    """
    global _taste_model_cache, _taste_model_mtime
    if not WEIGHTS_PATH.exists():
        return None
    mtime = WEIGHTS_PATH.stat().st_mtime
    if mtime != _taste_model_mtime:
        from core import taste

        _taste_model_cache = taste.load_taste_model(WEIGHTS_PATH)
        _taste_model_mtime = mtime
    return _taste_model_cache


def _taste_prediction(image) -> int | None:
    """Return P(corpus) as integer percentage 0–100, or None if unavailable.

    Prefers the stored ``predicted_score`` — classify_images() writes it at
    scrape time, so the common case is a single float read. Without this short
    circuit every review/gallery render re-ran predict_proba and copied the
    ~3 KB embedding blob via ``bytes(image.embedding)``, dominating the
    per-page cost. Legacy rows (embedded before a classifier existed) take
    the fallback once and persist the result, so subsequent renders are fast.
    """
    if image is None:
        return None
    if image.predicted_score is not None:
        return round(float(image.predicted_score) * 100)
    # A vector from an older encoder must never meet the current classifier
    # (V2), and a row without its SigLIP2 half has no feature yet (taste
    # contract V13); either way the row stays unpredicted until it is encoded.
    if not features.has_taste_features(image):
        return None
    taste_model = _get_taste_model()
    if taste_model is None:
        return None
    from core import brain

    classifier = taste_model.classifier_for(image.source_label)
    prob = float(brain.predict_proba(classifier, features.taste_features(image)))
    image.predicted_score = prob
    image.save(update_fields=["predicted_score"])
    return round(prob * 100)


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


_TRAIN_ETA_RE = re.compile(r"ETA (\d+)s")


def _training_eta(request) -> str | None:
    """Return a human-friendly ETA to training completion, or None.

    django-q runs training in a separate worker process that can't write the
    web session, so the only worker→web channel is the LogEntry stream (the
    same path ``elapsed`` rides). ``core.brain.encode`` logs progress lines
    like "train: 50/200 encoded (12.3 img/s, ETA 12s)"; the encode pass
    dominates training wall-clock, so its ETA is a good proxy for time left.

    Two filters keep the reading honest:
    - ``timestamp__gte`` the run's start stamp, because _trim_logs only drops
      entries >48h, so without it a fresh run would read the *previous* run's
      final "ETA 0s" before logging anything of its own.
    - ``train: `` prefix, so we ignore the later classify_images encode pass
      (logged under "classify_images: ") — otherwise the ETA would count down
      to ~0, training would not finish, then the ETA would jump back up.
    """
    started_str = request.session.get("training_started_at")
    if not started_str:
        return None
    started_at = datetime.fromisoformat(started_str)
    message = (
        LogEntry.objects.filter(
            source="train",
            timestamp__gte=started_at,
            message__startswith="train: ",
            message__contains="ETA ",
        )
        .order_by("-pk")
        .values_list("message", flat=True)
        .first()
    )
    if not message:
        return None
    match = _TRAIN_ETA_RE.search(message)
    return _fmt_elapsed(int(match.group(1))) if match else None


def _train_pending_ctx(request, task_id, elapsed) -> dict:
    """Build the _train_pending.html context with elapsed + parsed ETA.

    Centralised so every render site (the steady-state poll, the OrmQ-pending
    paths, and trigger_train) shows the ETA — sprinkling it inline risks one
    poll path silently dropping it.
    """
    return {"task_id": task_id, "elapsed": elapsed, "eta": _training_eta(request)}


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
    return int((datetime.now(UTC) - started_at).total_seconds())


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


def _queue_neighbor_hash(qs, image: Image, order: QueueOrder) -> str | None:
    """
    Next content_hash after `image` in queue order, falling back to the prior
    one if image is the tail, or None for an empty queue.

    Used by score/purge/toggle for rate-and-advance navigation; callers should
    only invoke this once they've confirmed `image` belongs to `qs` (a one-row
    PK `.exists()` check), otherwise an out-of-queue caller will navigate
    relative to its stale queue position instead of staying put. `order` must
    be the one `qs` was built with, otherwise the window filters split the
    queue differently from how it is sorted.
    """
    prev_filter, next_filter = order.position_filters(image)
    next_hash = (
        qs.filter(next_filter).values_list("content_hash", flat=True).first()
    )
    if next_hash is not None:
        return next_hash
    # The reversed fields walk the queue backwards, so `.first()` here is the
    # immediate predecessor regardless of whether it sits in the unseen block.
    return (
        qs.filter(prev_filter)
        .order_by(*order.reversed_order_fields())
        .values_list("content_hash", flat=True)
        .first()
    )


@login_required
def rate_nsfw_corpus(request, content_hash: str | None = None):
    """
    Entry point for the NSFW corpus review queue — same layout as review_corpus
    but filtered to is_nsfw=True images.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    _mark_left_image_seen(request)
    ctx = _review_nsfw_ctx(content_hash, show_nsfw, request)
    if request.htmx:
        return render(request, "ratings/_review_htmx.html", ctx)
    return render(request, "ratings/review.html", ctx)


@login_required
@require_POST
def nsfw_toggle(request):
    """
    Session toggle for the show-NSFW preference; redirects back to the page
    the switch was pressed on.

    The Referer header is client-supplied, so it is only followed when it
    points at this host (OWASP A01: unvalidated redirect); anything else
    falls back to the index.
    """
    request.session["show_nsfw"] = not request.session.get("show_nsfw", False)
    referer = request.META.get("HTTP_REFERER", "")
    referer_is_local = url_has_allowed_host_and_scheme(
        referer, allowed_hosts={request.get_host()}, require_https=request.is_secure()
    )
    if referer_is_local:
        return redirect(referer)
    return redirect("index")


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
        last_trained = datetime.fromtimestamp(mtime, tz=UTC)
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
    positive_count = scored_qs.filter(score__gte=3).count()

    score_dist = list(
        scored_qs.values("score").annotate(n=Count("content_hash")).order_by("-score")
    )
    score_dist_max = max((row["n"] for row in score_dist), default=1)

    # Per source: how many liked / disliked rows feed the trainer and whether the
    # source has reached its own classifier (taste contract V10). The split uses
    # the trainer's constants so the page never disagrees with the log lines.
    from core.trainer import HIGH_SCORE, LOW_SCORE

    source_breakdown = list(
        scored_qs.values("source_label")
        .annotate(
            n=Count("content_hash"),
            good=Count("content_hash", filter=Q(score__gte=HIGH_SCORE)),
            bad=Count("content_hash", filter=Q(score__lte=LOW_SCORE)),
        )
        .order_by("-n")
    )
    taste_model = _get_taste_model()
    sources_with_own_model = set(taste_model.per_source) if taste_model else set()
    for row in source_breakdown:
        row["own_model"] = row["source_label"] in sources_with_own_model

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
            "show_nsfw": show_nsfw,
            "last_trained": last_trained,
            "last_train": last_train_info,
            "gallery_total": gallery_total,
            "positive_count": positive_count,
            "mode": "stats",
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

    Training blocks for several minutes (DINOv3 encoding + LogReg fit), so it
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
                _train_pending_ctx(
                    request, ctx["active_task_id"], ctx["training_elapsed"]
                ),
            )
        _clear_training_session(request, existing_id)

    task_id = async_task("ratings.tasks.run_train")
    request.session["training_task_id"] = task_id
    request.session["training_started_at"] = timezone.now().isoformat()
    return render(
        request,
        "ratings/_train_pending.html",
        _train_pending_ctx(request, task_id, "0s"),
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
                _train_pending_ctx(request, task_id, _fmt_elapsed(elapsed)),
            )
        elapsed = _elapsed_from_session(request) if session_task else None
        return render(
            request,
            "ratings/_train_pending.html",
            _train_pending_ctx(
                request, session_task or task_id, _fmt_elapsed(elapsed)
            ),
        )

    if task.stopped is None:
        elapsed = _elapsed_from_session(request) if session_task == task_id else None
        return render(
            request,
            "ratings/_train_pending.html",
            _train_pending_ctx(request, task_id, _fmt_elapsed(elapsed)),
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
            "positive_n": Image.objects.filter(score__gte=3).count(),
            "negative_n": Image.objects.filter(score__lte=2).count(),
        },
    )


def _vision_ctx() -> dict:
    """Review-queue form state for the /config page (DB singleton via get_or_create)."""
    row = review_settings_row()
    return {
        "vision_sfw": row.sfw_threshold,
        "vision_nsfw": row.nsfw_threshold,
        "vision_buckets": range(1, 7),
        "queue_order": normalize_queue_order(row.queue_order),
        "queue_order_choices": ReviewThresholds.ORDER_CHOICES,
    }


@login_required
@require_POST
def set_vision_thresholds(request):
    """
    Persist the review-queue settings (SFW/NSFW hide thresholds and queue
    order) from the config page.

    Both thresholds are clamped to [1, 6] so a malformed POST can't disable
    the review queue with an out-of-range value, and the order is matched
    against the whitelist so an unknown name can never reach order_by (OWASP
    A03; a missing or unknown value is stored as the default, contract V9).
    update_or_create writes the singleton in one statement.
    """

    def _clamp(name: str) -> int:
        try:
            return max(1, min(6, int(request.POST.get(name, 1))))
        except (ValueError, TypeError):
            return 1

    sfw = _clamp("sfw_threshold")
    nsfw = _clamp("nsfw_threshold")
    queue_order = normalize_queue_order(request.POST.get("queue_order"))
    ReviewThresholds.objects.update_or_create(
        pk=1,
        defaults={
            "sfw_threshold": sfw,
            "nsfw_threshold": nsfw,
            "queue_order": queue_order,
        },
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
            "mode": "config",
            "sources": Source.objects.all(),
            "show_nsfw": show_nsfw,
            **counts,
            **_schedule_ctx(),
            **_vision_ctx(),
            **_channel_list_ctx(),
            **_index_status_ctx(),
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
    return with_toast(HttpResponse(""), "Source deleted")


@login_required
@require_POST
def source_import(request):
    """Bulk-import sources from config.toml, creating only records not already in the DB."""
    from ratings.scraper import import_from_config

    n = import_from_config(Path(settings.CONFIG_PATH))
    response = render(request, "ratings/_source_list.html", {"sources": Source.objects.all()})
    noun = "source" if n == 1 else "sources"
    return with_toast(response, f"Imported {n} new {noun} from config.toml", kind="info")


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
            "show_nsfw": show_nsfw,
            "entries": entries,
            "next_since": next_since,
            "active_source": source or "all",
            "mode": "logs",
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
    messages.success(request, "Log cleared")
    return redirect("logs")


# ── Browse context helper ─────────────────────────────────────────────────────


def _browse_ctx(
    qs,
    content_hash: str | None,
    mode: str,
    show_nsfw: bool,
    request,
    order: QueueOrder,
    extra: dict | None = None,
) -> dict:
    """
    Build browse context for a single image, with prev/next navigation when the
    image is part of the queue.

    Prev/next/position/total are resolved with windowed lookups against the
    queue's (queue_seen_at, secondary key) ordering (see
    `QueueOrder.position_filters`) instead of pulling every queued content_hash
    into Python on each request. `order` must be the one `qs` was sorted with.
    Two `.first()`s and two `.count()`s scale with the queue much better than
    the old materialise-and-index pass.

    Navigation stays correct under concurrent edits: if a sibling is rated or
    purged between requests, the windowed query simply finds whichever real
    neighbour exists at query time. The stronger "consistent snapshot" claim
    of the old code only ever held within a single request anyway.

    A requested content_hash that is NOT in the queue (e.g. an already-scored
    image opened from the gallery for re-review) is still shown — standalone,
    with no queue position and prev/next disabled — instead of silently
    falling back to the head of the queue and showing the wrong picture.
    """
    base = {"show_nsfw": show_nsfw, **_counts(show_nsfw)}
    if extra:
        base.update(extra)

    image = None
    in_queue = False
    if content_hash:
        # Try the queue first; in the common in-queue case this saves the
        # separate "does it exist anywhere?" lookup.
        image = (
            qs.filter(content_hash=content_hash).prefetch_related("tags").first()
        )
        in_queue = image is not None
        if image is None:
            image = (
                Image.objects.filter(content_hash=content_hash)
                .prefetch_related("tags")
                .first()
            )

    if image is not None and not in_queue:
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
        return ctx

    # No content_hash, or the requested image is gone (purged/deleted) — show
    # the head of the queue so the user lands on something predictable.
    if image is None:
        image = qs.prefetch_related("tags").first()
        in_queue = image is not None

    if image is None:
        ctx = {"image": None, "mode": mode, **base}
        return ctx

    prev_filter, next_filter = order.position_filters(image)
    prev_qs = qs.filter(prev_filter)
    next_qs = qs.filter(next_filter)
    prev_hash = (
        prev_qs.order_by(*order.reversed_order_fields())
        .values_list("content_hash", flat=True)
        .first()
    )
    next_hash = next_qs.values_list("content_hash", flat=True).first()
    prev_count = prev_qs.count()
    # Derive total from the two halves + the image itself to skip a third
    # COUNT(*) on the queue.
    total = prev_count + 1 + next_qs.count()
    ctx = {
        "image": image,
        "prev_hash": prev_hash,
        "next_hash": next_hash,
        "position": prev_count + 1,
        "total": total,
        "mode": mode,
        "prediction": _taste_prediction(image),
        **base,
    }
    return ctx


# ── Corpus review ─────────────────────────────────────────────────────────────


def _review_qs(show_nsfw: bool = False, order: QueueOrder | None = None):
    """
    Build the ordered queue for the primary review flow.

    Unscored images (score IS NULL) feed the queue. Unseen images
    (queue_seen_at IS NULL) sort first in SQLite ASC; inside that block the
    configured order (oldest / newest download, or shuffled) decides. The
    [vision] threshold further hides low-confidence images so the user only
    reviews things the model thinks they'll like.

    `order` defaults to the DB setting; callers that also build navigation
    pass the one they already fetched so queue and window filters agree.
    """
    if order is None:
        order = get_queue_order()
    sfw_bucket, nsfw_bucket = get_review_thresholds()
    return (
        Image.objects.filter(score__isnull=True, is_purged=False)
        .filter(visibility_q(sfw_bucket, nsfw_bucket, show_nsfw))
        .order_by(*order.order_fields())
    )


def _review_nsfw_qs(show_nsfw: bool = False, order: QueueOrder | None = None):
    """
    Same queue shape as _review_qs but filtered to NSFW images only.

    show_nsfw is accepted but ignored — this queue is always NSFW-only by
    definition. The parameter exists so qs_fn callers can treat both queues
    with the same (show_nsfw, order) → QuerySet signature.
    """
    if order is None:
        order = get_queue_order()
    _, nsfw_bucket = get_review_thresholds()
    return (
        Image.objects.filter(is_nsfw=True, score__isnull=True, is_purged=False)
        .filter(pred_score_visible(bucket_to_cutoff(nsfw_bucket)))
        .order_by(*order.order_fields())
    )


def _review_ctx(
    content_hash: str | None, show_nsfw: bool = False, request=None
) -> dict:
    """Build browse context for the primary corpus review queue."""
    order = get_queue_order()
    return _browse_ctx(
        _review_qs(show_nsfw, order),
        content_hash,
        "corpus",
        show_nsfw,
        request,
        order,
        extra={
            "scores": range(1, 7),
            "score_url": "score_corpus",
            "nsfw_url": "toggle_nsfw",
            "hx_target": "#review-card",
            "hx_swap": "outerHTML",
            "purge_url": "purge_corpus",
            "image_url": "review_corpus_image",
        },
    )


def _review_nsfw_ctx(
    content_hash: str | None, show_nsfw: bool = False, request=None
) -> dict:
    """Build browse context for the NSFW corpus review queue."""
    order = get_queue_order()
    return _browse_ctx(
        _review_nsfw_qs(show_nsfw, order),
        content_hash,
        "nsfw_corpus",
        show_nsfw,
        request,
        order,
        extra={
            "scores": range(1, 7),
            "score_url": "score_nsfw_corpus",
            "nsfw_url": "toggle_nsfw",
            "hx_target": "#review-card",
            "hx_swap": "outerHTML",
            "purge_url": "purge_nsfw_corpus",
            "image_url": "review_nsfw_corpus_image",
        },
    )


def _mark_left_image_seen(request) -> None:
    """
    Stamp the image the user just moved on from, named by `?left=<hash>` on a
    prev/next navigation, so it drops behind every unseen image (contract V3).

    Stamping on leave rather than on render keeps the card and the DB in step:
    prev/next and the position were computed for the state the row had at
    render time, and rate-and-advance reads that same state afterwards. With
    the earlier stamp-at-render the row moved to the back while its card still
    showed the unseen position, so scoring the first image of a session jumped
    to the far end of the unseen block instead of to its successor. A reload
    of the card is not "moving on", so it no longer skips the image either.

    `left` is client input used only as a parametrised primary-key match
    (OWASP A03); the only effect is an idempotent seen-stamp on an unrated row,
    so a forged value can do no more than re-order the queue.
    """
    left = request.GET.get("left")
    if not left:
        return
    Image.objects.filter(
        content_hash=left, score__isnull=True, queue_seen_at__isnull=True
    ).update(queue_seen_at=timezone.now())


@login_required
def review_corpus(request, content_hash: str | None = None):
    """
    Main corpus review page — full render on first visit, HTMX partial on navigation.

    The image being left (prev/next carry `?left=`) is stamped before the new
    card is built, so the unseen-first ordering persists across page loads and
    the new card already reflects the skipped image's move to the back.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    _mark_left_image_seen(request)
    ctx = _review_ctx(content_hash, show_nsfw, request)
    if request.htmx:
        return render(request, "ratings/_review_htmx.html", ctx)
    return render(request, "ratings/review.html", ctx)


# ── Shared corpus action helpers ──────────────────────────────────────────────


def _score_impl(request, content_hash: str, qs_fn, ctx_fn):
    """
    Shared score logic for both the normal and NSFW review queues.

    Scoring is allowed on any image, not just unscored ones, so an already-rated
    image opened from the gallery for re-review can be re-scored here. Navigation
    differs by origin: rating an *unrated* image is a rate-and-advance, so it
    moves to the queue neighbour; re-scoring an *already-rated* image (gallery
    re-review) stays put so the user sees the updated score instead of being
    thrown into the rate queue. next_hash is captured before scoring because
    scoring changes which images are in-queue.

    The advance/stay decision keys on whether the image was already rated, NOT
    on current queue membership: `_taste_prediction` lazily writes a
    predicted_score while rendering the card, which can drop a just-shown
    unrated image below the vision cutoff and out of the queue. Keying on
    queue membership there would wrongly treat it as a re-review and re-render
    the same image, stalling the rate flow (the user has to tap again).
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    order = get_queue_order()
    qs = qs_fn(show_nsfw, order)
    if image.score is None:
        next_hash = _queue_neighbor_hash(qs, image, order)
    else:
        next_hash = content_hash
    score_val = _score_from_post(request)
    if score_val is not None:
        image.score = score_val
        image.rated_at = timezone.now()
        image.save(update_fields=["score", "rated_at"])
    ctx = ctx_fn(next_hash, show_nsfw, request)
    return render(request, "ratings/_review_htmx.html", ctx)


def _purge_impl(request, content_hash: str, qs_fn, ctx_fn):
    """
    Shared purge logic for both the normal and NSFW review queues.

    The neighbour is captured first so we know where to navigate after the
    image is hard-deleted from disk. An out-of-queue purge (rare — only when
    re-reviewing an already-scored image from the gallery) falls back to the
    head of the queue, preserving the legacy `_neighbor_hash` behaviour.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    order = get_queue_order()
    qs = qs_fn(show_nsfw, order)
    if qs.filter(content_hash=content_hash).exists():
        next_hash = _queue_neighbor_hash(qs, image, order)
    else:
        next_hash = qs.values_list("content_hash", flat=True).first()
    _purge_image(image)
    ctx = ctx_fn(next_hash, show_nsfw, request)
    response = render(request, "ratings/_review_htmx.html", ctx)
    return with_toast(response, "Image purged")


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
    the user keeps seeing it instead of being teleported elsewhere.

    - Normal queue: marking NSFW only removes the image when show_nsfw is False
      (otherwise it stays visible in the queue).
    - NSFW queue: marking safe removes the image — it's by definition no longer
      in the NSFW queue regardless of show_nsfw.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    mode = request.POST.get("mode", "corpus")
    order = get_queue_order()

    if mode == "nsfw_corpus":
        qs = _review_nsfw_qs(show_nsfw, order)
        in_queue = qs.filter(content_hash=content_hash).exists()
        neighbor = _queue_neighbor_hash(qs, image, order) if in_queue else None
        image.is_nsfw = not image.is_nsfw
        image.save(update_fields=["is_nsfw"])
        # In-queue + marked safe → it left the NSFW queue, so advance.
        # Out-of-queue (re-review) or still NSFW → stay on the image.
        target = neighbor if (in_queue and not image.is_nsfw) else content_hash
        ctx = _review_nsfw_ctx(target, show_nsfw, request)
    else:
        qs = _review_qs(show_nsfw, order)
        in_queue = qs.filter(content_hash=content_hash).exists()
        neighbor = _queue_neighbor_hash(qs, image, order) if in_queue else None
        image.is_nsfw = not image.is_nsfw
        image.save(update_fields=["is_nsfw"])
        # In-queue + marked NSFW while NSFW is hidden → it left the queue, advance.
        # Out-of-queue (re-review) or still visible → stay on the image.
        if in_queue and not show_nsfw and image.is_nsfw:
            target = neighbor
        else:
            target = content_hash
        ctx = _review_ctx(target, show_nsfw, request)
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


def _images_in_rank_order(ranked: list[tuple[str, float]]) -> list[Image]:
    """
    Fetch the ranked hashes in one query and put them back into rank order.

    in_bulk is one query for the whole page; a hash purged between ranking
    and fetch is simply absent from the dict and drops out of the page.
    """
    by_hash = Image.objects.prefetch_related("tags").in_bulk([h for h, _ in ranked])
    ordered: list[Image] = []
    for content_hash, _similarity in ranked:
        image = by_hash.get(content_hash)
        if image is not None:
            ordered.append(image)
    return ordered


def _gallery_links(min_score: int, sort: str, active_tag: str, scope: str, mode_params: dict) -> dict:
    """
    Query strings for the gallery template, built here so it never assembles
    a URL from the search text by hand: `sticky` is what every score and tag
    link appends to stay inside the current search or similarity view,
    `scope_urls` switch the scope inside that view, `clear_url` leaves it.
    urlencode quotes the query text, so a `&` or `<` typed into the search box
    cannot split a link or inject markup (OWASP A03).
    """
    base = {"min_score": min_score, "sort": sort}
    if active_tag:
        base["tag"] = active_tag
    if not mode_params:
        return {"sticky": "", "scope_urls": {}, "clear_url": "?" + urlencode(base)}
    scope_urls = {}
    for name in search.SCOPES:
        scope_urls[name] = "?" + urlencode({**base, "scope": name, **mode_params})
    return {
        "sticky": urlencode({"scope": scope, **mode_params}),
        "scope_urls": scope_urls,
        "clear_url": "?" + urlencode(base),
    }


@login_required
def gallery(request):
    """
    Scored-image gallery with filter, sort and tag controls, plus two ranked
    modes on the same grid: a text search (`?q=`, SigLIP2, contract V5) and
    "similar images" (`?similar=<hash>`, DINOv3, V17).

    A ranked mode replaces the sort and caps the page at RESULT_LIMIT; the
    plain gallery keeps its 500-item cap because it renders everything into
    the DOM at once. `scope` widens the candidate set to the unrated backlog
    and only means something inside a ranked mode: the plain gallery is rated
    images by definition. Both ranked modes fall back to the plain gallery
    with a toast when they cannot run (missing weights, an anchor without a
    current vector), never to a silent empty grid (V10, V17). The query text
    reaches only the tokenizer and the template's autoescape (V8).
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
    scope = search.normalize_scope(request.GET.get("scope"))
    query = search.normalize_query(request.GET.get("q"))
    similar_hash = request.GET.get("similar", "").strip()

    anchor = None
    if similar_hash:
        anchor = get_object_or_404(Image, content_hash=similar_hash, is_purged=False)

    candidates = search.candidate_images(scope, min_score, active_tag, show_nsfw)
    ranked: list[tuple[str, float]] = []
    mode_params: dict[str, str] = {}
    unindexed = 0
    if anchor is not None:
        similar_ranked = similar.rank_similar(anchor, candidates, search.RESULT_LIMIT)
        if similar_ranked is None:
            messages.info(request, "This image has no current embedding yet. Run Train first.")
            anchor = None
        else:
            ranked = similar_ranked
            mode_params = {"similar": anchor.content_hash}
    elif query:
        try:
            ranked = search.rank_by_text(query, candidates, search.RESULT_LIMIT)
            unindexed = search.unindexed_count(candidates)
            mode_params = {"q": query}
        except EncoderUnavailableError as exc:
            messages.error(request, str(exc))
            query = ""

    if mode_params:
        images = _images_in_rank_order(ranked)
    else:
        scope = search.DEFAULT_SCOPE
        qs = search.candidate_images(scope, min_score, active_tag, show_nsfw)
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
            "images": images,
            "total": len(images),
            "min_score": min_score,
            "sort": sort,
            "scores": range(1, 7),
            "show_nsfw": show_nsfw,
            "mode": "gallery",
            "all_tags": all_tags,
            "active_tag": active_tag,
            "q": query if "q" in mode_params else "",
            "scope": scope,
            "is_search": "q" in mode_params,
            "is_similar": "similar" in mode_params,
            "similar_to": anchor,
            "unindexed": unindexed,
            **_gallery_links(min_score, sort, active_tag, scope, mode_params),
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
    return render(request, "ratings/tags.html", {"tags": tags, "mode": "tags", **_counts()})


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
    return with_toast(HttpResponse(""), "Tag deleted")


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
    # Nothing to swap: the outcome travels as a toast (204 keeps htmx from
    # touching the DOM while still processing the HX-Trigger header).
    parts = []
    if sent:
        parts.append("Shared to " + ", ".join(sent))
    parts.extend(errors)
    if not parts:
        parts.append("No channels selected")
    kind = "ok" if sent and not errors else "error"
    return with_toast(HttpResponse(status=204), " · ".join(parts), kind)


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
    return with_toast(HttpResponse(""), "Channel deleted")


def _index_status_ctx() -> dict:
    """
    Everything the index block on the config page needs, read from the
    database (counts) and the queue table (queued flag), never from the
    session (contract V15). The report about the last slice only matters when
    nothing is queued and rows are still waiting: while the chain runs the
    pending fragment is the status, and when it is done there is nothing to
    report.
    """
    indexed, total = search.index_counts()
    remaining = total - indexed
    queued = search.index_job_queued()
    report = None
    if not queued and remaining > 0:
        report = search.last_index_report()
    return {
        "indexed": indexed,
        "total": total,
        "remaining": remaining,
        "index_queued": queued,
        "index_report": report,
    }


def _render_index_status(request):
    ctx = _index_status_ctx()
    if ctx["index_queued"]:
        return render(request, "ratings/_index_pending.html", ctx)
    return render(request, "ratings/_index_result.html", ctx)


@login_required
@require_POST
def trigger_search_index(request):
    """
    Start the search index chain and return its status fragment.

    Unlike trigger_scrape / trigger_train there is no session bookkeeping: the
    chain is many short tasks, and enqueue_index_job_if_needed already refuses
    a second chain (V13), so a double tap just re-renders the pending state.
    """
    search.enqueue_index_job_if_needed()
    return _render_index_status(request)


@login_required
def search_index_status(request):
    """Polling target of _index_pending.html; the fragment it returns decides whether polling goes on."""
    return _render_index_status(request)


@login_required
def job_indicator(request):
    """
    Polling target for the nav's job indicator (UI contract V8).

    The context comes from the active_jobs context processor, so this view
    only picks the partial; it is polled every few seconds while a job runs
    and the partial stops polling itself once it renders the idle state.
    """
    return render(request, "ratings/_nav_job.html")


def _job_is_running(request) -> bool:
    """
    True while a scrape, train or search-index job is queued or running anywhere.

    The session knows about jobs this browser started; OrmQ (django-q's ORM
    broker table) still holds every task that is queued or running, including
    scheduled scrapes and jobs started from another session, until the worker
    acknowledges it after completion.
    """
    from django_q.models import OrmQ

    if _training_ctx(request)["active_task_id"]:
        return True
    if _scrape_ctx(request)["active_scrape_task_id"]:
        return True
    return OrmQ.objects.exists()


@login_required
@require_POST
def fresh_start_view(request):
    """
    Start over: wipe the library, keep the configuration (UI contract V12).

    Three guards, all enforced here and not only in the UI: login and POST via
    the decorators (OWASP A01), no running scrape or train job (its worker
    would write rows and files into the wiped library), and the exact
    confirmation word in the body (OWASP A04: a confirmation that lives only in
    the sheet is one crafted request away from nothing).
    """
    if _job_is_running(request):
        messages.error(
            request,
            "A scrape, train or index job is running. Wait for it to finish before starting over.",
        )
        return redirect("config")
    if request.POST.get("confirm", "").strip() != reset.CONFIRM_WORD:
        messages.error(
            request, f"Type {reset.CONFIRM_WORD} to confirm the fresh start. Nothing was deleted."
        )
        return redirect("config")
    result = reset.fresh_start(
        Path(settings.DATA_DIR),
        [Path(settings.WEIGHTS_PATH), Path(settings.NSFW_WEIGHTS_PATH)],
    )
    messages.success(
        request,
        f"Fresh start done: {result['images']} images and {result['weights']} classifier "
        "file(s) removed. Sources, channels and settings kept.",
    )
    return redirect("config")


def _score_from_post(request) -> int | None:
    """
    The score a form submitted, or None when it is missing or outside 0–6.

    0 is "trash": below the 1–6 scale but a real rating. Anything else (a
    missing field, text, 7) is ignored instead of answered with an error: the
    buttons only ever send 0–6, so a bad value means a stale page, and
    re-rendering the current state is the right answer to that.
    """
    try:
        value = int(request.POST.get("score", -1))
    except (ValueError, TypeError):
        return None
    if 0 <= value <= 6:
        return value
    return None


# ── Gallery lightbox (UI contract V9) ────────────────────────────────────────


def _lightbox_ctx(image: Image) -> dict:
    """
    Context for the lightbox panel. It names the same shared partials as the
    review card (score row, action row, tag editor) with its own endpoints,
    and `lightbox=True` makes prev/next client-side so the order is the
    grid's, which only the browser knows (random sort, tag filter).
    """
    return {
        "image": image,
        "prediction": _taste_prediction(image),
        "scores": range(1, 7),
        "lightbox": True,
        "score_url": "lightbox_score",
        "purge_url": "lightbox_purge",
        "nsfw_url": "lightbox_nsfw",
        "hx_target": "#lightbox-content",
        "hx_swap": "innerHTML",
    }


def _lightbox_update(request, image: Image | None, deleted_hash: str | None = None):
    """
    Answer a lightbox action with the panel plus, out of band, the grid card
    and the nav badges, so one response keeps every view of the image in step.
    A purged image has no panel; its card is deleted from the grid instead.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    ctx = {**_counts(show_nsfw), "show_nsfw": show_nsfw, "deleted_hash": deleted_hash}
    if image is not None:
        ctx.update(_lightbox_ctx(image))
    return render(request, "ratings/_lightbox_update.html", ctx)


@login_required
def lightbox(request, content_hash: str):
    image = get_object_or_404(
        Image.objects.prefetch_related("tags"), content_hash=content_hash, is_purged=False
    )
    return render(request, "ratings/_lightbox.html", _lightbox_ctx(image))


@login_required
@require_POST
def lightbox_score(request, content_hash: str):
    """Re-score from the gallery or Below cutoff; the image stays in view (no rate-and-advance there)."""
    image = get_object_or_404(Image, content_hash=content_hash, is_purged=False)
    score_val = _score_from_post(request)
    if score_val is not None:
        image.score = score_val
        image.rated_at = timezone.now()
        image.save(update_fields=["score", "rated_at"])
    return _lightbox_update(request, image)


@login_required
@require_POST
def lightbox_nsfw(request, content_hash: str):
    image = get_object_or_404(Image, content_hash=content_hash, is_purged=False)
    image.is_nsfw = not image.is_nsfw
    image.save(update_fields=["is_nsfw"])
    return _lightbox_update(request, image)


@login_required
@require_POST
def lightbox_purge(request, content_hash: str):
    """Hard-delete from the lightbox; the grid card leaves with the response and the toast reports it."""
    image = get_object_or_404(Image, content_hash=content_hash, is_purged=False)
    _purge_image(image)
    response = _lightbox_update(request, None, deleted_hash=content_hash)
    return with_toast(response, "Image purged")
