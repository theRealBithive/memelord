from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from ratings.models import Image, LogEntry, NotificationConfig, ScrapeSchedule, Source, Tag
from ratings.utils import move_image as _move_image_util, purge_image as _purge_image
import ratings.notifiers as notifiers

_INTERVAL_CHOICES = [1, 2, 4, 6, 12, 24, 48, 72, 168]

WEIGHTS_PATH = Path(settings.WEIGHTS_PATH)
DATA_DIR = Path(settings.DATA_DIR)


def index(request):
    return redirect("review_corpus")


def _counts(show_nsfw: bool = False) -> dict:
    """
    Aggregate image counts across all queues and locations in a single DB query.

    Used by every page for nav badges; a separate query per badge would be 6×
    the DB round-trips per request. show_nsfw controls whether NSFW images are
    folded into the main counts or kept separate so the user can see SFW and
    NSFW numbers independently.
    """
    qs = Image.objects.filter(file_deleted=False, is_purged=False)
    queue_filter = Q(
        location__in=[Image.INBOX, Image.CORPUS], score__isnull=True
    )
    if show_nsfw:
        return qs.aggregate(
            queue_count=Count("pk", filter=queue_filter),
            void_count=Count("pk", filter=Q(location=Image.VOID)),
            fav_count=Count("pk", filter=Q(location=Image.CORPUS, is_favourite=True)),
            nsfw_queue_count=Count("pk", filter=queue_filter & Q(is_nsfw=True)),
            nsfw_void_count=Count("pk", filter=Q(location=Image.VOID, is_nsfw=True)),
            nsfw_fav_count=Count(
                "pk", filter=Q(location=Image.CORPUS, is_favourite=True, is_nsfw=True)
            ),
        )
    return qs.aggregate(
        queue_count=Count("pk", filter=queue_filter & Q(is_nsfw=False)),
        void_count=Count("pk", filter=Q(location=Image.VOID, is_nsfw=False)),
        fav_count=Count(
            "pk", filter=Q(location=Image.CORPUS, is_favourite=True, is_nsfw=False)
        ),
        nsfw_queue_count=Count("pk", filter=queue_filter & Q(is_nsfw=True)),
        nsfw_void_count=Count("pk", filter=Q(location=Image.VOID, is_nsfw=True)),
        nsfw_fav_count=Count(
            "pk", filter=Q(location=Image.CORPUS, is_favourite=True, is_nsfw=True)
        ),
    )


def _get_next(
    mode: str, exclude_hash: str | None = None, show_nsfw: bool = False
) -> Image | None:
    """
    Pick the next image for the swipe-style rating view (random order).

    Random ordering intentionally avoids anchoring bias — sequential ordering
    would cause the user to mentally anticipate the next image rather than
    judging each one independently.
    """
    qs = Image.objects.filter(file_deleted=False)

    if mode == "nsfw_fav":
        qs = qs.filter(location=Image.CORPUS, is_favourite=True, is_nsfw=True)
    elif mode.startswith("nsfw_"):
        qs = qs.filter(location=mode[5:], is_nsfw=True)
    elif mode == "fav":
        qs = qs.filter(location=Image.CORPUS, is_favourite=True)
        if not show_nsfw:
            qs = qs.filter(is_nsfw=False)
    else:
        qs = qs.filter(location=mode)
        if not show_nsfw:
            qs = qs.filter(is_nsfw=False)

    qs = qs.order_by("?")

    if exclude_hash:
        qs = qs.exclude(content_hash=exclude_hash)
    return qs.first()


def _fmt_elapsed(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


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
    started_at_str = request.session.get("training_started_at")
    elapsed = None
    if started_at_str:
        started_at = datetime.fromisoformat(started_at_str)
        elapsed = int((datetime.now(dt_timezone.utc) - started_at).total_seconds())
        if elapsed > 14400:
            request.session.pop("training_task_id", None)
            request.session.pop("training_started_at", None)
            return {"active_task_id": None, "training_elapsed": None}
    return {"active_task_id": task_id, "training_elapsed": _fmt_elapsed(elapsed)}


def _build_ctx(
    mode: str, image: Image | None, show_nsfw: bool = False, request=None
) -> dict:
    """Build the minimal context dict shared by all swipe-mode rating templates."""
    counts = _counts(show_nsfw)
    ctx = {
        "mode": mode,
        "image": image,
        "queue_count": counts.get(f"{mode}_count", 0),
        "show_nsfw": show_nsfw,
        **counts,
    }
    if request is not None:
        ctx.update(_training_ctx(request))
    return ctx


def _mode_view(request, mode: str):
    show_nsfw = request.session.get("show_nsfw", False)
    image = _get_next(mode, show_nsfw=show_nsfw)
    return render(request, "ratings/rate.html", _build_ctx(mode, image, show_nsfw, request))


def _move_image(image: Image, new_location: str) -> None:
    _move_image_util(image, new_location, DATA_DIR)


def _apply_rating(image: Image, target_location: str, is_fav: bool, now) -> None:
    """
    Move image to target location and record the rating atomically.

    update_fields is used instead of a full save() so concurrent writes from
    other sessions don't clobber unrelated fields (e.g. embedding, phash)
    that may be updated by a background scrape at the same time.
    """
    fields = ["is_favourite", "rated_at"]
    if image.location != target_location:
        _move_image(image, target_location)
        fields += ["file_path", "location"]
    image.is_favourite = is_fav
    image.rated_at = now
    image.save(update_fields=fields)


def _neighbor_hash(all_hashes: list[str], content_hash: str) -> str | None:
    """Next hash in list, or previous if last, or None if single item."""
    if not all_hashes:
        return None
    idx = {h: i for i, h in enumerate(all_hashes)}.get(content_hash, 0)
    if idx < len(all_hashes) - 1:
        return all_hashes[idx + 1]
    return all_hashes[idx - 1] if idx > 0 else None


@login_required
def rate_inbox(request):
    return redirect("review_corpus")


@login_required
def rate_corpus(request):
    return redirect("review_corpus")


@login_required
def rate_void(request):
    return redirect("review_void")


@login_required
def rate_fav(request):
    return _mode_view(request, "fav")


@login_required
def rate_nsfw_inbox(request):
    return redirect("rate_nsfw_corpus")


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
def rate_nsfw_void(request):
    """
    NSFW trash grid — same layout as the regular void grid but filtered to
    is_nsfw=True. Reuses the same action endpoints since they operate on any
    void image regardless of NSFW status.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    images = list(
        Image.objects.filter(location=Image.VOID, is_nsfw=True, file_deleted=False)
        .order_by("void_seen_at", "-rated_at")[:500]
    )
    return render(
        request,
        "ratings/void_grid.html",
        {
            **_counts(show_nsfw),
            **_training_ctx(request),
            "images": images,
            "total": len(images),
            "show_nsfw": show_nsfw,
            "mode": "nsfw_void",
        },
    )


@login_required
def rate_nsfw_fav(request):
    return _mode_view(request, "nsfw_fav")


@login_required
@require_POST
def submit_rating(request, content_hash: str, action: str):
    """
    Handle a swipe/keypress rating action from the swipe-style review flow.

    Returns the HTMX partial with the next image if the request came via htmx,
    otherwise a full page render for non-JS fallback. mode is passed from the
    template so this single endpoint serves inbox, corpus, void, fav, and nsfw
    variants without separate URL patterns for each.
    """
    mode = request.POST.get("mode", "inbox")
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    now = timezone.now()

    if action in ("good", "fav"):
        _apply_rating(image, Image.CORPUS, action == "fav", now)
    elif action == "bad":
        _apply_rating(image, Image.VOID, False, now)
    elif action == "purge":
        _purge_image(image)
    elif action == "unfav":
        image.is_favourite = False
        image.save(update_fields=["is_favourite"])
    elif action == "mark_nsfw":
        image.is_nsfw = True
        image.save(update_fields=["is_nsfw"])
    elif action == "mark_safe":
        image.is_nsfw = False
        image.save(update_fields=["is_nsfw"])

    next_image = _get_next(mode, exclude_hash=content_hash, show_nsfw=show_nsfw)
    ctx = _build_ctx(mode, next_image, show_nsfw, request)
    if request.htmx:
        return render(request, "ratings/_htmx_rating.html", ctx)
    return render(request, "ratings/rate.html", ctx)


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
        Task.objects.filter(func="ratings.tasks.run_train")
        .order_by("-stopped")
        .first()
    )
    last_train_info: dict | None = None
    if last_train_task is not None:
        result = last_train_task.result if isinstance(last_train_task.result, dict) else {}
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

    gallery_qs = Image.objects.filter(location=Image.CORPUS, file_deleted=False, score__isnull=False)
    if not show_nsfw:
        gallery_qs = gallery_qs.filter(is_nsfw=False)

    gallery_total = gallery_qs.count()

    score_dist = list(
        gallery_qs.values("score").annotate(n=Count("content_hash")).order_by("-score")
    )
    score_dist_max = max((row["n"] for row in score_dist), default=1)

    source_breakdown = list(
        gallery_qs.values("source_label").annotate(n=Count("content_hash")).order_by("-n")
    )

    seven_days_ago = timezone.now() - timedelta(days=7)
    scraped_7d = Image.objects.filter(downloaded_at__gte=seven_days_ago, file_deleted=False).count()
    rated_7d = Image.objects.filter(
        rated_at__gte=seven_days_ago,
        location__in=[Image.CORPUS, Image.VOID],
        file_deleted=False,
    ).count()

    return render(
        request,
        "ratings/stats.html",
        {
            **counts,
            **_training_ctx(request),
            "show_nsfw": show_nsfw,
            "last_trained": last_trained,
            "last_train": last_train_info,
            "gallery_total": gallery_total,
            "score_dist": score_dist,
            "score_dist_max": score_dist_max,
            "source_breakdown": source_breakdown,
            "scraped_7d": scraped_7d,
            "rated_7d": rated_7d,
        },
    )


@login_required
@require_POST
def trigger_scrape(request):
    """
    Run a scrape synchronously in the request/response cycle.

    Scraping is synchronous (not async_task) because it's fast enough for a
    normal request timeout and the user expects to see the new image count
    immediately. Training is async because DINOv2 encoding takes minutes.
    """
    from loguru import logger

    from ratings import scraper
    from ratings.tasks import _db_sink, _trim_logs

    _trim_logs()
    sink_id = logger.add(_db_sink("scrape"), format="{message}")
    try:
        counts = scraper.run(
            config_path=Path(settings.CONFIG_PATH),
            data_dir=DATA_DIR,
            vision=scraper.vision_config_from_settings(),
        )
        ctx = {"ok": True, "total": sum(counts.values()), "counts": counts}
    except Exception as exc:
        logger.error("Scrape failed: {}", exc)
        ctx = {"ok": False, "error": str(exc)}
    finally:
        logger.remove(sink_id)
    return render(request, "ratings/_scrape_result.html", ctx)


@login_required
@require_POST
def trigger_train(request):
    """
    Enqueue a training job via django-q and return a polling fragment.

    Training blocks for several minutes (DINOv2 encoding + LogReg fit), so it
    runs in a background worker. The session stores the task ID so the polling
    template knows which job to watch via train_status.
    """
    from django_q.tasks import async_task

    if request.session.get("training_task_id"):
        ctx = _training_ctx(request)
        return render(
            request,
            "ratings/_train_pending.html",
            {
                "task_id": ctx["active_task_id"],
                "elapsed": ctx["training_elapsed"],
            },
        )
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

    task = fetch(task_id)
    if task is None or task.stopped is None:
        started_at_str = request.session.get("training_started_at")
        elapsed = None
        if started_at_str:
            started_at = datetime.fromisoformat(started_at_str)
            elapsed = int((datetime.now(dt_timezone.utc) - started_at).total_seconds())
        return render(
            request,
            "ratings/_train_pending.html",
            {
                "task_id": task_id,
                "elapsed": _fmt_elapsed(elapsed),
            },
        )

    request.session.pop("training_task_id", None)
    request.session.pop("training_started_at", None)

    result = task.result or {}
    return render(
        request,
        "ratings/_train_result.html",
        {
            "ok": result.get("ok", False),
            "error": result.get("error", "Unknown error."),
            "corpus_n": Image.objects.filter(location=Image.CORPUS, file_deleted=False).count(),
            "void_n": Image.objects.filter(location=Image.VOID, file_deleted=False).count(),
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
    """Render the configuration page combining sources, schedule, and training status."""
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
    if stype == Source.PIXELFED and not name.startswith("http"):
        return render(
            request,
            "ratings/_source_error.html",
            {"error": "Pixelfed value must be a URL (https://…)."},
        )
    if stype == Source.MASTODON and "@" not in name.lstrip("@"):
        return render(
            request,
            "ratings/_source_error.html",
            {"error": "Mastodon handle must include an instance, e.g. @user@mastodon.social"},
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
    Build browse context with stable prev/next navigation from a queryset.

    The full hash list is materialised once so prev/next positions are computed
    from the same snapshot. Fetching prev/next lazily with separate queries
    risks a race condition where an image is rated (and removed from the queue)
    between calls, shifting the navigation offsets.
    """
    all_hashes = list(qs.values_list("content_hash", flat=True))
    base = {"show_nsfw": show_nsfw, **_counts(show_nsfw)}
    if extra:
        base.update(extra)

    if not all_hashes:
        ctx: dict = {"image": None, "mode": mode, **base}
        if request is not None:
            ctx.update(_training_ctx(request))
        return ctx

    hash_index = {h: i for i, h in enumerate(all_hashes)}
    idx = hash_index.get(content_hash, 0) if content_hash else 0

    ctx = {
        "image": Image.objects.get(content_hash=all_hashes[idx]),
        "prev_hash": all_hashes[idx - 1] if idx > 0 else None,
        "next_hash": all_hashes[idx + 1] if idx < len(all_hashes) - 1 else None,
        "position": idx + 1,
        "total": len(all_hashes),
        "mode": mode,
        **base,
    }
    if request is not None:
        ctx.update(_training_ctx(request))
    return ctx


# ── Corpus review ─────────────────────────────────────────────────────────────


def _review_qs(show_nsfw: bool = False):
    """
    Build the ordered queue for the primary corpus review flow.

    Filters both inbox and corpus with score__isnull so unscored items from
    either location feed the same queue — an image moved to corpus without a
    score (e.g. by the taste classifier) still needs a manual score before it
    leaves the queue. Unseen images (queue_seen_at IS NULL) sort first in
    SQLite ASC; then oldest-downloaded.
    """
    qs = Image.objects.filter(
        location__in=[Image.INBOX, Image.CORPUS],
        score__isnull=True,
        is_purged=False,
        file_deleted=False,
    )
    if not show_nsfw:
        qs = qs.filter(is_nsfw=False)
    return qs.order_by("queue_seen_at", "downloaded_at")


def _review_nsfw_qs(show_nsfw: bool = False):
    """
    Same queue shape as _review_qs but filtered to NSFW images only.

    show_nsfw is accepted but ignored — this queue is always NSFW-only by
    definition. The parameter exists so qs_fn callers can treat both queues
    with the same (show_nsfw: bool) → QuerySet signature.
    """
    return Image.objects.filter(
        location__in=[Image.INBOX, Image.CORPUS],
        is_nsfw=True,
        score__isnull=True,
        is_purged=False,
        file_deleted=False,
    ).order_by("queue_seen_at", "downloaded_at")


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
            "fav_url": "toggle_fav_corpus",
            "trash_url": "trash_corpus",
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
            "fav_url": "toggle_fav_nsfw_corpus",
            "trash_url": "trash_nsfw_corpus",
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

    next_hash is captured before scoring because scoring changes queue ordering
    — the image disappears from its current position in the list once scored.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )
    all_hashes = list(qs_fn(show_nsfw).values_list("content_hash", flat=True))
    idx = {h: i for i, h in enumerate(all_hashes)}.get(content_hash, 0)
    next_hash = all_hashes[idx + 1] if idx < len(all_hashes) - 1 else None
    try:
        score_val = int(request.POST.get("score", 0))
        if 1 <= score_val <= 6:
            fields = ["score", "rated_at"]
            if image.location == Image.INBOX:
                _move_image(image, Image.CORPUS)
                fields += ["file_path", "location"]
            image.score = score_val
            image.rated_at = timezone.now()
            image.save(update_fields=fields)
    except (ValueError, TypeError):
        pass
    ctx = ctx_fn(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


def _trash_impl(request, content_hash: str, qs_fn, ctx_fn):
    """
    Shared trash logic for both the normal and NSFW review queues.

    Neighbour is captured before the move so the queue ordering is stable
    when we compute prev/next for the context.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )
    next_hash = _neighbor_hash(
        list(qs_fn(show_nsfw).values_list("content_hash", flat=True)),
        content_hash,
    )
    _move_image(image, Image.VOID)
    image.is_favourite = False
    image.rated_at = timezone.now()
    image.save(update_fields=["file_path", "location", "is_favourite", "rated_at"])
    ctx = ctx_fn(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


def _toggle_fav_impl(request, content_hash: str, qs_fn, ctx_fn):
    """
    Shared fav-toggle for both the normal and NSFW review queues.

    We stay on the same image after a fav toggle so the user can see the star
    update without losing their place in the queue.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )
    image.is_favourite = not image.is_favourite
    image.save(update_fields=["is_favourite"])
    ctx = ctx_fn(content_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


def _purge_impl(request, content_hash: str, qs_fn, ctx_fn):
    """
    Shared purge logic for both the normal and NSFW review queues.

    Like _trash_impl, the neighbour is captured first so we know where to
    navigate after the image is hard-deleted from disk.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )
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
def trash_corpus(request, content_hash: str):
    return _trash_impl(request, content_hash, _review_qs, _review_ctx)


@login_required
@require_POST
def toggle_fav_corpus(request, content_hash: str):
    return _toggle_fav_impl(request, content_hash, _review_qs, _review_ctx)


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
def trash_nsfw_corpus(request, content_hash: str):
    return _trash_impl(request, content_hash, _review_nsfw_qs, _review_nsfw_ctx)


@login_required
@require_POST
def toggle_fav_nsfw_corpus(request, content_hash: str):
    return _toggle_fav_impl(request, content_hash, _review_nsfw_qs, _review_nsfw_ctx)


@login_required
@require_POST
def purge_nsfw_corpus(request, content_hash: str):
    return _purge_impl(request, content_hash, _review_nsfw_qs, _review_nsfw_ctx)


@login_required
@require_POST
def toggle_nsfw(request, content_hash: str):
    """
    Toggle is_nsfw on any image, then navigate appropriately for the current mode.

    Navigation logic differs per queue:
    - Normal corpus: marking NSFW only removes the image when show_nsfw is False
      (otherwise it stays visible in the queue).
    - NSFW corpus: marking safe always navigates away — the image is by definition
      no longer in the NSFW queue regardless of show_nsfw.
    - Void: same show_nsfw rule as normal corpus.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    location = image.location

    if location in (Image.INBOX, Image.CORPUS):
        mode = request.POST.get("mode", "corpus")
        if mode == "nsfw_corpus":
            neighbor = _neighbor_hash(
                list(_review_nsfw_qs().values_list("content_hash", flat=True)),
                content_hash,
            )
            image.is_nsfw = not image.is_nsfw
            image.save(update_fields=["is_nsfw"])
            ctx = _review_nsfw_ctx(
                content_hash if image.is_nsfw else neighbor, show_nsfw, request
            )
        else:
            # Capture neighbour BEFORE saving so ordering is stable.
            neighbor = _neighbor_hash(
                list(_review_qs(show_nsfw).values_list("content_hash", flat=True)),
                content_hash,
            )
            image.is_nsfw = not image.is_nsfw
            image.save(update_fields=["is_nsfw"])
            if not show_nsfw and image.is_nsfw:
                ctx = _review_ctx(neighbor, show_nsfw, request)
            else:
                ctx = _review_ctx(content_hash, show_nsfw, request)
        _mark_queue_seen(ctx.get("image"))
        return render(request, "ratings/_review_htmx.html", ctx)

    if location == Image.VOID:
        neighbor = _neighbor_hash(
            list(_void_review_qs(show_nsfw).values_list("content_hash", flat=True)),
            content_hash,
        )
        image.is_nsfw = not image.is_nsfw
        image.save(update_fields=["is_nsfw"])
        if not show_nsfw and image.is_nsfw:
            ctx = _void_review_ctx(neighbor, show_nsfw, request)
        else:
            ctx = _void_review_ctx(content_hash, show_nsfw, request)
        return render(request, "ratings/_void_htmx.html", ctx)

    # Inbox (and any other location)
    mode = request.POST.get("mode", "inbox")
    image.is_nsfw = not image.is_nsfw
    image.save(update_fields=["is_nsfw"])
    if not show_nsfw and image.is_nsfw:
        next_image = _get_next(mode, exclude_hash=content_hash, show_nsfw=show_nsfw)
        if mode in ("inbox", "nsfw_inbox"):
            _mark_queue_seen(next_image)
        ctx = _build_ctx(mode, next_image, show_nsfw, request)
    else:
        ctx = _build_ctx(mode, image, show_nsfw, request)
    return render(request, "ratings/_htmx_rating.html", ctx)


# ── Void review ───────────────────────────────────────────────────────────────


def _void_review_qs(show_nsfw: bool = False):
    """
    Queue of void images ordered unseen-first, then newest-trashed.

    Newest-trashed secondary order means recently discarded images appear first
    after the unseen batch, making it easy to undo an accidental trash.
    """
    qs = Image.objects.filter(location=Image.VOID, file_deleted=False)
    if not show_nsfw:
        qs = qs.filter(is_nsfw=False)
    return qs.order_by("void_seen_at", "-rated_at")


def _mark_void_seen(image: Image | None) -> None:
    """Stamp void_seen_at once; unseen images sort first in the void queue."""
    if image is not None and image.void_seen_at is None:
        image.void_seen_at = timezone.now()
        image.save(update_fields=["void_seen_at"])


def _void_review_ctx(
    content_hash: str | None, show_nsfw: bool = False, request=None
) -> dict:
    """Build browse context for the void review queue."""
    return _browse_ctx(
        _void_review_qs(show_nsfw), content_hash, "void", show_nsfw, request
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

    fav_only = request.GET.get("fav") == "1"
    active_tag = request.GET.get("tag", "").strip().lower()

    qs = Image.objects.filter(
        location=Image.CORPUS,
        file_deleted=False,
        score__isnull=False,
        score__gte=min_score,
    )
    if not show_nsfw:
        qs = qs.filter(is_nsfw=False)
    if fav_only:
        qs = qs.filter(is_favourite=True)
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
            "fav_only": fav_only,
            "scores": range(1, 7),
            "show_nsfw": show_nsfw,
            "mode": "gallery",
            "all_tags": all_tags,
            "active_tag": active_tag,
        },
    )


@login_required
def review_void(request, content_hash: str | None = None):
    """
    Void grid view — shows all trash images at once for bulk selection and rescue.

    content_hash is ignored (kept for URL compat); the grid always shows the full
    void queue. Images are ordered unseen-first so recently trashed items appear at
    the top for quick undo.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    images = list(_void_review_qs(show_nsfw)[:500])
    return render(
        request,
        "ratings/void_grid.html",
        {
            **_counts(show_nsfw),
            **_training_ctx(request),
            "images": images,
            "total": len(images),
            "show_nsfw": show_nsfw,
            "mode": "void",
        },
    )


@login_required
@require_POST
def void_grid_action(request, content_hash: str):
    """
    JSON endpoint for single-image actions from the void grid lightbox.

    Returns {"rescued": True} or {"purged": True} so the JS can remove the
    item from the DOM without a page reload.
    """
    action = request.POST.get("action", "")
    image = get_object_or_404(Image, content_hash=content_hash, location=Image.VOID, file_deleted=False)
    if action == "purge":
        _purge_image(image)
        return JsonResponse({"purged": True})
    if action in ("rescue", "rescue_fav"):
        _move_image(image, Image.CORPUS)
        image.is_favourite = action == "rescue_fav"
        image.rated_at = timezone.now()
        image.save(update_fields=["file_path", "location", "is_favourite", "rated_at"])
        return JsonResponse({"rescued": True})
    return JsonResponse({"error": "unknown action"}, status=400)


@login_required
@require_POST
def void_bulk_rescue(request):
    """
    Rescue multiple void images to corpus in one request.

    Accepts a list of hashes via repeated 'hashes' POST values. Capped at 500
    to prevent accidental mass operations. The fav=1 param marks all rescued
    images as favourites.
    """
    hashes = request.POST.getlist("hashes")[:500]
    fav = request.POST.get("fav") == "1"
    rescued = 0
    for h in hashes:
        try:
            img = Image.objects.get(content_hash=h, location=Image.VOID, file_deleted=False)
        except Image.DoesNotExist:
            continue
        _move_image(img, Image.CORPUS)
        img.is_favourite = fav
        img.rated_at = timezone.now()
        img.save(update_fields=["file_path", "location", "is_favourite", "rated_at"])
        rescued += 1
    return JsonResponse({"rescued": rescued})


@login_required
@require_POST
def void_review_action(request, content_hash: str):
    """
    Handle rescue/purge/nsfw actions in the void review queue.

    Neighbour is captured before any state changes so the navigation target
    is stable regardless of which action removes the image from the queue.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    action = request.POST.get("action", "")
    image = get_object_or_404(Image, content_hash=content_hash, location=Image.VOID)

    next_hash = _neighbor_hash(
        list(_void_review_qs(show_nsfw).values_list("content_hash", flat=True)),
        content_hash,
    )

    if action == "purge":
        _purge_image(image)
        ctx = _void_review_ctx(next_hash, show_nsfw, request)
    elif action in ("rescue", "rescue_fav"):
        _move_image(image, Image.CORPUS)
        image.is_favourite = action == "rescue_fav"
        image.rated_at = timezone.now()
        image.save(update_fields=["file_path", "location", "is_favourite", "rated_at"])
        ctx = _void_review_ctx(next_hash, show_nsfw, request)
    elif action == "mark_nsfw":
        image.is_nsfw = not image.is_nsfw
        image.save(update_fields=["is_nsfw"])
        if not show_nsfw and image.is_nsfw:
            ctx = _void_review_ctx(next_hash, show_nsfw, request)
        else:
            ctx = _void_review_ctx(content_hash, show_nsfw, request)
    else:
        ctx = _void_review_ctx(content_hash, show_nsfw, request)

    _mark_void_seen(ctx.get("image"))
    return render(request, "ratings/_void_htmx.html", ctx)


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
@require_POST
def gallery_action(request, content_hash: str):
    """
    Handle inline score/fav/trash actions from the gallery grid.

    Returns JSON so the gallery JS can update the card in-place without a full
    page reload. Trash returns {"deleted": True} as a signal to remove the card
    from the DOM.
    """
    action = request.POST.get("action", "")
    image = get_object_or_404(Image, content_hash=content_hash)
    now = timezone.now()

    if action == "trash":
        if image.location == Image.CORPUS:
            _move_image(image, Image.VOID)
            image.is_favourite = False
            image.rated_at = now
            image.save(update_fields=["file_path", "location", "is_favourite", "rated_at"])
        return JsonResponse({"deleted": True})

    if action == "fav":
        image.is_favourite = not image.is_favourite
        image.save(update_fields=["is_favourite"])
    elif action == "score":
        try:
            score_val = int(request.POST.get("score", 0))
            if 1 <= score_val <= 6:
                image.score = score_val
                image.rated_at = now
                image.save(update_fields=["score", "rated_at"])
        except (ValueError, TypeError):
            pass
    elif action == "nsfw":
        image.is_nsfw = not image.is_nsfw
        image.save(update_fields=["is_nsfw"])

    return JsonResponse({"score": image.score, "fav": image.is_favourite, "nsfw": image.is_nsfw})


@login_required
@require_POST
def share_image(request, content_hash):
    """
    Share an image to Mattermost and/or Signal.

    Sends synchronously — both APIs are expected to be on the same LAN so
    latency is negligible. The image URL is built from the request so it works
    regardless of the deployment domain.
    """
    image = get_object_or_404(Image, content_hash=content_hash)
    cfg, _ = NotificationConfig.objects.get_or_create(pk=1)
    media_url = request.build_absolute_uri(settings.MEDIA_URL + image.file_path)
    errors = []
    if request.POST.get("mattermost") and cfg.mattermost_enabled and cfg.mattermost_token:
        try:
            notifiers.send_to_mattermost(cfg, media_url, image.source_label or "")
        except Exception as exc:
            errors.append(f"Mattermost: {exc}")
    if request.POST.get("signal") and cfg.signal_enabled and cfg.signal_api_url:
        try:
            notifiers.send_to_signal(cfg, media_url, image.source_label or "")
        except Exception as exc:
            errors.append(f"Signal: {exc}")
    return render(request, "ratings/_share_toast.html", {"errors": errors})


@login_required
@require_POST
def save_notification_config(request):
    """Persist Mattermost and Signal notification settings from the Config UI."""
    cfg, _ = NotificationConfig.objects.get_or_create(pk=1)
    cfg.mattermost_enabled = request.POST.get("mattermost_enabled") == "1"
    cfg.mattermost_base_url = request.POST.get("mattermost_base_url", "").strip()
    cfg.mattermost_token = request.POST.get("mattermost_token", "").strip()
    cfg.mattermost_channel_id = request.POST.get("mattermost_channel_id", "").strip()
    cfg.mattermost_message_prefix = request.POST.get("mattermost_message_prefix", "").strip()
    cfg.signal_enabled = request.POST.get("signal_enabled") == "1"
    cfg.signal_api_url = request.POST.get("signal_api_url", "").strip()
    cfg.signal_sender = request.POST.get("signal_sender", "").strip()
    cfg.signal_recipients = request.POST.get("signal_recipients", "").strip()
    cfg.signal_message_prefix = request.POST.get("signal_message_prefix", "").strip()
    cfg.save()
    return render(request, "ratings/_notification_config.html", {"notification_cfg": cfg})
