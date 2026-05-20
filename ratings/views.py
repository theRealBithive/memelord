from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from ratings.models import Image, LogEntry, ScrapeSchedule, Source
from ratings.utils import move_image as _move_image_util, purge_image as _purge_image

_INTERVAL_CHOICES = [1, 2, 4, 6, 12, 24, 48, 72, 168]

WEIGHTS_PATH = Path(settings.WEIGHTS_PATH)
DATA_DIR = Path(settings.DATA_DIR)


def index(request):
    return redirect("review_corpus")


def _counts(show_nsfw: bool = False) -> dict:
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
    task_id = request.session.get("training_task_id")
    if not task_id:
        return {"active_task_id": None, "training_elapsed": None}
    started_at_str = request.session.get("training_started_at")
    elapsed = None
    if started_at_str:
        started_at = datetime.fromisoformat(started_at_str)
        elapsed = int((datetime.now(dt_timezone.utc) - started_at).total_seconds())
        if elapsed > 1800:
            request.session.pop("training_task_id", None)
            request.session.pop("training_started_at", None)
            return {"active_task_id": None, "training_elapsed": None}
    return {"active_task_id": task_id, "training_elapsed": _fmt_elapsed(elapsed)}


def _build_ctx(
    mode: str, image: Image | None, show_nsfw: bool = False, request=None
) -> dict:
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
    show_nsfw = request.session.get("show_nsfw", False)
    ctx = _review_nsfw_ctx(content_hash, show_nsfw, request)
    if request.htmx:
        return render(request, "ratings/_review_htmx.html", ctx)
    return render(request, "ratings/review.html", ctx)


@login_required
def rate_nsfw_void(request):
    return _mode_view(request, "nsfw_void")


@login_required
def rate_nsfw_fav(request):
    return _mode_view(request, "nsfw_fav")


@login_required
@require_POST
def submit_rating(request, content_hash: str, action: str):
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
    request.session["show_nsfw"] = not request.session.get("show_nsfw", False)
    return redirect(request.META.get("HTTP_REFERER") or "index")


@login_required
def stats(request):
    show_nsfw = request.session.get("show_nsfw", False)
    counts = _counts(show_nsfw)

    last_trained = None
    try:
        mtime = WEIGHTS_PATH.stat().st_mtime
        last_trained = datetime.fromtimestamp(mtime, tz=dt_timezone.utc)
    except FileNotFoundError:
        pass

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

    source, created = Source.objects.get_or_create(type=stype, name=name)
    if not created:
        source.enabled = True
        source.save(update_fields=["enabled"])
    return render(request, "ratings/_source_row.html", {"source": source})


@login_required
@require_POST
def source_toggle(request, pk):
    source = get_object_or_404(Source, pk=pk)
    source.enabled = not source.enabled
    source.save(update_fields=["enabled"])
    return render(request, "ratings/_source_row.html", {"source": source})


@login_required
@require_POST
def source_delete(request, pk):
    get_object_or_404(Source, pk=pk).delete()
    return HttpResponse("")


@login_required
@require_POST
def source_import(request):
    from ratings.scraper import import_from_config

    n = import_from_config(Path(settings.CONFIG_PATH))
    return render(
        request,
        "ratings/_source_list.html",
        {"sources": Source.objects.all(), "imported": n},
    )


@login_required
def logs_page(request):
    show_nsfw = request.session.get("show_nsfw", False)
    entries = list(LogEntry.objects.order_by("-pk")[:500])
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
        },
    )


@login_required
def log_entries(request):
    try:
        since_id = int(request.GET.get("since", 0))
    except (ValueError, TypeError):
        since_id = 0
    entries = list(LogEntry.objects.filter(pk__gt=since_id).order_by("-pk")[:100])
    next_since = entries[-1].pk if entries else since_id
    return render(
        request,
        "ratings/_log_entries.html",
        {
            "entries": entries,
            "next_since": next_since,
        },
    )


@login_required
@require_POST
def log_clear(request):
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


# ── Corpus review ────────────────────────────────────────────────────────────


def _review_qs(show_nsfw: bool = False):
    qs = Image.objects.filter(
        location__in=[Image.INBOX, Image.CORPUS],
        score__isnull=True,
        is_purged=False,
        file_deleted=False,
    )
    if not show_nsfw:
        qs = qs.filter(is_nsfw=False)
    # Unseen images (queue_seen_at IS NULL) sort first in SQLite ASC; then oldest-downloaded.
    return qs.order_by("queue_seen_at", "downloaded_at")


def _review_nsfw_qs():
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


@login_required
def review_corpus(request, content_hash: str | None = None):
    show_nsfw = request.session.get("show_nsfw", False)
    ctx = _review_ctx(content_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    if request.htmx:
        return render(request, "ratings/_review_htmx.html", ctx)
    return render(request, "ratings/review.html", ctx)


@login_required
@require_POST
def score_corpus(request, content_hash: str):
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )

    # Capture next position before the score changes ordering.
    all_hashes = list(_review_qs(show_nsfw).values_list("content_hash", flat=True))
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

    ctx = _review_ctx(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


@login_required
@require_POST
def trash_corpus(request, content_hash: str):
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )

    # Capture neighbour before removing from queue.
    next_hash = _neighbor_hash(
        list(_review_qs(show_nsfw).values_list("content_hash", flat=True)),
        content_hash,
    )

    _move_image(image, Image.VOID)
    image.is_favourite = False
    image.rated_at = timezone.now()
    image.save(update_fields=["file_path", "location", "is_favourite", "rated_at"])

    ctx = _review_ctx(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


@login_required
@require_POST
def toggle_fav_corpus(request, content_hash: str):
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )
    image.is_favourite = not image.is_favourite
    image.save(update_fields=["is_favourite"])
    ctx = _review_ctx(content_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


@login_required
@require_POST
def purge_corpus(request, content_hash: str):
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )

    next_hash = _neighbor_hash(
        list(_review_qs(show_nsfw).values_list("content_hash", flat=True)),
        content_hash,
    )

    _purge_image(image)
    ctx = _review_ctx(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


# ── NSFW corpus review (same 1–6 + fav flow as normal corpus) ────────────────


@login_required
@require_POST
def score_nsfw_corpus(request, content_hash: str):
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )

    all_hashes = list(_review_nsfw_qs().values_list("content_hash", flat=True))
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

    ctx = _review_nsfw_ctx(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


@login_required
@require_POST
def trash_nsfw_corpus(request, content_hash: str):
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )

    next_hash = _neighbor_hash(
        list(_review_nsfw_qs().values_list("content_hash", flat=True)),
        content_hash,
    )

    _move_image(image, Image.VOID)
    image.is_favourite = False
    image.rated_at = timezone.now()
    image.save(update_fields=["file_path", "location", "is_favourite", "rated_at"])

    ctx = _review_nsfw_ctx(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


@login_required
@require_POST
def toggle_fav_nsfw_corpus(request, content_hash: str):
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )
    image.is_favourite = not image.is_favourite
    image.save(update_fields=["is_favourite"])
    ctx = _review_nsfw_ctx(content_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


@login_required
@require_POST
def purge_nsfw_corpus(request, content_hash: str):
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(
        Image, content_hash=content_hash, location__in=[Image.INBOX, Image.CORPUS]
    )

    next_hash = _neighbor_hash(
        list(_review_nsfw_qs().values_list("content_hash", flat=True)),
        content_hash,
    )

    _purge_image(image)
    ctx = _review_nsfw_ctx(next_hash, show_nsfw, request)
    _mark_queue_seen(ctx.get("image"))
    return render(request, "ratings/_review_htmx.html", ctx)


@login_required
@require_POST
def toggle_nsfw(request, content_hash: str):
    """Generic NSFW toggle for inbox / corpus / void images."""
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    location = image.location

    if location in (Image.INBOX, Image.CORPUS):
        mode = request.POST.get("mode", "corpus")
        if mode == "nsfw_corpus":
            # In the NSFW corpus queue toggling safe removes the image — always navigate.
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
    qs = Image.objects.filter(location=Image.VOID, file_deleted=False)
    if not show_nsfw:
        qs = qs.filter(is_nsfw=False)
    # Unseen images (void_seen_at IS NULL) sort first in SQLite ASC; then newest-trashed.
    return qs.order_by("void_seen_at", "-rated_at")


def _mark_void_seen(image: Image | None) -> None:
    if image is not None and image.void_seen_at is None:
        image.void_seen_at = timezone.now()
        image.save(update_fields=["void_seen_at"])


def _mark_queue_seen(image: Image | None) -> None:
    if image is not None and image.queue_seen_at is None:
        image.queue_seen_at = timezone.now()
        image.save(update_fields=["queue_seen_at"])


def _void_review_ctx(
    content_hash: str | None, show_nsfw: bool = False, request=None
) -> dict:
    return _browse_ctx(
        _void_review_qs(show_nsfw), content_hash, "void", show_nsfw, request
    )


@login_required
def gallery(request):
    show_nsfw = request.session.get("show_nsfw", False)

    try:
        min_score = max(1, min(6, int(request.GET.get("min_score", 1))))
    except (ValueError, TypeError):
        min_score = 1

    sort = request.GET.get("sort", "newest")
    if sort not in ("newest", "oldest", "random"):
        sort = "newest"

    fav_only = request.GET.get("fav") == "1"

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

    if sort == "random":
        qs = qs.order_by("?")
    elif sort == "oldest":
        qs = qs.order_by("downloaded_at")
    else:
        qs = qs.order_by("-downloaded_at")

    images = list(qs[:500])

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
        },
    )


@login_required
def review_void(request, content_hash: str | None = None):
    show_nsfw = request.session.get("show_nsfw", False)
    ctx = _void_review_ctx(content_hash, show_nsfw, request)
    _mark_void_seen(ctx.get("image"))
    if request.htmx:
        return render(request, "ratings/_void_htmx.html", ctx)
    return render(request, "ratings/void_review.html", ctx)


@login_required
@require_POST
def void_review_action(request, content_hash: str):
    show_nsfw = request.session.get("show_nsfw", False)
    action = request.POST.get("action", "")
    image = get_object_or_404(Image, content_hash=content_hash, location=Image.VOID)

    # Capture neighbour before any move changes the list.
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
        # When hiding NSFW from the queue, advance to neighbour.
        if not show_nsfw and image.is_nsfw:
            ctx = _void_review_ctx(next_hash, show_nsfw, request)
        else:
            ctx = _void_review_ctx(content_hash, show_nsfw, request)
    else:
        ctx = _void_review_ctx(content_hash, show_nsfw, request)

    _mark_void_seen(ctx.get("image"))
    return render(request, "ratings/_void_htmx.html", ctx)


@login_required
@require_POST
def gallery_action(request, content_hash: str):
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

    return JsonResponse({"score": image.score, "fav": image.is_favourite})
