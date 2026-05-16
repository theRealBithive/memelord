from datetime import datetime, timezone as dt_timezone
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from ratings.models import Image, LogEntry, Source
from ratings.utils import move_image as _move_image_util

WEIGHTS_PATH = Path(settings.WEIGHTS_PATH)
DATA_DIR = Path(settings.DATA_DIR)


def index(request):
    return redirect("rate_inbox")


def _counts(show_nsfw: bool = False) -> dict:
    qs = Image.objects.filter(file_deleted=False)
    if show_nsfw:
        return qs.aggregate(
            inbox_count=Count("pk", filter=Q(location=Image.INBOX)),
            corpus_count=Count("pk", filter=Q(location=Image.CORPUS)),
            void_count=Count("pk", filter=Q(location=Image.VOID)),
            fav_count=Count("pk", filter=Q(location=Image.CORPUS, is_favourite=True)),
            nsfw_inbox_count=Count("pk", filter=Q(location=Image.INBOX, is_nsfw=True)),
            nsfw_corpus_count=Count("pk", filter=Q(location=Image.CORPUS, is_nsfw=True)),
            nsfw_void_count=Count("pk", filter=Q(location=Image.VOID, is_nsfw=True)),
            nsfw_fav_count=Count("pk", filter=Q(location=Image.CORPUS, is_favourite=True, is_nsfw=True)),
        )
    return qs.aggregate(
        inbox_count=Count("pk", filter=Q(location=Image.INBOX, is_nsfw=False)),
        corpus_count=Count("pk", filter=Q(location=Image.CORPUS, is_nsfw=False)),
        void_count=Count("pk", filter=Q(location=Image.VOID, is_nsfw=False)),
        fav_count=Count("pk", filter=Q(location=Image.CORPUS, is_favourite=True, is_nsfw=False)),
        nsfw_inbox_count=Count("pk", filter=Q(location=Image.INBOX, is_nsfw=True)),
        nsfw_corpus_count=Count("pk", filter=Q(location=Image.CORPUS, is_nsfw=True)),
        nsfw_void_count=Count("pk", filter=Q(location=Image.VOID, is_nsfw=True)),
        nsfw_fav_count=Count("pk", filter=Q(location=Image.CORPUS, is_favourite=True, is_nsfw=True)),
    )


def _get_next(mode: str, exclude_hash: str | None = None, show_nsfw: bool = False) -> Image | None:
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

    if mode in ("inbox", "nsfw_inbox"):
        qs = qs.order_by("downloaded_at")
    else:
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
        from datetime import timezone as tz
        started_at = datetime.fromisoformat(started_at_str)
        elapsed = int((datetime.now(tz.utc) - started_at).total_seconds())
        if elapsed > 1800:
            request.session.pop("training_task_id", None)
            request.session.pop("training_started_at", None)
            return {"active_task_id": None, "training_elapsed": None}
    return {"active_task_id": task_id, "training_elapsed": _fmt_elapsed(elapsed)}


def _build_ctx(mode: str, image: Image | None, show_nsfw: bool = False, request=None) -> dict:
    counts = _counts(show_nsfw)
    ctx = {
        "mode": mode,
        "image": image,
        "queue_count": counts[f"{mode}_count"],
        "show_nsfw": show_nsfw,
        **counts,
    }
    if request is not None:
        ctx.update(_training_ctx(request))
    return ctx


def _mode_view(request, mode: str):
    show_nsfw = request.session.get("show_nsfw", False)
    return render(request, "ratings/rate.html", _build_ctx(mode, _get_next(mode, show_nsfw=show_nsfw), show_nsfw, request))


def _move_image(image: Image, new_location: str) -> None:
    _move_image_util(image, new_location, DATA_DIR)


@login_required
def rate_inbox(request):
    return _mode_view(request, "inbox")


@login_required
def rate_corpus(request):
    return _mode_view(request, "corpus")


@login_required
def rate_void(request):
    return _mode_view(request, "void")


@login_required
def rate_fav(request):
    return _mode_view(request, "fav")


@login_required
def rate_nsfw_inbox(request):
    return _mode_view(request, "nsfw_inbox")


@login_required
def rate_nsfw_corpus(request):
    return _mode_view(request, "nsfw_corpus")


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
        if image.location != Image.CORPUS:
            _move_image(image, Image.CORPUS)
        image.is_favourite = action == "fav"
        image.rated_at = now
        image.save(update_fields=["file_path", "location", "is_favourite", "rated_at"])
    elif action == "bad":
        if image.location != Image.VOID:
            _move_image(image, Image.VOID)
        image.is_favourite = False
        image.rated_at = now
        image.save(update_fields=["file_path", "location", "is_favourite", "rated_at"])
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
    template = "ratings/_card.html" if request.htmx else "ratings/rate.html"
    return render(request, template, ctx)


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

    source_breakdown = (
        Image.objects.filter(location=Image.CORPUS, file_deleted=False)
        .values("source_label")
        .annotate(n=Count("content_hash"))
        .order_by("-n")
    )
    favs = Image.objects.filter(
        location=Image.CORPUS, is_favourite=True, file_deleted=False
    ).order_by("-rated_at")[:24]

    return render(request, "ratings/stats.html", {
        **counts,
        **_training_ctx(request),
        "show_nsfw": show_nsfw,
        "last_trained": last_trained,
        "source_breakdown": source_breakdown,
        "favs": favs,
    })


@login_required
@require_POST
def trigger_scrape(request):
    from ratings import scraper

    try:
        counts = scraper.run(
            config_path=Path(settings.CONFIG_PATH),
            data_dir=DATA_DIR,
            weights_path=WEIGHTS_PATH,
        )
        ctx = {"ok": True, "total": sum(counts.values()), "counts": counts}
    except Exception as exc:
        ctx = {"ok": False, "error": str(exc)}
    return render(request, "ratings/_scrape_result.html", ctx)


@login_required
@require_POST
def trigger_train(request):
    from django_q.tasks import async_task
    if request.session.get("training_task_id"):
        ctx = _training_ctx(request)
        return render(request, "ratings/_train_pending.html", {
            "task_id": ctx["active_task_id"],
            "elapsed": ctx["training_elapsed"],
        })
    task_id = async_task("ratings.tasks.run_train")
    request.session["training_task_id"] = task_id
    request.session["training_started_at"] = timezone.now().isoformat()
    return render(request, "ratings/_train_pending.html", {"task_id": task_id, "elapsed": "0s"})


@login_required
def train_status(request, task_id: str):
    from django_q.tasks import fetch
    task = fetch(task_id)
    if task is None or task.stopped is None:
        started_at_str = request.session.get("training_started_at")
        elapsed = None
        if started_at_str:
            from datetime import timezone as tz
            started_at = datetime.fromisoformat(started_at_str)
            elapsed = int((datetime.now(tz.utc) - started_at).total_seconds())
        return render(request, "ratings/_train_pending.html", {
            "task_id": task_id,
            "elapsed": _fmt_elapsed(elapsed),
        })

    request.session.pop("training_task_id", None)
    request.session.pop("training_started_at", None)

    result = task.result or {}
    show_nsfw = request.session.get("show_nsfw", False)
    counts = _counts(show_nsfw)
    return render(request, "ratings/_train_result.html", {
        "ok": result.get("ok", False),
        "error": result.get("error", "Unknown error."),
        "corpus_n": counts["corpus_count"],
        "void_n": counts["void_count"],
    })


@login_required
def config_view(request):
    show_nsfw = request.session.get("show_nsfw", False)
    counts = _counts(show_nsfw)
    return render(request, "ratings/config.html", {
        "sources": Source.objects.all(),
        "show_nsfw": show_nsfw,
        **_training_ctx(request),
        **counts,
    })


@login_required
@require_POST
def source_add(request):
    stype = request.POST.get("type", "").strip()
    name = request.POST.get("name", "").strip()

    if stype not in dict(Source.TYPE_CHOICES):
        return render(request, "ratings/_source_error.html", {"error": "Invalid source type."})
    if not name:
        return render(request, "ratings/_source_error.html", {"error": "Name is required."})
    if stype == Source.PIXELFED and not name.startswith("http"):
        return render(request, "ratings/_source_error.html", {"error": "Pixelfed value must be a URL (https://…)."})

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
def source_nsfw_toggle(request, pk):
    source = get_object_or_404(Source, pk=pk)
    source.is_nsfw = not source.is_nsfw
    source.save(update_fields=["is_nsfw"])
    return render(request, "ratings/_source_row.html", {"source": source})


@login_required
@require_POST
def source_import(request):
    from ratings.scraper import import_from_config
    n = import_from_config(Path(settings.CONFIG_PATH))
    return render(request, "ratings/_source_list.html", {"sources": Source.objects.all(), "imported": n})


@login_required
def logs_page(request):
    show_nsfw = request.session.get("show_nsfw", False)
    entries = list(LogEntry.objects.order_by("pk")[:500])
    next_since = entries[-1].pk if entries else 0
    return render(request, "ratings/logs.html", {
        **_counts(show_nsfw),
        **_training_ctx(request),
        "show_nsfw": show_nsfw,
        "entries": entries,
        "next_since": next_since,
    })


@login_required
def log_entries(request):
    since_id = int(request.GET.get("since", 0))
    entries = list(LogEntry.objects.filter(pk__gt=since_id).order_by("pk")[:100])
    next_since = entries[-1].pk if entries else since_id
    return render(request, "ratings/_log_entries.html", {
        "entries": entries,
        "next_since": next_since,
    })


@login_required
@require_POST
def log_clear(request):
    LogEntry.objects.all().delete()
    return redirect("logs")
