import shutil
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from ratings.models import Image, Source

WEIGHTS_PATH = Path(settings.WEIGHTS_PATH)
DATA_DIR = Path(settings.DATA_DIR)


def index(request):
    return redirect("rate_inbox")


def _counts() -> dict:
    return Image.objects.filter(file_deleted=False).aggregate(
        inbox_count=Count("pk", filter=Q(location=Image.INBOX)),
        corpus_count=Count("pk", filter=Q(location=Image.CORPUS)),
        void_count=Count("pk", filter=Q(location=Image.VOID)),
    )


def _get_next(mode: str, exclude_hash: str | None = None) -> Image | None:
    qs = Image.objects.filter(file_deleted=False)
    if mode == "inbox":
        qs = qs.filter(location=Image.INBOX).order_by("downloaded_at")
    elif mode == "corpus":
        qs = qs.filter(location=Image.CORPUS).order_by("?")
    else:
        qs = qs.filter(location=Image.VOID).order_by("?")
    if exclude_hash:
        qs = qs.exclude(content_hash=exclude_hash)
    return qs.first()


def _build_ctx(mode: str, image: Image | None) -> dict:
    counts = _counts()
    return {"mode": mode, "image": image, "queue_count": counts[f"{mode}_count"], **counts}


def _mode_view(request, mode: str):
    return render(request, "ratings/rate.html", _build_ctx(mode, _get_next(mode)))


@login_required
def rate_inbox(request):
    return _mode_view(request, "inbox")


@login_required
def rate_corpus(request):
    return _mode_view(request, "corpus")


@login_required
def rate_void(request):
    return _mode_view(request, "void")


def _unique_dest(directory: Path, name: str) -> Path:
    dest = directory / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _move_image(image: Image, new_location: str) -> None:
    src = DATA_DIR / image.file_path
    dest_dir = DATA_DIR / new_location
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(dest_dir, src.name)
    try:
        shutil.move(str(src), str(dest))
    except FileNotFoundError:
        pass
    image.file_path = str(dest.relative_to(DATA_DIR))
    image.location = new_location


@login_required
@require_POST
def submit_rating(request, content_hash: str, action: str):
    mode = request.POST.get("mode", "inbox")
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

    next_image = _get_next(mode, exclude_hash=content_hash)
    ctx = _build_ctx(mode, next_image)
    template = "ratings/_card.html" if request.htmx else "ratings/rate.html"
    return render(request, template, ctx)


@login_required
def stats(request):
    counts = _counts()

    last_trained = None
    try:
        mtime = WEIGHTS_PATH.stat().st_mtime
        last_trained = datetime.fromtimestamp(mtime, tz=timezone.utc)
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
        "last_trained": last_trained,
        "source_breakdown": source_breakdown,
        "favs": favs,
    })


@login_required
@require_POST
def trigger_scrape(request):
    from ratings import scraper

    try:
        counts = scraper.run(config_path=Path(settings.CONFIG_PATH), data_dir=DATA_DIR)
        ctx = {"ok": True, "total": sum(counts.values()), "counts": counts}
    except Exception as exc:
        ctx = {"ok": False, "error": str(exc)}
    return render(request, "ratings/_scrape_result.html", ctx)


@login_required
@require_POST
def trigger_train(request):
    from core import trainer

    counts = _counts()
    try:
        trainer.run(data_dir=DATA_DIR, weights_path=WEIGHTS_PATH)
        ctx = {"ok": True, "corpus_n": counts["corpus_count"], "void_n": counts["void_count"]}
    except SystemExit:
        ctx = {
            "ok": False,
            "corpus_n": counts["corpus_count"],
            "void_n": counts["void_count"],
            "error": "Need both corpus and void images to train.",
        }
    return render(request, "ratings/_train_result.html", ctx)


@login_required
def config_view(request):
    return render(request, "ratings/config.html", {"sources": Source.objects.all()})


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
def source_import(request):
    from ratings.scraper import import_from_config
    n = import_from_config(Path(settings.CONFIG_PATH))
    return render(request, "ratings/_source_list.html", {"sources": Source.objects.all(), "imported": n})
