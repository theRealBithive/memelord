import shutil
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from ratings.models import Image

WEIGHTS_PATH = Path(settings.WEIGHTS_PATH)

DATA_DIR = Path(settings.DATA_DIR)


def index(request):
    return redirect("rate_inbox")


def _counts() -> dict:
    return {
        "inbox_count": Image.objects.filter(location=Image.INBOX, file_deleted=False).count(),
        "corpus_count": Image.objects.filter(location=Image.CORPUS, file_deleted=False).count(),
        "void_count": Image.objects.filter(location=Image.VOID, file_deleted=False).count(),
    }


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
    queue_count = counts[f"{mode}_count"]
    return {"mode": mode, "image": image, "queue_count": queue_count, **counts}


def _mode_view(request, mode: str):
    image = _get_next(mode)
    return render(request, "ratings/rate.html", _build_ctx(mode, image))


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
    parts = name.rsplit(".", 1)
    stem, suffix = (parts[0], parts[1]) if len(parts) == 2 else (name, "")
    i = 1
    while True:
        candidate = directory / (f"{stem}_{i}.{suffix}" if suffix else f"{stem}_{i}")
        if not candidate.exists():
            return candidate
        i += 1


def _move_image(image: Image, new_location: str) -> None:
    src = DATA_DIR / image.file_path
    dest_dir = DATA_DIR / new_location
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(dest_dir, src.name)
    if src.exists():
        shutil.move(str(src), str(dest))
    image.file_path = str(dest.relative_to(DATA_DIR))
    image.location = new_location


@login_required
@require_POST
def submit_rating(request, content_hash: str, action: str):
    mode = request.POST.get("mode", "inbox")
    image = get_object_or_404(Image, content_hash=content_hash)
    now = timezone.now()

    if action == "good":
        if image.location != Image.CORPUS:
            _move_image(image, Image.CORPUS)
        image.is_favourite = False
        image.rated_at = now
        image.save()
    elif action == "fav":
        if image.location != Image.CORPUS:
            _move_image(image, Image.CORPUS)
        image.is_favourite = True
        image.rated_at = now
        image.save()
    elif action == "bad":
        if image.location != Image.VOID:
            _move_image(image, Image.VOID)
        image.is_favourite = False
        image.rated_at = now
        image.save()
    # skip: no change

    next_image = _get_next(mode, exclude_hash=content_hash)
    ctx = _build_ctx(mode, next_image)
    template = "ratings/_card.html" if request.htmx else "ratings/rate.html"
    return render(request, template, ctx)


@login_required
def stats(request):
    from datetime import datetime, timezone
    from django.db.models import Count

    counts = _counts()

    weights_path = WEIGHTS_PATH
    last_trained = None
    if weights_path.exists():
        mtime = weights_path.stat().st_mtime
        last_trained = datetime.fromtimestamp(mtime, tz=timezone.utc)

    source_breakdown = (
        Image.objects.filter(location=Image.CORPUS, file_deleted=False)
        .values("source_label")
        .annotate(n=Count("content_hash"))
        .order_by("-n")
    )

    favs = Image.objects.filter(
        location=Image.CORPUS, is_favourite=True, file_deleted=False
    ).order_by("-rated_at")[:24]

    ctx = {
        **counts,
        "last_trained": last_trained,
        "source_breakdown": source_breakdown,
        "favs": favs,
    }
    return render(request, "ratings/stats.html", ctx)


@login_required
@require_POST
def trigger_scrape(request):
    from ratings import scraper

    try:
        counts = scraper.run(
            config_path=Path(settings.CONFIG_PATH),
            data_dir=DATA_DIR,
        )
        total = sum(counts.values())
        ctx = {"ok": True, "total": total, "counts": counts}
    except Exception as exc:
        ctx = {"ok": False, "error": str(exc)}

    return render(request, "ratings/_scrape_result.html", ctx)


@login_required
@require_POST
def trigger_train(request):
    from core import trainer

    counts = _counts()
    corpus_n = counts["corpus_count"]
    void_n = counts["void_count"]

    try:
        trainer.run(data_dir=DATA_DIR, weights_path=WEIGHTS_PATH)
        weights_mtime = (
            WEIGHTS_PATH.stat().st_mtime if WEIGHTS_PATH.exists() else None
        )
        ctx = {
            "ok": True,
            "corpus_n": corpus_n,
            "void_n": void_n,
            "weights_path": str(WEIGHTS_PATH),
        }
    except SystemExit:
        ctx = {
            "ok": False,
            "corpus_n": corpus_n,
            "void_n": void_n,
            "error": "Need both corpus and void images to train.",
        }

    return render(request, "ratings/_train_result.html", ctx)
