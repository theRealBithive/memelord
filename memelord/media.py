"""Authenticated serving of image files under MEDIA_ROOT."""

import mimetypes
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, Http404

from ratings.models import Image

ALLOWED_PREFIXES = ("images/",)

# Thumbnails are stored under DATA_DIR/thumbs/ mirroring the original structure.
# 400px covers retina gallery grids without serving multi-megabyte originals.
_THUMB_SIZE = 400
_THUMB_DIR = "thumbs"


def _normalize_media_path(file_path: str) -> str:
    normalized = Path(file_path).as_posix().lstrip("/")
    if not normalized or ".." in normalized.split("/"):
        raise Http404()
    if not any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        raise Http404()
    return normalized


def _resolve_media_file(file_path: str) -> Path:
    normalized = _normalize_media_path(file_path)
    root = settings.MEDIA_ROOT.resolve()
    full = (settings.MEDIA_ROOT / normalized).resolve()
    if not full.is_relative_to(root) or not full.is_file():
        raise Http404()
    if Image.objects.filter(file_path=normalized, is_purged=True).exists():
        raise Http404()
    return full


def _thumbnail_path(normalized: str) -> Path:
    """
    Thumbnails sit in DATA_DIR/thumbs/ at the same relative path but with a
    .jpg extension. Mirroring the structure makes cache invalidation trivial:
    delete thumbs/images/foo.jpg when the original is purged.
    """
    p = Path(normalized)
    return settings.MEDIA_ROOT / _THUMB_DIR / p.parent / (p.stem + ".jpg")


def _generate_thumbnail(source: Path, dest: Path) -> None:
    """
    Resize so the longest side is at most _THUMB_SIZE pixels, then save as
    JPEG. JPEG is always used regardless of the source format so the gallery
    doesn't serve large PNGs or animated GIFs at thumbnail slots.
    """
    from PIL import Image as PilImage

    dest.parent.mkdir(parents=True, exist_ok=True)
    with PilImage.open(source) as img:
        img = img.convert("RGB")
        img.thumbnail((_THUMB_SIZE, _THUMB_SIZE), PilImage.Resampling.LANCZOS)
        img.save(dest, "JPEG", quality=82, optimize=True)


@login_required
def serve_media(request, file_path: str) -> FileResponse:
    """Serve an image from images/ for authenticated users only."""
    full = _resolve_media_file(file_path)
    content_type, _ = mimetypes.guess_type(str(full))
    return FileResponse(
        full.open("rb"),
        content_type=content_type or "application/octet-stream",
    )


@login_required
def serve_thumbnail(request, file_path: str) -> FileResponse:
    """
    Serve a cached 400px thumbnail. Generated on first request and stored in
    DATA_DIR/thumbs/ so subsequent requests skip PIL entirely. The original
    file is still validated through _resolve_media_file so auth and
    is_purged checks apply to thumbnails the same way as full-size media.
    """
    full = _resolve_media_file(file_path)
    normalized = _normalize_media_path(file_path)
    thumb = _thumbnail_path(normalized)
    if not thumb.exists():
        _generate_thumbnail(full, thumb)
    return FileResponse(thumb.open("rb"), content_type="image/jpeg")
