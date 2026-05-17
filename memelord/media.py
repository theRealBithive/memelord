"""Authenticated serving of image files under MEDIA_ROOT."""

import mimetypes
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, Http404

from ratings.models import Image

ALLOWED_PREFIXES = ("inbox/", "corpus/", "void/")


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
    if Image.objects.filter(file_path=normalized, file_deleted=True).exists():
        raise Http404()
    return full


@login_required
def serve_media(request, file_path: str) -> FileResponse:
    """Serve an image from inbox/corpus/void for authenticated users only."""
    full = _resolve_media_file(file_path)
    content_type, _ = mimetypes.guess_type(str(full))
    return FileResponse(
        full.open("rb"),
        content_type=content_type or "application/octet-stream",
    )
