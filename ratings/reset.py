"""Fresh start: wipe the image library, keep the configuration (UI contract V12)."""

from __future__ import annotations

from pathlib import Path

from django.db import transaction

from ratings.models import Image, LogEntry, Source

CONFIRM_WORD = "RESET"
IMAGE_DIR = "images"
THUMB_DIR = "thumbs"


def fresh_start(data_dir: Path, weights_paths: list[Path]) -> dict[str, int]:
    """
    Delete every image row and file, both classifier weight files and the
    scrape cursors; keep sources, channels, thresholds, schedule, users, the
    tag vocabulary and the log.

    Rows go first, inside one transaction, so a crash while deleting files
    leaves orphan files (harmless: re-scraped images get new names) rather
    than rows pointing at missing files. The whole images/ and thumbs/ trees
    are emptied instead of only the rows' paths so leftovers from earlier
    runs vanish too; a fresh start that keeps stray files is not fresh.
    Deleting Image rows cascades the image↔tag links; Tag rows are untouched.
    The log entry is written last so it describes what actually happened.
    """
    with transaction.atomic():
        image_count = Image.objects.count()
        Image.objects.all().delete()
        cursor_count = Source.objects.filter(cursor__isnull=False).update(cursor=None)
    file_count = _remove_files_below(data_dir / IMAGE_DIR)
    file_count += _remove_files_below(data_dir / THUMB_DIR)
    weights_count = 0
    for path in weights_paths:
        if path.exists():
            path.unlink()
            weights_count += 1
    LogEntry.objects.create(
        level="INFO",
        source="reset",
        message=(
            f"Fresh start: removed {image_count} image rows, {file_count} files and "
            f"{weights_count} classifier file(s); reset {cursor_count} scrape cursor(s). "
            "Sources, channels, thresholds, schedule, users and tags kept."
        ),
    )
    return {
        "images": image_count,
        "files": file_count,
        "weights": weights_count,
        "cursors": cursor_count,
    }


def _remove_files_below(directory: Path) -> int:
    """Delete every file under `directory` (directories stay) and return how many."""
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.rglob("*"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed
