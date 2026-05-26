from pathlib import Path

from django.conf import settings

from ratings.models import Image


def purge_image(image: Image) -> None:
    """
    Hard-delete the file from disk and mark the DB record as purged.

    is_purged=True keeps the content_hash in the dedup index so the same image
    can't be re-downloaded and re-presented to the user after a scrape.
    """
    abs_path = settings.DATA_DIR / image.file_path
    if abs_path.exists():
        abs_path.unlink()
    image.is_purged = True
    image.save(update_fields=["is_purged"])
