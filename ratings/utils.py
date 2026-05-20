import shutil
from pathlib import Path

from django.conf import settings

from ratings.models import Image


def _unique_dest(directory: Path, name: str) -> Path:
    """
    Return a non-conflicting destination path, appending _1, _2, … if needed.

    Image filenames embed the source (board name, blog hash) so collisions are
    rare but possible when two sources independently download the same filename
    — e.g. two 4chan boards that both have a post named 1234567890.jpg.
    """
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


def purge_image(image: Image) -> None:
    """
    Hard-delete the file from disk and mark the DB record as purged.

    Both file_deleted and is_purged are set so:
    - file_deleted=True excludes the image from normal queries (never shown again).
    - is_purged=True keeps the content_hash in the dedup index so the same URL
      can't be re-downloaded and re-presented to the user after a scrape.
    """
    abs_path = settings.DATA_DIR / image.file_path
    if abs_path.exists():
        abs_path.unlink()
    image.file_deleted = True
    image.is_purged = True
    image.save(update_fields=["file_deleted", "is_purged"])


def move_image(image: Image, new_location: str, data_dir: Path) -> None:
    """
    Move the image file between inbox/corpus/void directories and update the DB record.

    shutil.move is used instead of Path.rename because src and dst may be on
    different filesystems (e.g. if DATA_DIR is a bind-mounted Docker volume).
    Only file_path and location are updated here; the caller is responsible for
    saving these fields plus any additional fields (rated_at, is_favourite, etc.)
    in the same save() call to keep the DB and filesystem in sync.
    """
    src = data_dir / image.file_path
    dest_dir = data_dir / new_location
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(dest_dir, src.name)
    try:
        shutil.move(str(src), str(dest))
    except FileNotFoundError:
        pass
    image.file_path = str(dest.relative_to(data_dir))
    image.location = new_location
