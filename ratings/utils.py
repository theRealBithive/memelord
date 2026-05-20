import shutil
from pathlib import Path

from django.conf import settings

from ratings.models import Image


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


def purge_image(image: Image) -> None:
    abs_path = settings.DATA_DIR / image.file_path
    if abs_path.exists():
        abs_path.unlink()
    image.file_deleted = True
    image.is_purged = True
    image.save(update_fields=["file_deleted", "is_purged"])


def move_image(image: Image, new_location: str, data_dir: Path) -> None:
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
