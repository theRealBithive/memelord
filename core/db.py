"""SQLite database and Image model for storing and indexing downloaded images.

Image file_path is stored relative to the data root (directory containing corpus/
and void/) so the DB is portable across hosts and Docker. Resolve at runtime
with resolve_file_path(data_root, row.file_path).
"""

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from peewee import (
    BooleanField,
    CharField,
    DateTimeField,
    Model,
    Proxy,
    SqliteDatabase,
    fn,
)

_db: SqliteDatabase | None = None
db_proxy = Proxy()


def resolve_file_path(data_root: Path | str, file_path: str) -> Path:
    """
    Resolve file_path to an absolute Path.

    If file_path is already absolute (e.g. from an older DB), return it as-is.
    Otherwise treat it as relative to data_root (portable storage for Docker).
    """
    p = file_path.strip()
    if not p:
        return Path(data_root) / ""
    if Path(p).is_absolute():
        return Path(p)
    return Path(data_root) / p


def get_db() -> SqliteDatabase:
    """Return the global database instance. Call init_db first."""
    if _db is None:
        raise RuntimeError("Database not initialized; call init_db() first")
    return _db


def init_db(db_path: Path | str) -> SqliteDatabase:
    """
    Initialize the database and create tables. Safe to call multiple times.

    Args:
        db_path: Path to the SQLite file (e.g. data/janulon.db).

    Returns:
        The database instance.
    """
    global _db
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _db = SqliteDatabase(str(path))
    _db.connect()
    db_proxy.initialize(_db)
    _db.create_tables([Image])
    _migrate_add_location_and_file_deleted(_db)
    return _db


def _migrate_add_location_and_file_deleted(database: SqliteDatabase) -> None:
    """Add location and file_deleted columns if missing (for existing DBs)."""
    cursor = database.execute_sql("PRAGMA table_info(image)")
    columns = {row[1] for row in cursor.fetchall()}
    cursor.close()
    if "location" not in columns:
        database.execute_sql(
            "ALTER TABLE image ADD COLUMN location VARCHAR(32) NOT NULL DEFAULT 'corpus'"
        )
    if "file_deleted" not in columns:
        database.execute_sql(
            "ALTER TABLE image ADD COLUMN file_deleted INTEGER NOT NULL DEFAULT 0"
        )


class BaseModel(Model):
    """Base model bound to the global SQLite database."""

    class Meta:
        database = db_proxy


class Image(BaseModel):
    """
    One row per distinct image content, keyed by content hash.

    Used for content deduplication and for the bot to track what has been posted.
    location: "corpus" (positive) or "void" (negative).
    file_deleted: True if the file on disk was removed; row kept for reference.
    """

    content_hash = CharField(primary_key=True, max_length=64)
    file_path = CharField(max_length=2048)
    source_url = CharField(null=True, max_length=2048)
    source_label = CharField(max_length=255)
    location = CharField(max_length=32, default="corpus")  # "corpus" | "void"
    downloaded_at = DateTimeField(default=lambda: datetime.now(timezone.utc))
    posted_at = DateTimeField(null=True)
    file_deleted = BooleanField(default=False)

    class Meta:
        table_name = "image"


def get_random_unposted_corpus_image(data_root: Path | str):  # noqa: ANN201
    """
    Return a random corpus image that has not been posted yet, or None.

    Only considers rows where location is "corpus", posted_at is NULL,
    file_deleted is False, and the file exists on disk. file_path is resolved
    against data_root (relative paths). Call init_db first.
    """
    root = Path(data_root)
    candidates = list(
        Image.select()
        .where(
            Image.location == "corpus",
            Image.posted_at.is_null(),
            Image.file_deleted == False,
        )
        .order_by(fn.Random())
        .limit(50)
    )
    for row in candidates:
        if resolve_file_path(root, row.file_path).exists():
            return row
    return None


def _source_label_from_filename(path: Path) -> str:
    """Derive source_label from filename (e.g. funny_abc.jpg -> funny, wg_123.png -> wg)."""
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem or "unknown"


def import_data(
    data_dir: Path | str,
    db_path: Path | str,
    *,
    is_image_path: Callable[[Path], bool] | None = None,
) -> tuple[int, int]:
    """
    Import images from data_dir/corpus and data_dir/void into the database.

    Each file is hashed (SHA-256); if the hash already exists, the file is skipped.
    location is set from the subdir (corpus or void). source_label is derived from
    the filename prefix before the first underscore.

    Args:
        data_dir: Root data directory containing corpus/ and void/ subdirs.
        db_path: Path to the SQLite database (created if missing).
        is_image_path: Predicate(path) -> bool; defaults to core.brain.is_image_path.

    Returns:
        (inserted_count, skipped_count) where skipped = already in DB by hash.
    """
    from core import brain as brain_module

    data_dir = Path(data_dir)
    db_path = Path(db_path)
    if is_image_path is None:
        is_image_path = brain_module.is_image_path

    init_db(db_path)
    corpus_dir = data_dir / "corpus"
    void_dir = data_dir / "void"
    inserted = 0
    skipped = 0

    for location, dir_path in [("corpus", corpus_dir), ("void", void_dir)]:
        if not dir_path.is_dir():
            continue
        for path in sorted(dir_path.iterdir()):
            if not path.is_file() or not is_image_path(path):
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            content_hash = hashlib.sha256(raw).hexdigest()
            if Image.get_or_none(Image.content_hash == content_hash) is not None:
                skipped += 1
                continue
            try:
                file_path_str = str(path.relative_to(data_dir))
            except ValueError:
                file_path_str = str(path.resolve())
            Image.create(
                content_hash=content_hash,
                file_path=file_path_str,
                source_url=None,
                source_label=_source_label_from_filename(path),
                location=location,
                file_deleted=False,
            )
            inserted += 1

    return inserted, skipped


def cleanup_posted_and_void_files(db_path: Path | str, data_root: Path | str) -> int:
    """
    Delete from disk all files for images that are posted or in the void.
    Sets file_deleted=True for each removed file. Rows are kept for dedup.
    file_path is resolved against data_root (relative paths). Returns files removed.
    """
    init_db(db_path)
    root = Path(data_root)
    removed = 0
    candidates = Image.select().where(
        (Image.posted_at.is_null(False) | (Image.location == "void")),
        Image.file_deleted == False,
    )
    for row in candidates:
        path = resolve_file_path(root, row.file_path)
        if path.is_file():
            try:
                path.unlink()
                row.file_deleted = True
                row.save()
                removed += 1
            except OSError:
                continue
    return removed
