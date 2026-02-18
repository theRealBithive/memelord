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
    IntegerField,
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
    _migrate_add_engagement_columns(_db)
    return _db


def _migrate_add_engagement_columns(database: SqliteDatabase) -> None:
    """Add Mastodon status ID and engagement columns if missing (for existing DBs)."""
    cursor = database.execute_sql("PRAGMA table_info(image)")
    columns = {row[1] for row in cursor.fetchall()}
    cursor.close()
    if "mastodon_status_id" not in columns:
        database.execute_sql(
            "ALTER TABLE image ADD COLUMN mastodon_status_id VARCHAR(32) NULL"
        )
    if "engagement_favourites" not in columns:
        database.execute_sql(
            "ALTER TABLE image ADD COLUMN engagement_favourites INTEGER NULL"
        )
    if "engagement_reblogs" not in columns:
        database.execute_sql(
            "ALTER TABLE image ADD COLUMN engagement_reblogs INTEGER NULL"
        )
    if "engagement_replies" not in columns:
        database.execute_sql(
            "ALTER TABLE image ADD COLUMN engagement_replies INTEGER NULL"
        )
    if "engagement_fetched_at" not in columns:
        database.execute_sql(
            "ALTER TABLE image ADD COLUMN engagement_fetched_at DATETIME NULL"
        )


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
    location: "inbox" (pending judge), "corpus" (positive), or "void" (negative).
    file_deleted: True if the file on disk was removed; row kept for reference.
    mastodon_status_id: ID of the Mastodon status after posting (for engagement).
    engagement_*: Fetched from Mastodon API; updated when we post or refresh.
    """

    content_hash = CharField(primary_key=True, max_length=64)
    file_path = CharField(max_length=2048)
    source_url = CharField(null=True, max_length=2048)
    source_label = CharField(max_length=255)
    location = CharField(max_length=32, default="corpus")  # "inbox" | "corpus" | "void"
    downloaded_at = DateTimeField(default=lambda: datetime.now(timezone.utc))
    posted_at = DateTimeField(null=True)
    file_deleted = BooleanField(default=False)
    mastodon_status_id = CharField(null=True, max_length=32)
    engagement_favourites = IntegerField(null=True)
    engagement_reblogs = IntegerField(null=True)
    engagement_replies = IntegerField(null=True)
    engagement_fetched_at = DateTimeField(null=True)

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


def count_unposted_corpus_images(data_root: Path | str) -> int:
    """
    Return the number of corpus images not yet posted (and with file on disk).

    Same eligibility as get_random_unposted_corpus_image: location corpus,
    posted_at null, file_deleted false, file exists. Call init_db first.
    """
    root = Path(data_root)
    candidates = list(
        Image.select().where(
            Image.location == "corpus",
            Image.posted_at.is_null(),
            Image.file_deleted == False,
        )
    )
    return sum(
        1 for row in candidates if resolve_file_path(root, row.file_path).exists()
    )


def get_posted_images_with_status():  # noqa: ANN201
    """
    Return all Image rows that have been posted and have a Mastodon status ID.

    Used to refresh engagement counts. Call init_db first.
    """
    return list(
        Image.select().where(
            Image.posted_at.is_null(False),
            Image.mastodon_status_id.is_null(False),
        )
    )


def get_top_posted_by_engagement(limit: int = 10):  # noqa: ANN201
    """
    Return posted images with status ID, sorted by favourites + reblogs descending.

    Call init_db first. Returns at most `limit` rows.
    """
    total = fn.COALESCE(Image.engagement_favourites, 0) + fn.COALESCE(
        Image.engagement_reblogs, 0
    )
    return list(
        Image.select()
        .where(
            Image.posted_at.is_null(False),
            Image.mastodon_status_id.is_null(False),
        )
        .order_by(total.desc())
        .limit(limit)
    )


def engagement_weight(
    favourites: int,
    reblogs: int,
    replies: int,
) -> float:
    """
    Compute training weight from engagement: faves (+1), replies (+0.5), reblogs (+2).
    Used for sample_weight when retraining with posted images.
    """
    return float(favourites) + 0.5 * float(replies) + 2.0 * float(reblogs)


def get_posted_engagement_weights():  # noqa: ANN201
    """
    Return dict mapping file_path (normalized) -> engagement weight for training.

    Only includes posted corpus images (void images are not weighted). Call init_db first.
    Keys use forward slashes for portability.
    """
    rows = [
        r
        for r in get_posted_images_with_status()
        if (r.location or "").strip() == "corpus"
    ]
    out: dict[str, float] = {}
    for row in rows:
        f = row.engagement_favourites or 0
        r = row.engagement_reblogs or 0
        rep = row.engagement_replies or 0
        key = (row.file_path or "").strip().replace("\\", "/")
        if key:
            out[key] = engagement_weight(f, r, rep)
    return out


def update_image_engagement(
    row: Image,
    favourites: int,
    reblogs: int,
    replies: int,
) -> None:
    """
    Set engagement counts and engagement_fetched_at on an Image row and save.
    """
    row.engagement_favourites = favourites
    row.engagement_reblogs = reblogs
    row.engagement_replies = replies
    row.engagement_fetched_at = datetime.now(timezone.utc)
    row.save()


def _source_label_from_filename(path: Path) -> str:
    """Derive source_label from filename (e.g. funny_abc.jpg -> funny, wg_123.png -> wg)."""
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem or "unknown"


def insert_inbox_image(
    data_root: Path | str,
    path: Path,
    source_url: str | None,
    source_label: str,
) -> bool:
    """
    Insert a row for a newly downloaded image (location='inbox').

    Hashes the file; if content_hash already exists, returns False (skip).
    Otherwise creates a row with file_path relative to data_root.
    Returns True if inserted.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    content_hash = hashlib.sha256(raw).hexdigest()
    if Image.get_or_none(Image.content_hash == content_hash) is not None:
        return False
    root = Path(data_root)
    try:
        file_path_str = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        file_path_str = path.as_posix()
    Image.create(
        content_hash=content_hash,
        file_path=file_path_str,
        source_url=source_url,
        source_label=source_label,
        location="inbox",
        file_deleted=False,
    )
    return True


def record_judged_image(
    dest_path: Path,
    location: str,
    data_root: Path | None = None,
) -> None:
    """
    Record an image that was moved to corpus or void by the judge.

    If a row exists with this content_hash and location='inbox', updates file_path
    and location. Otherwise creates a new row (e.g. legacy inbox file without a row).
    """
    try:
        raw = dest_path.read_bytes()
    except OSError:
        return
    content_hash = hashlib.sha256(raw).hexdigest()
    row = Image.get_or_none(Image.content_hash == content_hash)
    if data_root is not None:
        try:
            file_path_str = str(dest_path.relative_to(data_root)).replace("\\", "/")
        except ValueError:
            file_path_str = dest_path.as_posix()
    else:
        file_path_str = dest_path.as_posix()
    if row is not None and row.location == "inbox":
        row.file_path = file_path_str
        row.location = location
        row.save()
        return
    if row is not None:
        return
    Image.create(
        content_hash=content_hash,
        file_path=file_path_str,
        source_url=None,
        source_label=_source_label_from_filename(dest_path),
        location=location,
        file_deleted=False,
    )


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
                file_path_str = str(path.relative_to(data_dir)).replace("\\", "/")
            except ValueError:
                file_path_str = path.as_posix()
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


def cleanup_void_files(db_path: Path | str, data_root: Path | str) -> int:
    """
    Delete from disk all files for images in the void.
    Sets file_deleted=True for each removed file. Rows are kept for dedup.
    Posted corpus images are left on disk (for Phase II engagement / retraining).
    file_path is resolved against data_root (relative paths). Returns files removed.
    """
    init_db(db_path)
    root = Path(data_root)
    removed = 0
    candidates = Image.select().where(
        Image.location == "void",
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


def sync_db_to_filesystem(
    data_root: Path | str, db_path: Path | str
) -> tuple[int, int]:
    """
    Align the DB with the filesystem: mark rows as file_deleted when the file
    is missing; update file_path and location when files were moved between
    corpus and void. When the same content exists in both dirs, corpus wins.

    Returns:
        (marked_deleted_count, updated_count).
    """
    from core import brain as brain_module

    init_db(db_path)
    root = Path(data_root)
    marked_deleted = 0
    updated = 0

    for row in Image.select().where(Image.file_deleted == False):
        path = resolve_file_path(root, row.file_path)
        if not path.is_file():
            row.file_deleted = True
            row.save()
            marked_deleted += 1

    corpus_dir = root / "corpus"
    void_dir = root / "void"
    corpus_hashes: set[str] = set()

    for dir_path, location in [(corpus_dir, "corpus"), (void_dir, "void")]:
        if not dir_path.is_dir():
            continue
        for path in sorted(dir_path.iterdir()):
            if not path.is_file() or not brain_module.is_image_path(path):
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            content_hash = hashlib.sha256(raw).hexdigest()
            if location == "corpus":
                corpus_hashes.add(content_hash)
            elif content_hash in corpus_hashes:
                continue
            row = Image.get_or_none(Image.content_hash == content_hash)
            if row is None:
                continue
            try:
                file_path_str = str(path.relative_to(root)).replace("\\", "/")
            except ValueError:
                file_path_str = path.as_posix()
            if row.file_path != file_path_str or row.location != location:
                row.file_path = file_path_str
                row.location = location
                row.file_deleted = False
                row.save()
                updated += 1

    return marked_deleted, updated
