"""Tests for core.db: database init and Image model."""

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from peewee import SqliteDatabase

from core import db
from tests.conftest import minimal_png_bytes


@pytest.fixture
def database():
    """Initialize DB with in-memory SQLite; use for tests that need the DB."""
    return db.init_db(":memory:")


def test_get_db_raises_before_init() -> None:
    """get_db raises RuntimeError if init_db has not been called."""
    saved = db._db
    try:
        db._db = None
        with pytest.raises(RuntimeError, match="not initialized"):
            db.get_db()
    finally:
        db._db = saved


def test_resolve_file_path_relative() -> None:
    """resolve_file_path joins data_root with relative file_path."""
    assert db.resolve_file_path("/data", "corpus/funny.jpg") == Path(
        "/data/corpus/funny.jpg"
    )
    assert db.resolve_file_path(Path("/app/data"), "void/x.png") == Path(
        "/app/data/void/x.png"
    )


def test_resolve_file_path_absolute() -> None:
    """resolve_file_path returns path as-is when file_path is absolute (backward compat)."""
    abs_path = "/var/lib/corpus/old.jpg"
    assert db.resolve_file_path("/data", abs_path) == Path(abs_path)


def test_init_db_creates_tables(database: SqliteDatabase) -> None:
    """init_db creates the image table."""
    cursor = database.execute_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='image'"
    )
    row = cursor.fetchone()
    cursor.close()
    assert row is not None
    assert row[0] == "image"


def test_image_create_and_retrieve(database: SqliteDatabase) -> None:
    """Creating an Image and retrieving by content_hash returns the same data."""
    db.Image.create(
        content_hash="a1b2c3d4",
        file_path="/data/corpus/funny_abc.jpg",
        source_label="funny",
        source_url="https://example.com/img.jpg",
    )
    row = db.Image.get_by_id("a1b2c3d4")
    assert row.content_hash == "a1b2c3d4"
    assert row.file_path == "/data/corpus/funny_abc.jpg"
    assert row.source_label == "funny"
    assert row.source_url == "https://example.com/img.jpg"
    assert row.posted_at is None
    assert row.downloaded_at is not None


def test_image_get_or_none_for_dedup(database: SqliteDatabase) -> None:
    """get_or_none by content_hash supports dedup check before insert."""
    existing = db.Image.create(
        content_hash="deadbeef",
        file_path="/data/corpus/existing.jpg",
        source_label="wg",
    )
    found = db.Image.get_or_none(db.Image.content_hash == "deadbeef")
    assert found is not None
    assert found.content_hash == existing.content_hash
    missing = db.Image.get_or_none(db.Image.content_hash == "nonexistent")
    assert missing is None


def test_image_posted_at_null_by_default(database: SqliteDatabase) -> None:
    """New images have posted_at set to None."""
    db.Image.create(
        content_hash="new1",
        file_path="/path/to/img.png",
        source_label="blog",
    )
    row = db.Image.get_by_id("new1")
    assert row.posted_at is None


def test_image_source_url_optional(database: SqliteDatabase) -> None:
    """source_url can be omitted (null)."""
    db.Image.create(
        content_hash="no_url",
        file_path="/data/void/img.jpg",
        source_label="tg",
        source_url=None,
    )
    row = db.Image.get_by_id("no_url")
    assert row.source_url is None


def test_image_location_corpus_or_void(database: SqliteDatabase) -> None:
    """location stores corpus or void."""
    db.Image.create(
        content_hash="c1",
        file_path="/data/corpus/a.jpg",
        source_label="funny",
        location="corpus",
    )
    db.Image.create(
        content_hash="v1",
        file_path="/data/void/b.jpg",
        source_label="wg",
        location="void",
    )
    assert db.Image.get_by_id("c1").location == "corpus"
    assert db.Image.get_by_id("v1").location == "void"


def test_image_location_defaults_to_corpus(database: SqliteDatabase) -> None:
    """New images default to location corpus."""
    db.Image.create(
        content_hash="default_loc",
        file_path="/path/to/img.jpg",
        source_label="blog",
    )
    assert db.Image.get_by_id("default_loc").location == "corpus"


def test_image_file_deleted_tracks_disk_removal(database: SqliteDatabase) -> None:
    """file_deleted defaults to False; can be set when file on disk is removed."""
    db.Image.create(
        content_hash="del1",
        file_path="/data/corpus/gone.jpg",
        source_label="tg",
    )
    row = db.Image.get_by_id("del1")
    assert row.file_deleted is False
    row.file_deleted = True
    row.save()
    assert db.Image.get_by_id("del1").file_deleted is True


def test_migrate_adds_location_and_file_deleted_to_existing_table(
    database: SqliteDatabase,
) -> None:
    """Migration adds location and file_deleted columns to tables created without them."""
    database.execute_sql("DROP TABLE image")
    database.execute_sql(
        """
        CREATE TABLE image (
            content_hash VARCHAR(64) PRIMARY KEY,
            file_path VARCHAR(2048),
            source_url VARCHAR(2048),
            source_label VARCHAR(255),
            downloaded_at DATETIME,
            posted_at DATETIME
        )
    """
    )
    db._migrate_add_location_and_file_deleted(database)
    db._migrate_add_engagement_columns(database)
    db.Image.create(
        content_hash="migrated",
        file_path="/data/corpus/m.jpg",
        source_label="funny",
        location="void",
        file_deleted=True,
    )
    row = db.Image.get_by_id("migrated")
    assert row.location == "void"
    assert row.file_deleted is True


def test_source_label_from_filename() -> None:
    """_source_label_from_filename takes prefix before first underscore."""
    assert db._source_label_from_filename(Path("funny_abc123.jpg")) == "funny"
    assert db._source_label_from_filename(Path("wg_1754153031540706.jpg")) == "wg"
    assert db._source_label_from_filename(Path("image.png")) == "image"
    assert db._source_label_from_filename(Path("a_b_c.webp")) == "a"


def test_insert_inbox_image_creates_row_with_location_inbox(
    database: SqliteDatabase, tmp_path: Path
) -> None:
    """insert_inbox_image creates a row with location=inbox and source_url/source_label."""
    root = tmp_path / "out"
    root.mkdir()
    (root / "inbox").mkdir()
    path = root / "inbox" / "wg_99.png"
    path.write_bytes(minimal_png_bytes())
    inserted = db.insert_inbox_image(root, path, "https://i.4cdn.org/wg/99.png", "wg")
    assert inserted is True
    row = db.Image.get_or_none(db.Image.location == "inbox")
    assert row is not None
    assert row.source_url == "https://i.4cdn.org/wg/99.png"
    assert row.source_label == "wg"
    assert "inbox" in row.file_path


def test_insert_inbox_image_skips_duplicate_hash(
    database: SqliteDatabase, tmp_path: Path
) -> None:
    """insert_inbox_image returns False when content_hash already exists."""
    root = tmp_path / "out"
    root.mkdir()
    (root / "inbox").mkdir()
    path = root / "inbox" / "wg_99.png"
    path.write_bytes(minimal_png_bytes())
    assert db.insert_inbox_image(root, path, "https://example.com/1.png", "wg") is True
    assert db.insert_inbox_image(root, path, "https://example.com/2.png", "wg") is False
    assert db.Image.select().where(db.Image.location == "inbox").count() == 1


def test_record_judged_image_updates_inbox_row(
    database: SqliteDatabase, tmp_path: Path
) -> None:
    """record_judged_image updates file_path and location when row has location=inbox."""
    root = tmp_path / "out"
    (root / "corpus").mkdir(parents=True)
    path = root / "corpus" / "wg_42.png"
    path.write_bytes(minimal_png_bytes())
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    db.Image.create(
        content_hash=content_hash,
        file_path="inbox/wg_42.png",
        source_url="https://i.4cdn.org/wg/42.png",
        source_label="wg",
        location="inbox",
        file_deleted=False,
    )
    db.record_judged_image(path, "corpus", data_root=root)
    row = db.Image.get_by_id(content_hash)
    assert row.file_path == "corpus/wg_42.png"
    assert row.location == "corpus"
    assert row.source_url == "https://i.4cdn.org/wg/42.png"
    assert row.source_label == "wg"


def test_record_judged_image_creates_row_when_no_inbox_row(
    database: SqliteDatabase, tmp_path: Path
) -> None:
    """record_judged_image creates a row with source from filename when no existing row."""
    root = tmp_path / "out"
    (root / "corpus").mkdir(parents=True)
    path = root / "corpus" / "wg_99.png"
    path.write_bytes(minimal_png_bytes())
    db.record_judged_image(path, "corpus", data_root=root)
    row = db.Image.select().where(db.Image.location == "corpus").first()
    assert row is not None
    assert row.source_url is None
    assert row.source_label == "wg"


def test_import_data_inserts_corpus_and_void(
    database: SqliteDatabase,
) -> None:
    """import_data hashes files and inserts rows with location from subdir."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir()
        (data_dir / "void").mkdir()
        (data_dir / "corpus" / "funny_one.jpg").write_bytes(minimal_png_bytes())
        (data_dir / "void" / "wg_two.png").write_bytes(b"different content for void")
        db_path = data_dir / "import.db"
        inserted, skipped = db.import_data(data_dir, db_path)
    assert inserted == 2
    assert skipped == 0
    row_corpus = db.Image.select().where(db.Image.location == "corpus").first()
    row_void = db.Image.select().where(db.Image.location == "void").first()
    assert row_corpus is not None
    assert "corpus" in row_corpus.file_path
    assert row_corpus.source_label == "funny"
    assert row_void is not None
    assert "void" in row_void.file_path
    assert row_void.source_label == "wg"


def test_import_data_skips_duplicate_hash(
    database: SqliteDatabase,
) -> None:
    """import_data skips a file when its content hash already exists."""
    same_content = minimal_png_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir()
        (data_dir / "void").mkdir()
        (data_dir / "corpus" / "first.jpg").write_bytes(same_content)
        (data_dir / "void" / "second.jpg").write_bytes(same_content)
        inserted, skipped = db.import_data(data_dir, data_dir / "dup.db")
    assert inserted == 1
    assert skipped == 1


def test_import_data_uses_is_image_path(
    database: SqliteDatabase,
) -> None:
    """import_data only imports files that pass is_image_path."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir()
        (data_dir / "corpus" / "image.jpg").write_bytes(minimal_png_bytes())
        (data_dir / "corpus" / "readme.txt").write_text("not an image")

        def only_png(p: Path) -> bool:
            return p.suffix.lower() == ".png"

        inserted, skipped = db.import_data(
            data_dir, data_dir / "filter.db", is_image_path=only_png
        )
    assert inserted == 0
    assert skipped == 0


def test_import_data_skips_missing_dirs(
    database: SqliteDatabase,
) -> None:
    """import_data does not fail when corpus or void dir is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir()
        (data_dir / "corpus" / "only.jpg").write_bytes(minimal_png_bytes())
        inserted, skipped = db.import_data(data_dir, data_dir / "nom.db")
    assert inserted == 1
    assert skipped == 0


def test_get_random_unposted_corpus_image_returns_none_when_none_eligible(
    database: SqliteDatabase,
) -> None:
    """get_random_unposted_corpus_image returns None when no corpus or all posted."""
    root = Path("/")
    assert db.get_random_unposted_corpus_image(root) is None
    db.Image.create(
        content_hash="posted1",
        file_path="/nonexistent/corpus/a.jpg",
        source_label="funny",
        location="corpus",
        posted_at=datetime.now(timezone.utc),
    )
    assert db.get_random_unposted_corpus_image(root) is None
    db.Image.create(
        content_hash="deleted1",
        file_path="/also/nonexistent/b.jpg",
        source_label="wg",
        location="corpus",
        file_deleted=True,
    )
    assert db.get_random_unposted_corpus_image(root) is None


def test_get_random_unposted_corpus_image_returns_one_when_file_exists(
    database: SqliteDatabase,
) -> None:
    """get_random_unposted_corpus_image returns an unposted corpus row when file exists."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = tmp_path / "corpus_img.png"
        path.write_bytes(minimal_png_bytes())
        db.Image.create(
            content_hash="unposted1",
            file_path="corpus_img.png",
            source_label="funny",
            location="corpus",
        )
        row = db.get_random_unposted_corpus_image(tmp_path)
        assert row is not None
        assert row.content_hash == "unposted1"
        assert row.posted_at is None
        assert db.resolve_file_path(tmp_path, row.file_path).exists()


def test_count_unposted_corpus_images_counts_only_unposted_corpus_with_existing_file(
    database: SqliteDatabase,
) -> None:
    """count_unposted_corpus_images returns count of corpus rows with posted_at null and file on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "corpus").mkdir()
        a = root / "corpus" / "a.png"
        b = root / "corpus" / "b.png"
        a.write_bytes(minimal_png_bytes())
        b.write_bytes(minimal_png_bytes())
        db.Image.create(
            content_hash="h1",
            file_path="corpus/a.png",
            source_label="wg",
            location="corpus",
        )
        db.Image.create(
            content_hash="h2",
            file_path="corpus/b.png",
            source_label="wg",
            location="corpus",
        )
        assert db.count_unposted_corpus_images(root) == 2
        row = db.Image.get(db.Image.content_hash == "h1")
        row.posted_at = datetime.now(timezone.utc)
        row.save()
        assert db.count_unposted_corpus_images(root) == 1
        # void and missing file are not counted
        db.Image.create(
            content_hash="h3",
            file_path="corpus/c.png",
            source_label="wg",
            location="corpus",
        )
        assert db.count_unposted_corpus_images(root) == 1  # c.png does not exist
        db.Image.create(
            content_hash="h4",
            file_path="void/d.png",
            source_label="wg",
            location="void",
        )
        assert db.count_unposted_corpus_images(root) == 1


def test_cleanup_void_files_removes_void_only_keeps_posted_corpus() -> None:
    """cleanup_void_files deletes void files only; posted and unposted corpus files are kept."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "janulon.db"
        db.init_db(db_path)
        posted_file = tmp_path / "posted.png"
        void_file = tmp_path / "void.png"
        corpus_file = tmp_path / "corpus.png"
        posted_file.write_bytes(minimal_png_bytes())
        void_file.write_bytes(minimal_png_bytes())
        corpus_file.write_bytes(minimal_png_bytes())
        db.Image.create(
            content_hash="p1",
            file_path=str(posted_file.resolve()),
            source_label="wg",
            location="corpus",
            posted_at=datetime.now(timezone.utc),
        )
        db.Image.create(
            content_hash="v1",
            file_path=str(void_file.resolve()),
            source_label="funny",
            location="void",
        )
        db.Image.create(
            content_hash="c1",
            file_path=str(corpus_file.resolve()),
            source_label="wg",
            location="corpus",
        )
        removed = db.cleanup_void_files(db_path, tmp_path)
        assert removed == 1
        assert posted_file.exists()
        assert not void_file.exists()
        assert corpus_file.exists()
        assert db.Image.get_by_id("p1").file_deleted is False
        assert db.Image.get_by_id("v1").file_deleted is True
        assert db.Image.get_by_id("c1").file_deleted is False


def test_cleanup_void_files_skips_missing_files() -> None:
    """cleanup_void_files does not update row when file already missing."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "janulon.db"
        db.init_db(db_path)
        db.Image.create(
            content_hash="gone",
            file_path=str((tmp_path / "nonexistent.png").resolve()),
            source_label="wg",
            location="void",
        )
        removed = db.cleanup_void_files(db_path, tmp_path)
        assert removed == 0
        assert db.Image.get_by_id("gone").file_deleted is False


def test_cleanup_void_files_skips_already_file_deleted() -> None:
    """cleanup_void_files does not process rows already marked file_deleted."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "janulon.db"
        db.init_db(db_path)
        f = tmp_path / "already_gone.png"
        f.write_bytes(minimal_png_bytes())
        db.Image.create(
            content_hash="del",
            file_path=str(f.resolve()),
            source_label="wg",
            location="void",
            file_deleted=True,
        )
        removed = db.cleanup_void_files(db_path, tmp_path)
        assert removed == 0
        assert f.exists()


def test_get_posted_images_with_status_returns_only_posted_with_status_id(
    database: SqliteDatabase,
) -> None:
    """get_posted_images_with_status returns rows that have posted_at and mastodon_status_id."""
    db.Image.create(
        content_hash="unposted",
        file_path="corpus/a.jpg",
        source_label="wg",
        location="corpus",
    )
    db.Image.create(
        content_hash="posted_no_id",
        file_path="corpus/b.jpg",
        source_label="wg",
        location="corpus",
        posted_at=datetime.now(timezone.utc),
    )
    db.Image.create(
        content_hash="posted_with_id",
        file_path="corpus/c.jpg",
        source_label="wg",
        location="corpus",
        posted_at=datetime.now(timezone.utc),
        mastodon_status_id="12345",
    )
    rows = db.get_posted_images_with_status()
    assert len(rows) == 1
    assert rows[0].content_hash == "posted_with_id"
    assert rows[0].mastodon_status_id == "12345"


def test_update_image_engagement_sets_and_persists(
    database: SqliteDatabase,
) -> None:
    """update_image_engagement sets engagement fields and saves."""
    db.Image.create(
        content_hash="eng1",
        file_path="corpus/x.jpg",
        source_label="wg",
        location="corpus",
        posted_at=datetime.now(timezone.utc),
        mastodon_status_id="999",
    )
    row = db.Image.get_by_id("eng1")
    assert row.engagement_favourites is None
    db.update_image_engagement(row, favourites=10, reblogs=2, replies=1)
    row2 = db.Image.get_by_id("eng1")
    assert row2.engagement_favourites == 10
    assert row2.engagement_reblogs == 2
    assert row2.engagement_replies == 1
    assert row2.engagement_fetched_at is not None


def test_engagement_weight_formula() -> None:
    """engagement_weight = faves + 0.5*replies + 2*reblogs."""
    assert db.engagement_weight(0, 0, 0) == 0.0
    assert db.engagement_weight(10, 0, 0) == 10.0
    assert db.engagement_weight(0, 2, 0) == 4.0
    assert db.engagement_weight(0, 0, 4) == 2.0
    assert db.engagement_weight(1, 1, 2) == 1.0 + 2.0 + 1.0  # 4.0


def test_get_posted_engagement_weights_returns_map(database: SqliteDatabase) -> None:
    """get_posted_engagement_weights returns file_path -> weight for posted with engagement."""
    now = datetime.now(timezone.utc)
    db.Image.create(
        content_hash="a",
        file_path="corpus/one.jpg",
        source_label="wg",
        location="corpus",
        posted_at=now,
        mastodon_status_id="1",
        engagement_favourites=2,
        engagement_reblogs=1,
        engagement_replies=0,
    )
    db.Image.create(
        content_hash="b",
        file_path="corpus/two.jpg",
        source_label="wg",
        location="corpus",
        posted_at=now,
        mastodon_status_id="2",
        engagement_favourites=0,
        engagement_reblogs=2,
        engagement_replies=2,
    )
    weights = db.get_posted_engagement_weights()
    assert weights == {"corpus/one.jpg": 4.0, "corpus/two.jpg": 5.0}  # 2+2*1; 2*2+0.5*2


def test_get_posted_engagement_weights_excludes_void(database: SqliteDatabase) -> None:
    """get_posted_engagement_weights only includes corpus images; void are not weighted."""
    now = datetime.now(timezone.utc)
    db.Image.create(
        content_hash="corpus_posted",
        file_path="corpus/keep.jpg",
        source_label="wg",
        location="corpus",
        posted_at=now,
        mastodon_status_id="1",
        engagement_favourites=1,
    )
    db.Image.create(
        content_hash="void_posted",
        file_path="void/drop.jpg",
        source_label="wg",
        location="void",
        posted_at=now,
        mastodon_status_id="2",
        engagement_favourites=5,
    )
    weights = db.get_posted_engagement_weights()
    assert list(weights.keys()) == ["corpus/keep.jpg"]
    assert weights["corpus/keep.jpg"] == 1.0


def test_sync_marks_missing_file_as_deleted() -> None:
    """sync_db_to_filesystem sets file_deleted=True when the file is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir(parents=True)
        (data_dir / "void").mkdir(parents=True)
        db_path = data_dir / "sync.db"
        db.init_db(db_path)
        db.Image.create(
            content_hash="missing123",
            file_path="corpus/nonexistent.jpg",
            source_label="wg",
            location="corpus",
            file_deleted=False,
        )
        marked, updated = db.sync_db_to_filesystem(data_dir, db_path)
    assert marked == 1
    assert updated == 0
    row = db.Image.get_by_id("missing123")
    assert row.file_deleted is True


def test_sync_updates_path_and_location_when_moved() -> None:
    """sync_db_to_filesystem updates file_path and location when file is in void."""
    png = minimal_png_bytes()
    content_hash = hashlib.sha256(png).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir(parents=True)
        (data_dir / "void").mkdir(parents=True)
        (data_dir / "void" / "moved.png").write_bytes(png)
        db_path = data_dir / "sync2.db"
        db.init_db(db_path)
        db.Image.create(
            content_hash=content_hash,
            file_path="corpus/was_here.png",
            source_label="wg",
            location="corpus",
            file_deleted=False,
        )
        marked, updated = db.sync_db_to_filesystem(data_dir, db_path)
    assert marked == 1
    assert updated == 1
    row = db.Image.get_by_id(content_hash)
    assert row.file_path == "void/moved.png"
    assert row.location == "void"
    assert row.file_deleted is False


def test_sync_same_hash_in_corpus_and_void_stays_corpus() -> None:
    """When the same file exists in both corpus and void, sync keeps location corpus."""
    png = minimal_png_bytes()
    content_hash = hashlib.sha256(png).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir(parents=True)
        (data_dir / "void").mkdir(parents=True)
        (data_dir / "corpus" / "a.png").write_bytes(png)
        (data_dir / "void" / "b.png").write_bytes(png)
        db_path = data_dir / "sync3.db"
        db.init_db(db_path)
        db.Image.create(
            content_hash=content_hash,
            file_path="corpus/a.png",
            source_label="wg",
            location="corpus",
            file_deleted=False,
        )
        marked, _ = db.sync_db_to_filesystem(data_dir, db_path)
    assert marked == 0
    row = db.Image.get_by_id(content_hash)
    assert row.location == "corpus"
    assert "corpus" in row.file_path


def test_get_top_posted_by_engagement_returns_sorted_by_faves_plus_reblogs(
    database: SqliteDatabase,
) -> None:
    """get_top_posted_by_engagement returns rows ordered by favourites + reblogs desc."""
    now = datetime.now(timezone.utc)
    for content_hash, faves, reblogs in [
        ("low", 1, 0),
        ("mid", 5, 2),
        ("high", 10, 3),
    ]:
        db.Image.create(
            content_hash=content_hash,
            file_path=f"corpus/{content_hash}.jpg",
            source_label="wg",
            location="corpus",
            posted_at=now,
            mastodon_status_id=content_hash + "id",
            engagement_favourites=faves,
            engagement_reblogs=reblogs,
        )
    top = db.get_top_posted_by_engagement(limit=10)
    assert [r.content_hash for r in top] == ["high", "mid", "low"]
    top2 = db.get_top_posted_by_engagement(limit=2)
    assert len(top2) == 2
    assert top2[0].content_hash == "high"
    assert top2[1].content_hash == "mid"
