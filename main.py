"""Janulon CLI: scrape sources and run the aesthetic pipeline (download + judge)."""

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import schedule
from loguru import logger
from retina import fourchan, image_validation, imgur, tumblr

from core import brain, caption, db, mastodon as mastodon_module

_INBOX_DIR = "inbox"
_CONFIG_DEFAULT = "config.toml"
_CORPUS_DIR = "corpus"
_VOID_DIR = "void"


def _load_config(config_path: Path) -> dict:
    """
    Load config.toml; return dict with "4chan", "tumblr", "imgur", "mastodon" keys.
    Sources use "boards"/"blogs"/"topics"; mastodon uses "base_url" and "access_token".
    """
    if not config_path.exists():
        logger.error(
            "Config file {} not found. Create it for --source all (see config.toml).",
            config_path.resolve(),
        )
        raise SystemExit(1)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    out: dict = {
        "4chan": {"boards": []},
        "tumblr": {"blogs": []},
        "imgur": {"topics": []},
        "mastodon": {"base_url": "", "access_token": ""},
    }
    for key in ("4chan", "tumblr", "imgur"):
        if key not in data or not isinstance(data[key], dict):
            continue
        section = data[key]
        if key == "4chan" and "boards" in section:
            raw = section["boards"]
            out[key]["boards"] = [str(x).strip() for x in raw if isinstance(x, str)]
        elif key == "tumblr" and "blogs" in section:
            raw = section["blogs"]
            out[key]["blogs"] = [str(x).strip() for x in raw if isinstance(x, str)]
        elif key == "imgur" and "topics" in section:
            raw = section["topics"]
            out[key]["topics"] = [str(x).strip() for x in raw if isinstance(x, str)]
    if "mastodon" in data and isinstance(data["mastodon"], dict):
        m = data["mastodon"]
        out["mastodon"]["base_url"] = str(m.get("base_url", "")).strip()
        out["mastodon"]["access_token"] = str(
            m.get("access_token") or os.environ.get("MASTODON_ACCESS_TOKEN", "")
        ).strip()
    return out


def _get_schedule_from_config(config_path: Path) -> dict:
    """
    Load [schedule] from config.toml. Reads scrape_every_hours, post_every_hours,
    retrain_every_hours (floats supported, e.g. 0.5 for 30 min).
    Returns scrape_every_minutes, post_every_minutes, retrain_every_minutes
    (defaults 360, 1440, 10080 if section missing). Cleanup is not scheduled;
    run `main cleanup` manually when needed.
    """
    defaults_h = {
        "scrape_every_hours": 6.0,
        "post_every_hours": 24.0,
        "retrain_every_hours": 168.0,  # weekly
    }
    if not config_path.exists():
        return {
            k.replace("_hours", "_minutes"): max(1, int(round(v * 60)))
            for k, v in defaults_h.items()
        }
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    if "schedule" not in data or not isinstance(data["schedule"], dict):
        return {
            k.replace("_hours", "_minutes"): max(1, int(round(v * 60)))
            for k, v in defaults_h.items()
        }
    s = data["schedule"]
    return {
        "scrape_every_minutes": max(
            1,
            int(
                round(
                    float(s.get("scrape_every_hours", defaults_h["scrape_every_hours"]))
                    * 60
                )
            ),
        ),
        "post_every_minutes": max(
            1,
            int(
                round(
                    float(s.get("post_every_hours", defaults_h["post_every_hours"]))
                    * 60
                )
            ),
        ),
        "retrain_every_minutes": max(
            1,
            int(
                round(
                    float(
                        s.get(
                            "retrain_every_hours",
                            defaults_h["retrain_every_hours"],
                        )
                    )
                    * 60
                )
            ),
        ),
    }


def _refresh_posted_engagement(
    client,
    *,
    exclude_content_hash: str | None = None,
) -> None:
    """
    Fetch current engagement from Mastodon for all posted images with a status ID,
    and update DB. Skips the row with content_hash == exclude_content_hash.
    Rate-limits to one request per second.
    """
    for row in db.get_posted_images_with_status():
        if (
            exclude_content_hash is not None
            and row.content_hash == exclude_content_hash
        ):
            continue
        try:
            eng = mastodon_module.fetch_status_engagement(
                client, row.mastodon_status_id
            )
            db.update_image_engagement(
                row,
                eng["favourites_count"],
                eng["reblogs_count"],
                eng["replies_count"],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not refresh engagement for status {}: {}",
                row.mastodon_status_id,
                e,
            )
        time.sleep(1)


def _log_top_posts_by_engagement(limit: int = 10) -> None:
    """Log the top N posted images by engagement (favourites + reblogs)."""
    top = db.get_top_posted_by_engagement(limit=limit)
    if not top:
        logger.info("No posted images with engagement data yet.")
        return
    logger.info("Top {} posts by engagement (faves + reblogs):", limit)
    for i, row in enumerate(top, 1):
        f = row.engagement_favourites or 0
        r = row.engagement_reblogs or 0
        replies = row.engagement_replies or 0
        total = f + r
        name = Path(row.file_path).name if row.file_path else row.content_hash[:12]
        logger.info(
            "  {:2}. {}  total={} (faves={} reblogs={} replies={})",
            i,
            name,
            total,
            f,
            r,
            replies,
        )


def _skip_dirs(data_dir: Path | None, output_folder: Path) -> list[Path]:
    """skip_dirs: data corpus/void + output corpus/void to avoid re-download."""
    out = []
    if data_dir is not None:
        out.extend([data_dir / "corpus", data_dir / "void"])
    out.extend([output_folder / _CORPUS_DIR, output_folder / _VOID_DIR])
    return out


def _run_4chan(
    board: str,
    inbox_dir: Path,
    index_pages: int,
    skip_dirs: list[Path],
    skip_paths: set[str] | None = None,
) -> None:
    logger.info(
        "Starting 4chan scrape: board=/{}/, inbox={}, index_pages={}",
        board,
        inbox_dir.resolve(),
        index_pages,
    )
    urls = fourchan.iter_image_urls(
        board=board,
        index_pages=index_pages,
    )
    if not urls:
        logger.warning("No image URLs found.")
        return
    paths = fourchan.download_images(
        urls,
        inbox_dir,
        board,
        skip_dirs=skip_dirs or None,
        skip_paths=skip_paths,
    )
    logger.success("Done. Downloaded {} images to {}", len(paths), inbox_dir.resolve())


def _run_tumblr(
    blog: str,
    inbox_dir: Path,
    num_posts: int,
    skip_dirs: list[Path],
    skip_paths: set[str] | None = None,
) -> None:
    logger.info(
        "Starting Tumblr scrape: blog={}, inbox={}, num_posts={}",
        blog,
        inbox_dir.resolve(),
        num_posts,
    )
    urls = tumblr.iter_image_urls(blog=blog, num_posts=num_posts)
    if not urls:
        logger.warning("No image URLs found.")
        return
    paths = tumblr.download_images(
        urls,
        inbox_dir,
        blog,
        skip_dirs=skip_dirs or None,
        skip_paths=skip_paths,
    )
    logger.success("Done. Downloaded {} images to {}", len(paths), inbox_dir.resolve())


def _run_imgur(
    topic: str,
    inbox_dir: Path,
    max_items: int,
    client_id: str,
    skip_dirs: list[Path],
    skip_paths: set[str] | None = None,
) -> None:
    logger.info(
        "Starting Imgur scrape: topic={}, inbox={}, max_items={}",
        topic,
        inbox_dir.resolve(),
        max_items,
    )
    urls = imgur.iter_image_urls(topic=topic, client_id=client_id, max_items=max_items)
    if not urls:
        logger.warning("No image URLs found.")
        return
    paths = imgur.download_images(
        urls,
        inbox_dir,
        topic,
        skip_dirs=skip_dirs or None,
        skip_paths=skip_paths,
    )
    logger.success("Done. Downloaded {} images to {}", len(paths), inbox_dir.resolve())


def _unique_dest(parent: Path, name: str) -> Path:
    """Return parent/name, or parent/name_2, name_3, ... if name exists."""
    p = parent / name
    if not p.exists():
        return p
    stem = Path(name).stem
    suffix = Path(name).suffix
    n = 2
    while (parent / f"{stem}_{n}{suffix}").exists():
        n += 1
    return parent / f"{stem}_{n}{suffix}"


def _remove_inbox_duplicates_by_hash(inbox_dir: Path) -> None:
    """Delete inbox files whose content hash is already in the database."""
    paths = sorted(
        p for p in inbox_dir.iterdir() if p.is_file() and brain.is_image_path(p)
    )
    paths = [p for p in paths if image_validation.is_readable_image(p)]
    removed = 0
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        h = hashlib.sha256(raw).hexdigest()
        if db.Image.get_or_none(db.Image.content_hash == h) is not None:
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        logger.info(
            "Removed {} inbox duplicates (content already in database).", removed
        )


def _insert_judged_image(
    dest_path: Path, location: str, data_root: Path | None = None
) -> None:
    """Insert a judged image (moved to corpus/void) into the database.

    file_path is stored relative to data_root when given (portable for Docker).
    """
    try:
        raw = dest_path.read_bytes()
    except OSError:
        return
    content_hash = hashlib.sha256(raw).hexdigest()
    if db.Image.get_or_none(db.Image.content_hash == content_hash) is not None:
        return
    if data_root is not None:
        try:
            file_path_str = str(dest_path.relative_to(data_root))
        except ValueError:
            file_path_str = str(dest_path.resolve())
    else:
        file_path_str = str(dest_path.resolve())
    db.Image.create(
        content_hash=content_hash,
        file_path=file_path_str,
        source_url=None,
        source_label=db._source_label_from_filename(dest_path),
        location=location,
        file_deleted=False,
    )


def _judge_and_sort(
    output_folder: Path,
    weights_path: Path,
    threshold: float,
) -> list[tuple[Path, str]]:
    """
    Encode inbox images, move to corpus (>= threshold) or void (< threshold).
    Returns list of (dest_path, location) for each moved file (for DB insert).
    """
    inbox_dir = output_folder / _INBOX_DIR
    corpus_dir = output_folder / _CORPUS_DIR
    void_dir = output_folder / _VOID_DIR
    inbox_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    void_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in inbox_dir.iterdir() if p.is_file() and brain.is_image_path(p)
    )
    paths = [p for p in paths if image_validation.is_readable_image(p)]
    if not paths:
        logger.info("No images in inbox to judge.")
        return []

    logger.info("Judging {} images with encoder + classifier", len(paths))
    encoder = brain.get_encoder()
    transform = brain.get_transform()
    classifier = brain.load_classifier(weights_path)
    embeddings = brain.encode(encoder, paths, transform=transform)
    probas = brain.predict_proba(classifier, embeddings)
    if not hasattr(probas, "__len__"):
        probas = [probas]
    elif len(probas) != len(paths):
        probas = list(probas)

    moved: list[tuple[Path, str]] = []
    for path, proba in zip(paths, probas):
        name = path.name
        if proba >= threshold:
            dest = _unique_dest(corpus_dir, name)
            path.rename(dest)
            moved.append((dest, "corpus"))
        else:
            dest = _unique_dest(void_dir, name)
            path.rename(dest)
            moved.append((dest, "void"))
    logger.success(
        "Judged {} images: {} → corpus, {} → void",
        len(paths),
        sum(1 for _, loc in moved if loc == "corpus"),
        sum(1 for _, loc in moved if loc == "void"),
    )
    return moved


def _run_schedule(
    config_path: Path,
    db_path: Path,
    weights_path: Path,
    output_folder: Path,
    data_dir: Path,
) -> None:
    """Run scrape, post, and retrain on intervals from config; exit on SIGTERM. Run cleanup manually when needed."""
    intervals = _get_schedule_from_config(config_path)
    scrape_m = intervals["scrape_every_minutes"]
    post_m = intervals["post_every_minutes"]
    retrain_m = intervals["retrain_every_minutes"]
    logger.info(
        "Schedule: scrape every {}m, post every {}m, retrain every {}m",
        scrape_m,
        post_m,
        retrain_m,
    )

    if not weights_path.exists():
        logger.info("No weights file at {}; running initial train.", weights_path)
        base_train = [
            sys.executable,
            "-m",
            "main",
            "train",
            "--data_dir",
            str(data_dir),
            "--weights",
            str(weights_path),
        ]
        if db_path.exists():
            base_train.extend(["--db", str(db_path)])
        subprocess.run(base_train, check=False)

    base_run = [
        sys.executable,
        "-m",
        "main",
        "run",
        "--source",
        "all",
        "--config",
        str(config_path),
        "--db",
        str(db_path),
        "--weights",
        str(weights_path),
        "--output_folder",
        str(output_folder),
        "--data_dir",
        str(data_dir),
    ]
    base_post = [
        sys.executable,
        "-m",
        "main",
        "post",
        "--config",
        str(config_path),
        "--db",
        str(db_path),
        "--data_dir",
        str(data_dir),
    ]
    base_train = [
        sys.executable,
        "-m",
        "main",
        "train",
        "--data_dir",
        str(data_dir),
        "--weights",
        str(weights_path),
    ]
    if db_path.exists():
        base_train.extend(["--db", str(db_path)])

    shutdown = False

    def on_signal(_signum: int, _frame: object) -> None:
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def job_run() -> None:
        logger.info("Scheduled run (scrape + judge)")
        subprocess.run(base_run, check=False)

    def job_post() -> None:
        logger.info("Scheduled post")
        subprocess.run(base_post, check=False)

    def job_retrain() -> None:
        logger.info("Scheduled retrain")
        subprocess.run(base_train, check=False)

    schedule.every(scrape_m).minutes.do(job_run)
    schedule.every(post_m).minutes.do(job_post)
    schedule.every(retrain_m).minutes.do(job_retrain)

    job_run()  # initial scrape at startup so there is something to post

    while not shutdown:
        schedule.run_pending()
        time.sleep(60)
    logger.info("Schedule shutting down")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Janulon: scrape image sources and filter by taste.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Scrape sources and optionally judge."
    )
    run_parser.add_argument(
        "--source",
        choices=["4chan", "all", "imgur", "reddit", "tumblr"],
        required=True,
        help="Source. 'all' = load config and run all listed boards/blogs/topics.",
    )
    run_parser.add_argument(
        "--board",
        default="wg",
        help="4chan board. Default: wg (wallpaper general).",
    )
    run_parser.add_argument(
        "--blog",
        help="Tumblr blog (tumblr only). Example: staff or blogname.tumblr.com",
    )
    run_parser.add_argument(
        "--topic",
        help="Imgur topic (imgur only). Example: funny (imgur.com/t/funny)",
    )
    run_parser.add_argument(
        "--imgur_client_id",
        default=os.environ.get("IMGUR_CLIENT_ID", ""),
        help="Imgur API Client ID (imgur only). Default: env IMGUR_CLIENT_ID",
    )
    run_parser.add_argument(
        "--max_items",
        type=int,
        default=120,
        metavar="N",
        help="Max image items (imgur only). Default: 120",
    )
    run_parser.add_argument(
        "--subreddit",
        help="Subreddit to scrape (reddit only).",
    )
    run_parser.add_argument(
        "--num_posts",
        type=int,
        default=50,
        metavar="N",
        help="Max posts to fetch (tumblr only). Default: 50",
    )
    run_parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("output"),
        help="Folder for inbox/ and judged corpus/void. Default: output",
    )
    run_parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Skip download if image in data/corpus or data/void. Default: data",
    )
    run_parser.add_argument(
        "--index_pages",
        type=int,
        default=2,
        metavar="N",
        help="Index pages to scrape (4chan only). Default: 2",
    )
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Score >= threshold → corpus, else void. Default: 0.85",
    )
    run_parser.add_argument(
        "--weights",
        type=Path,
        default=Path("Janulon_weights.pkl"),
        help="Path to trained classifier. Default: Janulon_weights.pkl",
    )
    run_parser.add_argument(
        "--no_judge",
        action="store_true",
        help="Only download; skip encoder/classifier and corpus/void move.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path(_CONFIG_DEFAULT),
        help="Config file for --source all (4chan boards, tumblr blogs, imgur topics).",
    )
    run_parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/janulon.db"),
        help="SQLite database for dedup and judged image index. Default: data/janulon.db",
    )

    import_parser = subparsers.add_parser(
        "import-data",
        help="Import images from data/corpus and data/void into the SQLite database.",
    )
    import_parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing corpus/ and void/. Default: data",
    )
    import_parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/janulon.db"),
        help="SQLite database path. Default: data/janulon.db",
    )

    post_parser = subparsers.add_parser(
        "post",
        help="Post a random unposted corpus image to Mastodon (with generated alt text).",
    )
    post_parser.add_argument(
        "--config",
        type=Path,
        default=Path(_CONFIG_DEFAULT),
        help="Config file with [mastodon] base_url and access_token.",
    )
    post_parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/janulon.db"),
        help="SQLite database. Default: data/janulon.db",
    )
    post_parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Data root (corpus/void live here). Default: data",
    )

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Remove from disk void images only; set file_deleted. Posted corpus images are kept.",
    )
    cleanup_parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/janulon.db"),
        help="SQLite database path. Default: data/janulon.db",
    )
    cleanup_parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Data root (corpus/void live here). Default: data",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train or retrain the classifier on data_dir/corpus and data_dir/void (optional --db for engagement weights).",
    )
    train_parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing corpus/ and void/. Default: data",
    )
    train_parser.add_argument(
        "--weights",
        type=Path,
        default=Path("Janulon_weights.pkl"),
        help="Output path for classifier weights. Default: Janulon_weights.pkl",
    )
    train_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional SQLite DB path for engagement-weighted retrain (faves +0.5*replies +2*reblogs).",
    )

    schedule_parser = subparsers.add_parser(
        "schedule",
        help="Run scrape, post, cleanup, and retrain on intervals from config [schedule].",
    )
    schedule_parser.add_argument(
        "--config",
        type=Path,
        default=Path(_CONFIG_DEFAULT),
        help="Config file (sources + [schedule] intervals).",
    )
    schedule_parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/janulon.db"),
        help="SQLite database path. Default: data/janulon.db",
    )
    schedule_parser.add_argument(
        "--weights",
        type=Path,
        default=Path("Janulon_weights.pkl"),
        help="Path to trained classifier (for run).",
    )
    schedule_parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("output"),
        help="Output folder for run (inbox/corpus/void).",
    )
    schedule_parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Data dir for run (skip_dirs).",
    )

    args = parser.parse_args()

    if args.command == "post":
        cfg = _load_config(Path(args.config))
        base_url = cfg["mastodon"]["base_url"]
        access_token = cfg["mastodon"]["access_token"]
        if not base_url or not access_token:
            logger.error(
                "Mastodon config missing: set [mastodon] base_url and access_token in {} "
                "or MASTODON_ACCESS_TOKEN env.",
                args.config,
            )
            raise SystemExit(1)
        db.init_db(args.db)
        row = db.get_random_unposted_corpus_image(args.data_dir)
        if row is None:
            logger.warning("No unposted corpus image found.")
            return
        path = db.resolve_file_path(args.data_dir, row.file_path)
        logger.info("Posting {} (source: {})", path.name, row.source_label)
        alt_text = caption.describe_for_alt(path)
        client = mastodon_module.create_client(base_url, access_token)
        status = mastodon_module.post_image(client, path, alt_text)
        status_id = (
            status.get("id")
            if isinstance(status, dict)
            else getattr(status, "id", None)
        )
        row.posted_at = datetime.now(timezone.utc)
        row.mastodon_status_id = str(status_id) if status_id is not None else None
        eng = mastodon_module.engagement_from_status(status)
        db.update_image_engagement(
            row,
            eng["favourites_count"],
            eng["reblogs_count"],
            eng["replies_count"],
        )
        logger.success("Posted and marked as posted: {}", path.name)
        _refresh_posted_engagement(client, exclude_content_hash=row.content_hash)
        _log_top_posts_by_engagement(limit=10)
        return

    if args.command == "import-data":
        inserted, skipped = db.import_data(args.data_dir, args.db)
        logger.info(
            "Import done: {} inserted, {} skipped (duplicate content hash).",
            inserted,
            skipped,
        )
        return

    if args.command == "cleanup":
        removed = db.cleanup_void_files(args.db, args.data_dir)
        logger.info("Cleanup removed {} void file(s) from disk.", removed)
        return

    if args.command == "train":
        from core import trainer as trainer_module

        trainer_module.run(
            data_dir=args.data_dir,
            weights_path=args.weights,
            db_path=args.db,
        )
        return

    if args.command == "schedule":
        _run_schedule(
            config_path=args.config,
            db_path=args.db,
            weights_path=args.weights,
            output_folder=args.output_folder,
            data_dir=args.data_dir,
        )
        return

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    inbox_dir = output_folder / _INBOX_DIR
    inbox_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = _skip_dirs(args.data_dir, output_folder)

    db.init_db(args.db)
    data_root = Path(args.data_dir)
    skip_paths = {
        str(db.resolve_file_path(data_root, r.file_path))
        for r in db.Image.select(db.Image.file_path).iterator()
    }

    if args.source == "4chan":
        _run_4chan(
            board=args.board,
            inbox_dir=inbox_dir,
            index_pages=args.index_pages,
            skip_dirs=skip_dirs,
            skip_paths=skip_paths,
        )
    elif args.source == "imgur":
        if not args.topic:
            logger.error("Imgur requires --topic (e.g. funny for imgur.com/t/funny).")
            raise SystemExit(1)
        client_id = (args.imgur_client_id or "").strip()
        if not client_id:
            logger.info("No Imgur Client ID; scraping topic pages only.")
        _run_imgur(
            topic=args.topic.strip(),
            inbox_dir=inbox_dir,
            max_items=args.max_items,
            client_id=client_id,
            skip_dirs=skip_dirs,
            skip_paths=skip_paths,
        )
    elif args.source == "tumblr":
        if not args.blog:
            logger.error("Tumblr requires --blog (e.g. staff or blogname.tumblr.com).")
            raise SystemExit(1)
        _run_tumblr(
            blog=args.blog.strip(),
            inbox_dir=inbox_dir,
            num_posts=args.num_posts,
            skip_dirs=skip_dirs,
            skip_paths=skip_paths,
        )
    elif args.source == "all":
        cfg = _load_config(Path(args.config))
        client_id = (args.imgur_client_id or "").strip()
        for board in cfg["4chan"]["boards"]:
            if board:
                _run_4chan(
                    board=board,
                    inbox_dir=inbox_dir,
                    index_pages=args.index_pages,
                    skip_dirs=skip_dirs,
                    skip_paths=skip_paths,
                )
        for topic in cfg["imgur"]["topics"]:
            if topic:
                _run_imgur(
                    topic=topic,
                    inbox_dir=inbox_dir,
                    max_items=args.max_items,
                    client_id=client_id,
                    skip_dirs=skip_dirs,
                    skip_paths=skip_paths,
                )
        for blog in cfg["tumblr"]["blogs"]:
            if blog:
                _run_tumblr(
                    blog=blog,
                    inbox_dir=inbox_dir,
                    num_posts=args.num_posts,
                    skip_dirs=skip_dirs,
                    skip_paths=skip_paths,
                )
    elif args.source == "reddit":
        logger.error("Reddit source is not implemented (API key required).")
        raise SystemExit(1)

    if not args.no_judge and args.source != "reddit":
        if not args.weights.exists():
            logger.warning(
                "Weights {} not found; run trainer or use --no_judge.",
                args.weights.resolve(),
            )
        else:
            _remove_inbox_duplicates_by_hash(inbox_dir)
            moved = _judge_and_sort(
                output_folder=output_folder,
                weights_path=args.weights,
                threshold=args.threshold,
            )
            for dest_path, location in moved:
                _insert_judged_image(dest_path, location, data_root=output_folder)


if __name__ == "__main__":
    main()
