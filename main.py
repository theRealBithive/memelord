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
from retina import fourchan, image_validation, imgur, pixelfed, tumblr

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
        "pixelfed": {"instance_base": "", "limit": 40, "access_token": ""},
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
    if "pixelfed" in data and isinstance(data["pixelfed"], dict):
        p = data["pixelfed"]
        out["pixelfed"]["instance_base"] = str(p.get("instance_base", "")).strip()
        out["pixelfed"]["limit"] = (
            int(p.get("limit", 40)) if p.get("limit") is not None else 40
        )
        out["pixelfed"]["access_token"] = str(
            p.get("access_token") or os.environ.get("PIXELFED_ACCESS_TOKEN", "")
        ).strip()
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
    retrain_every_hours, sync_every_hours (floats supported, e.g. 0.5 for 30 min).
    Returns *_every_minutes for each (defaults 360, 1440, 10080, 2880 if missing).
    Cleanup is not scheduled; run `main cleanup` manually when needed.
    """
    defaults_h = {
        "scrape_every_hours": 6.0,
        "post_every_hours": 24.0,
        "retrain_every_hours": 168.0,  # weekly
        "sync_every_hours": 48.0,
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
        "sync_every_minutes": max(
            1,
            int(
                round(
                    float(s.get("sync_every_hours", defaults_h["sync_every_hours"]))
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


def _post_run_summary(
    config_path: Path,
    corpus_count: int,
    void_count: int,
    unposted_count: int = 0,
) -> None:
    """Post an ominous artefacts summary to Mastodon (text-only status)."""
    if not config_path.exists():
        logger.warning("Config {} not found; skipping run summary post.", config_path)
        return
    cfg = _load_config(config_path)
    base_url = cfg["mastodon"]["base_url"]
    access_token = cfg["mastodon"]["access_token"]
    if not base_url or not access_token:
        logger.warning(
            "Mastodon config missing in {}; skipping run summary post.", config_path
        )
        return
    # Ominous wording: artefacts, corpus, void, vault/stock
    if corpus_count == 0 and void_count == 0:
        base = "Acquired new artefacts. None deemed worthy of the corpus; none cast into the void."
    else:
        base = (
            f"Acquired new artefacts. {corpus_count} have been added to the corpus "
            f"and {void_count} cast into the void."
        )
    if unposted_count > 0:
        status_text = f"{base} {unposted_count} remain in the vault, awaiting the hour."
    else:
        status_text = f"{base} The vault stands empty."
    try:
        client = mastodon_module.create_client(base_url, access_token)
        mastodon_module.post_status(client, status_text)
        logger.info(
            "Posted run summary to Mastodon: {}",
            status_text[:57] + "…" if len(status_text) > 60 else status_text,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to post run summary to Mastodon: {}", e)


def _post_retrain_summary(config_path: Path) -> None:
    """Post an ominous retrain notice to Mastodon (text-only status)."""
    if not config_path.exists():
        logger.warning(
            "Config {} not found; skipping retrain summary post.", config_path
        )
        return
    cfg = _load_config(config_path)
    base_url = cfg["mastodon"]["base_url"]
    access_token = cfg["mastodon"]["access_token"]
    if not base_url or not access_token:
        logger.warning(
            "Mastodon config missing in {}; skipping retrain summary post.", config_path
        )
        return
    status_text = "Pondering the means of discernment anew."
    try:
        client = mastodon_module.create_client(base_url, access_token)
        mastodon_module.post_status(client, status_text)
        logger.info("Posted retrain summary to Mastodon: {}", status_text)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to post retrain summary to Mastodon: {}", e)


def _post_sync_summary(config_path: Path) -> None:
    """Post an ominous sync notice to Mastodon (text-only status)."""
    if not config_path.exists():
        logger.warning("Config {} not found; skipping sync summary post.", config_path)
        return
    cfg = _load_config(config_path)
    base_url = cfg["mastodon"]["base_url"]
    access_token = cfg["mastodon"]["access_token"]
    if not base_url or not access_token:
        logger.warning(
            "Mastodon config missing in {}; skipping sync summary post.",
            config_path,
        )
        return
    status_text = "Alignment with reality restored."
    try:
        client = mastodon_module.create_client(base_url, access_token)
        mastodon_module.post_status(client, status_text)
        logger.info("Posted sync summary to Mastodon: {}", status_text)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to post sync summary to Mastodon: {}", e)


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
) -> list[tuple[Path, str, str]]:
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
        return []
    paths = fourchan.download_images(
        urls,
        inbox_dir,
        board,
        skip_dirs=skip_dirs or None,
        skip_paths=skip_paths,
    )
    logger.success("Done. Downloaded {} images to {}", len(paths), inbox_dir.resolve())
    return paths


def _run_tumblr(
    blog: str,
    inbox_dir: Path,
    num_posts: int,
    skip_dirs: list[Path],
    skip_paths: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    logger.info(
        "Starting Tumblr scrape: blog={}, inbox={}, num_posts={}",
        blog,
        inbox_dir.resolve(),
        num_posts,
    )
    urls = tumblr.iter_image_urls(blog=blog, num_posts=num_posts)
    if not urls:
        logger.warning("No image URLs found.")
        return []
    paths = tumblr.download_images(
        urls,
        inbox_dir,
        blog,
        skip_dirs=skip_dirs or None,
        skip_paths=skip_paths,
    )
    logger.success("Done. Downloaded {} images to {}", len(paths), inbox_dir.resolve())
    return paths


def _run_imgur(
    topic: str,
    inbox_dir: Path,
    max_items: int,
    client_id: str,
    skip_dirs: list[Path],
    skip_paths: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    logger.info(
        "Starting Imgur scrape: topic={}, inbox={}, max_items={}",
        topic,
        inbox_dir.resolve(),
        max_items,
    )
    urls = imgur.iter_image_urls(topic=topic, client_id=client_id, max_items=max_items)
    if not urls:
        logger.warning("No image URLs found.")
        return []
    paths = imgur.download_images(
        urls,
        inbox_dir,
        topic,
        skip_dirs=skip_dirs or None,
        skip_paths=skip_paths,
    )
    logger.success("Done. Downloaded {} images to {}", len(paths), inbox_dir.resolve())
    return paths


def _run_pixelfed(
    instance_base: str,
    limit: int,
    inbox_dir: Path,
    skip_dirs: list[Path],
    skip_paths: set[str] | None = None,
    access_token: str | None = None,
) -> list[tuple[Path, str, str]]:
    logger.info(
        "Starting Pixelfed scrape: instance={}, limit={}, inbox={}",
        instance_base,
        limit,
        inbox_dir.resolve(),
    )
    items = pixelfed.iter_image_items(
        instance_base, limit=limit, access_token=access_token
    )
    if not items:
        logger.warning("No image items found from Pixelfed timeline.")
        return []
    paths = pixelfed.download_images(
        items,
        inbox_dir,
        skip_dirs=skip_dirs or None,
        skip_paths=skip_paths,
    )
    logger.success("Done. Downloaded {} images to {}", len(paths), inbox_dir.resolve())
    return paths


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
    """
    Delete inbox files whose content is already in corpus or void.

    Only removes when the existing row has location in ('corpus', 'void'), so we
    don't delete files that were just inserted as inbox (which would remove
    everything before the judge runs).
    """
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
        row = db.Image.get_or_none(db.Image.content_hash == h)
        if row is not None and row.location in ("corpus", "void"):
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        logger.info(
            "Removed {} inbox duplicates (content already in corpus/void).",
            removed,
        )


def _insert_inbox_downloads(
    downloaded: list[tuple[Path, str, str]],
    output_folder: Path,
) -> int:
    """Insert a DB row for each downloaded image (location='inbox'). Returns count inserted."""
    inserted = 0
    for path, source_url, source_label in downloaded:
        if db.insert_inbox_image(output_folder, path, source_url, source_label):
            inserted += 1
    return inserted


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
    sync_m = intervals["sync_every_minutes"]
    logger.info(
        "Schedule: scrape every {}m, post every {}m, retrain every {}m, sync every {}m",
        scrape_m,
        post_m,
        retrain_m,
        sync_m,
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
        "--post_summary",
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
        "--config",
        str(config_path),
        "--post_summary",
    ]
    if db_path.exists():
        base_train.extend(["--db", str(db_path)])

    base_sync = [
        sys.executable,
        "-m",
        "main",
        "sync",
        "--db",
        str(db_path),
        "--data_dir",
        str(data_dir),
        "--config",
        str(config_path),
        "--post_summary",
    ]

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

    def job_sync() -> None:
        logger.info("Scheduled sync")
        subprocess.run(base_sync, check=False)

    schedule.every(scrape_m).minutes.do(job_run)
    schedule.every(post_m).minutes.do(job_post)
    schedule.every(retrain_m).minutes.do(job_retrain)
    schedule.every(sync_m).minutes.do(job_sync)

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
        choices=["4chan", "all", "imgur", "pixelfed", "reddit", "tumblr"],
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
    run_parser.add_argument(
        "--post_summary",
        action="store_true",
        help="Post an artefacts summary to Mastodon after judging (requires config with [mastodon]).",
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

    sync_parser = subparsers.add_parser(
        "sync",
        help="Align DB with filesystem: mark missing files, update path/location if moved.",
    )
    sync_parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/janulon.db"),
        help="SQLite database path. Default: data/janulon.db",
    )
    sync_parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Data root (corpus/void live here). Default: data",
    )
    sync_parser.add_argument(
        "--config",
        type=Path,
        default=Path(_CONFIG_DEFAULT),
        help="Config for --post_summary (Mastodon). Default: config.toml",
    )
    sync_parser.add_argument(
        "--post_summary",
        action="store_true",
        help="Post a sync notice to Mastodon after syncing (requires [mastodon]).",
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
    train_parser.add_argument(
        "--config",
        type=Path,
        default=Path(_CONFIG_DEFAULT),
        help="Config file for --post_summary (Mastodon). Default: config.toml",
    )
    train_parser.add_argument(
        "--post_summary",
        action="store_true",
        help="Post a retrain notice to Mastodon after training (requires config with [mastodon]).",
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
        client = mastodon_module.create_client(base_url, access_token)
        if row.source_label == "pixelfed" and (row.source_url or "").strip():
            status_id = mastodon_module.resolve_remote_url(
                client, row.source_url.strip()
            )
            if status_id is None:
                logger.error(
                    "Could not resolve Pixelfed post URL on Mastodon: {}",
                    row.source_url,
                )
                raise SystemExit(1)
            status = mastodon_module.boost_status(client, status_id)
        else:
            alt_text = caption.describe_for_alt(path)
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

    if args.command == "sync":
        marked, updated = db.sync_db_to_filesystem(args.data_dir, args.db)
        logger.info(
            "Sync: {} entries marked missing, {} entries updated path/location.",
            marked,
            updated,
        )
        if args.post_summary:
            _post_sync_summary(Path(args.config))
        return

    if args.command == "train":
        from core import trainer as trainer_module

        trainer_module.run(
            data_dir=args.data_dir,
            weights_path=args.weights,
            db_path=args.db,
        )
        if args.post_summary:
            _post_retrain_summary(Path(args.config))
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
    output_folder_resolved = output_folder.resolve()
    skip_paths = {
        str(db.resolve_file_path(output_folder_resolved, r.file_path))
        for r in db.Image.select(db.Image.file_path).iterator()
    }

    if args.source == "4chan":
        downloaded = _run_4chan(
            board=args.board,
            inbox_dir=inbox_dir,
            index_pages=args.index_pages,
            skip_dirs=skip_dirs,
            skip_paths=skip_paths,
        )
        _insert_inbox_downloads(downloaded, output_folder)
    elif args.source == "imgur":
        if not args.topic:
            logger.error("Imgur requires --topic (e.g. funny for imgur.com/t/funny).")
            raise SystemExit(1)
        client_id = (args.imgur_client_id or "").strip()
        if not client_id:
            logger.info("No Imgur Client ID; scraping topic pages only.")
        downloaded = _run_imgur(
            topic=args.topic.strip(),
            inbox_dir=inbox_dir,
            max_items=args.max_items,
            client_id=client_id,
            skip_dirs=skip_dirs,
            skip_paths=skip_paths,
        )
        _insert_inbox_downloads(downloaded, output_folder)
    elif args.source == "tumblr":
        if not args.blog:
            logger.error("Tumblr requires --blog (e.g. staff or blogname.tumblr.com).")
            raise SystemExit(1)
        downloaded = _run_tumblr(
            blog=args.blog.strip(),
            inbox_dir=inbox_dir,
            num_posts=args.num_posts,
            skip_dirs=skip_dirs,
            skip_paths=skip_paths,
        )
        _insert_inbox_downloads(downloaded, output_folder)
    elif args.source == "all":
        cfg = _load_config(Path(args.config))
        client_id = (args.imgur_client_id or "").strip()
        for board in cfg["4chan"]["boards"]:
            if board:
                downloaded = _run_4chan(
                    board=board,
                    inbox_dir=inbox_dir,
                    index_pages=args.index_pages,
                    skip_dirs=skip_dirs,
                    skip_paths=skip_paths,
                )
                _insert_inbox_downloads(downloaded, output_folder)
        for topic in cfg["imgur"]["topics"]:
            if topic:
                downloaded = _run_imgur(
                    topic=topic,
                    inbox_dir=inbox_dir,
                    max_items=args.max_items,
                    client_id=client_id,
                    skip_dirs=skip_dirs,
                    skip_paths=skip_paths,
                )
                _insert_inbox_downloads(downloaded, output_folder)
        for blog in cfg["tumblr"]["blogs"]:
            if blog:
                downloaded = _run_tumblr(
                    blog=blog,
                    inbox_dir=inbox_dir,
                    num_posts=args.num_posts,
                    skip_dirs=skip_dirs,
                    skip_paths=skip_paths,
                )
                _insert_inbox_downloads(downloaded, output_folder)
        p_cfg = cfg.get("pixelfed") or {}
        base = p_cfg.get("instance_base", "").strip()
        if base:
            limit = p_cfg.get("limit", 40)
            token = p_cfg.get("access_token", "").strip() or None
            downloaded = _run_pixelfed(
                instance_base=base,
                limit=limit,
                inbox_dir=inbox_dir,
                skip_dirs=skip_dirs,
                skip_paths=skip_paths,
                access_token=token,
            )
            _insert_inbox_downloads(downloaded, output_folder)
    elif args.source == "pixelfed":
        cfg = _load_config(Path(args.config))
        p_cfg = cfg.get("pixelfed") or {}
        base = p_cfg.get("instance_base", "").strip()
        limit = p_cfg.get("limit", 40)
        if not base:
            logger.error(
                "Pixelfed requires [pixelfed] instance_base in config (e.g. https://pixelfed.social)."
            )
            raise SystemExit(1)
        token = p_cfg.get("access_token", "").strip() or None
        downloaded = _run_pixelfed(
            instance_base=base,
            limit=limit,
            inbox_dir=inbox_dir,
            skip_dirs=skip_dirs,
            skip_paths=skip_paths,
            access_token=token,
        )
        _insert_inbox_downloads(downloaded, output_folder)
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
                db.record_judged_image(dest_path, location, data_root=output_folder)

            if args.post_summary and moved:
                unposted = db.count_unposted_corpus_images(output_folder)
                _post_run_summary(
                    config_path=Path(args.config),
                    corpus_count=sum(1 for _, loc in moved if loc == "corpus"),
                    void_count=sum(1 for _, loc in moved if loc == "void"),
                    unposted_count=unposted,
                )


if __name__ == "__main__":
    main()
