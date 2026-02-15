"""Janulon CLI: scrape sources and run the aesthetic pipeline (download + judge)."""

import argparse
import hashlib
import os
import tomllib
from pathlib import Path

from loguru import logger
from retina import fourchan, image_validation, imgur, tumblr

from core import brain, db

_INBOX_DIR = "inbox"
_CONFIG_DEFAULT = "config.toml"
_CORPUS_DIR = "corpus"
_VOID_DIR = "void"


def _load_config(config_path: Path) -> dict:
    """
    Load config.toml; return dict with "4chan", "tumblr", "imgur" keys.
    Each value is a dict with "boards"/"blogs"/"topics" list of strings.
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
    return out


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


def _insert_judged_image(dest_path: Path, location: str) -> None:
    """Insert a judged image (moved to corpus/void) into the database."""
    try:
        raw = dest_path.read_bytes()
    except OSError:
        return
    content_hash = hashlib.sha256(raw).hexdigest()
    if db.Image.get_or_none(db.Image.content_hash == content_hash) is not None:
        return
    db.Image.create(
        content_hash=content_hash,
        file_path=str(dest_path.resolve()),
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

    args = parser.parse_args()

    if args.command == "import-data":
        inserted, skipped = db.import_data(args.data_dir, args.db)
        logger.info(
            "Import done: {} inserted, {} skipped (duplicate content hash).",
            inserted,
            skipped,
        )
        return

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    inbox_dir = output_folder / _INBOX_DIR
    inbox_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = _skip_dirs(args.data_dir, output_folder)

    db.init_db(args.db)
    skip_paths = {
        str(Path(r.file_path).resolve())
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
                _insert_judged_image(dest_path, location)


if __name__ == "__main__":
    main()
