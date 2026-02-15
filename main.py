"""Janulon CLI: scrape sources and run the aesthetic pipeline (download + judge)."""

import argparse
import os
import tomllib
from pathlib import Path

from loguru import logger
from retina import fourchan, image_validation, imgur, tumblr

from core import brain

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
        urls, inbox_dir, board, skip_dirs=skip_dirs or None
    )
    logger.success("Done. Downloaded {} images to {}", len(paths), inbox_dir.resolve())


def _run_tumblr(
    blog: str,
    inbox_dir: Path,
    num_posts: int,
    skip_dirs: list[Path],
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
    paths = tumblr.download_images(urls, inbox_dir, blog, skip_dirs=skip_dirs or None)
    logger.success("Done. Downloaded {} images to {}", len(paths), inbox_dir.resolve())


def _run_imgur(
    topic: str,
    inbox_dir: Path,
    max_items: int,
    client_id: str,
    skip_dirs: list[Path],
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
    paths = imgur.download_images(urls, inbox_dir, topic, skip_dirs=skip_dirs or None)
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


def _judge_and_sort(
    output_folder: Path,
    weights_path: Path,
    threshold: float,
) -> None:
    """
    Encode inbox images, move to corpus (>= threshold) or void (< threshold).
    Leaves data_dir untouched.
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
        return

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

    moved_corpus = 0
    moved_void = 0
    for path, proba in zip(paths, probas):
        name = path.name
        if proba >= threshold:
            dest = _unique_dest(corpus_dir, name)
            path.rename(dest)
            moved_corpus += 1
        else:
            dest = _unique_dest(void_dir, name)
            path.rename(dest)
            moved_void += 1
    logger.success(
        "Judged {} images: {} → corpus, {} → void",
        len(paths),
        moved_corpus,
        moved_void,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Janulon: scrape image sources and filter by taste.",
    )
    parser.add_argument(
        "--source",
        choices=["4chan", "all", "imgur", "reddit", "tumblr"],
        required=True,
        help="Source. 'all' = load config and run all listed boards/blogs/topics.",
    )
    parser.add_argument(
        "--board",
        default="wg",
        help="4chan board. Default: wg (wallpaper general).",
    )
    parser.add_argument(
        "--blog",
        help="Tumblr blog (tumblr only). Example: staff or blogname.tumblr.com",
    )
    parser.add_argument(
        "--topic",
        help="Imgur topic (imgur only). Example: funny (imgur.com/t/funny)",
    )
    parser.add_argument(
        "--imgur_client_id",
        default=os.environ.get("IMGUR_CLIENT_ID", ""),
        help="Imgur API Client ID (imgur only). Default: env IMGUR_CLIENT_ID",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=120,
        metavar="N",
        help="Max image items (imgur only). Default: 120",
    )
    parser.add_argument(
        "--subreddit",
        help="Subreddit to scrape (reddit only).",
    )
    parser.add_argument(
        "--num_posts",
        type=int,
        default=50,
        metavar="N",
        help="Max posts to fetch (tumblr only). Default: 50",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("output"),
        help="Folder for inbox/ and judged corpus/void. Default: output",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Skip download if image in data/corpus or data/void. Default: data",
    )
    parser.add_argument(
        "--index_pages",
        type=int,
        default=2,
        metavar="N",
        help="Index pages to scrape (4chan only). Default: 2",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Score >= threshold → corpus, else void. Default: 0.85",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("Janulon_weights.pkl"),
        help="Path to trained classifier. Default: Janulon_weights.pkl",
    )
    parser.add_argument(
        "--no_judge",
        action="store_true",
        help="Only download; skip encoder/classifier and corpus/void move.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(_CONFIG_DEFAULT),
        help="Config file for --source all (4chan boards, tumblr blogs, imgur topics).",
    )
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    inbox_dir = output_folder / _INBOX_DIR
    inbox_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = _skip_dirs(args.data_dir, output_folder)

    if args.source == "4chan":
        _run_4chan(
            board=args.board,
            inbox_dir=inbox_dir,
            index_pages=args.index_pages,
            skip_dirs=skip_dirs,
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
                )
        for topic in cfg["imgur"]["topics"]:
            if topic:
                _run_imgur(
                    topic=topic,
                    inbox_dir=inbox_dir,
                    max_items=args.max_items,
                    client_id=client_id,
                    skip_dirs=skip_dirs,
                )
        for blog in cfg["tumblr"]["blogs"]:
            if blog:
                _run_tumblr(
                    blog=blog,
                    inbox_dir=inbox_dir,
                    num_posts=args.num_posts,
                    skip_dirs=skip_dirs,
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
            _judge_and_sort(
                output_folder=output_folder,
                weights_path=args.weights,
                threshold=args.threshold,
            )


if __name__ == "__main__":
    main()
