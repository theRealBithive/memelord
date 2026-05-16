"""Orchestrates retina/ scrapers → Django Image inbox."""

import hashlib
import tomllib
from pathlib import Path

from loguru import logger

from ratings.models import Image
from retina import fourchan, imgur, pixelfed, tumblr


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.warning("Config file {} not found, using empty config.", config_path)
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _insert(path: Path, source_url: str, source_label: str, data_dir: Path, existing: set) -> bool:
    """Hash file, skip if duplicate, otherwise insert into DB. Returns True if inserted."""
    if not path.exists():
        return False
    h = _content_hash(path)
    if h in existing:
        path.unlink(missing_ok=True)
        return False
    Image.objects.create(
        content_hash=h,
        file_path=str(path.relative_to(data_dir)),
        source_url=source_url or "",
        source_label=source_label,
        location=Image.INBOX,
    )
    existing.add(h)
    return True


def run(config_path: Path, data_dir: Path) -> dict[str, int]:
    """
    Scrape all configured sources into data_dir/inbox/.
    Returns per-source counts of newly inserted images.
    """
    cfg = _load_config(config_path)
    inbox_dir = data_dir / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)

    skip_dirs = [data_dir / d for d in ("corpus", "void", "inbox") if (data_dir / d).exists()]
    existing: set[str] = set(Image.objects.values_list("content_hash", flat=True))

    counts: dict[str, int] = {}

    # 4chan
    for board in cfg.get("4chan", {}).get("boards", []):
        logger.info("Scraping 4chan /{}/", board)
        urls = fourchan.iter_image_urls(board)
        downloaded = fourchan.download_images(urls, inbox_dir, board, skip_dirs=skip_dirs)
        n = sum(_insert(p, url, label, data_dir, existing) for p, url, label in downloaded)
        counts[f"4chan/{board}"] = n
        logger.info("4chan /{}: {} new images", board, n)

    # Imgur
    for topic in cfg.get("imgur", {}).get("topics", []):
        logger.info("Scraping Imgur topic: {}", topic)
        urls = imgur.iter_image_urls(topic)
        downloaded = imgur.download_images(urls, inbox_dir, topic, skip_dirs=skip_dirs)
        n = sum(_insert(p, url, label, data_dir, existing) for p, url, label in downloaded)
        counts[f"imgur/{topic}"] = n
        logger.info("Imgur {}: {} new images", topic, n)

    # Tumblr
    for blog in cfg.get("tumblr", {}).get("blogs", []):
        logger.info("Scraping Tumblr blog: {}", blog)
        urls = tumblr.iter_image_urls(blog)
        downloaded = tumblr.download_images(urls, inbox_dir, blog, skip_dirs=skip_dirs)
        n = sum(_insert(p, url, label, data_dir, existing) for p, url, label in downloaded)
        counts[f"tumblr/{blog}"] = n
        logger.info("Tumblr {}: {} new images", blog, n)

    # Pixelfed
    instance_base = cfg.get("pixelfed", {}).get("instance_base", "").strip()
    if instance_base:
        logger.info("Scraping Pixelfed: {}", instance_base)
        items = pixelfed.iter_image_items(instance_base)
        downloaded = pixelfed.download_images(items, inbox_dir, skip_dirs=skip_dirs)
        n = sum(_insert(p, url, label, data_dir, existing) for p, url, label in downloaded)
        counts["pixelfed"] = n
        logger.info("Pixelfed: {} new images", n)

    return counts
