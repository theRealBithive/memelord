"""Orchestrates retina/ scrapers → Django Image inbox."""

import hashlib
import tomllib
from pathlib import Path

from loguru import logger

from ratings.models import Image, Source
from retina import fourchan, imgur, pixelfed, tumblr

# Maps Source.type → (toml_section, toml_key, result_key) for list-based sources.
# Pixelfed is kept separate (single URL, not a list).
_SOURCE_MAP = [
    (Source.FOURCHAN, "4chan",  "boards", "boards"),
    (Source.IMGUR,    "imgur",  "topics", "topics"),
    (Source.TUMBLR,   "tumblr", "blogs",  "blogs"),
]


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.warning("Config file {} not found.", config_path)
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _load_sources(config_path: Path) -> dict:
    """Return scrape sources from the DB, falling back to config.toml if none are configured."""
    db_sources = list(Source.objects.filter(enabled=True))
    if db_sources:
        result = {rk: [s.name for s in db_sources if s.type == stype]
                  for stype, _, _, rk in _SOURCE_MAP}
        result["pixelfed"] = next((s.name for s in db_sources if s.type == Source.PIXELFED), "")
        return result

    logger.info("No sources in DB — falling back to config.toml")
    cfg = _load_config(config_path)
    result = {rk: cfg.get(section, {}).get(key, [])
              for _, section, key, rk in _SOURCE_MAP}
    result["pixelfed"] = cfg.get("pixelfed", {}).get("instance_base", "").strip()
    return result


def import_from_config(config_path: Path) -> int:
    """Read config.toml and create Source records for any not already in the DB. Returns count created."""
    cfg = _load_config(config_path)
    created = 0
    for stype, section, key, _ in _SOURCE_MAP:
        for name in cfg.get(section, {}).get(key, []):
            if name and Source.objects.get_or_create(type=stype, name=name)[1]:
                created += 1
    pf = cfg.get("pixelfed", {}).get("instance_base", "").strip()
    if pf and Source.objects.get_or_create(type=Source.PIXELFED, name=pf)[1]:
        created += 1
    return created


def _insert(path: Path, source_url: str, source_label: str, data_dir: Path, existing: set) -> bool:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return False
    h = hashlib.sha256(raw).hexdigest()
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
    """Scrape all enabled sources into data_dir/inbox/. Returns per-source new-image counts."""
    sources = _load_sources(config_path)
    inbox_dir = data_dir / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = [data_dir / d for d in ("corpus", "void", "inbox") if (data_dir / d).exists()]
    existing: set[str] = set(Image.objects.values_list("content_hash", flat=True))
    counts: dict[str, int] = {}

    for board in sources["boards"]:
        logger.info("Scraping 4chan /{}/", board)
        urls = fourchan.iter_image_urls(board)
        downloaded = fourchan.download_images(urls, inbox_dir, board, skip_dirs=skip_dirs)
        counts[f"4chan/{board}"] = sum(_insert(p, u, l, data_dir, existing) for p, u, l in downloaded)

    for topic in sources["topics"]:
        logger.info("Scraping Imgur: {}", topic)
        urls = imgur.iter_image_urls(topic)
        downloaded = imgur.download_images(urls, inbox_dir, topic, skip_dirs=skip_dirs)
        counts[f"imgur/{topic}"] = sum(_insert(p, u, l, data_dir, existing) for p, u, l in downloaded)

    for blog in sources["blogs"]:
        logger.info("Scraping Tumblr: {}", blog)
        urls = tumblr.iter_image_urls(blog)
        downloaded = tumblr.download_images(urls, inbox_dir, blog, skip_dirs=skip_dirs)
        counts[f"tumblr/{blog}"] = sum(_insert(p, u, l, data_dir, existing) for p, u, l in downloaded)

    if sources["pixelfed"]:
        logger.info("Scraping Pixelfed: {}", sources["pixelfed"])
        items = pixelfed.iter_image_items(sources["pixelfed"])
        downloaded = pixelfed.download_images(items, inbox_dir, skip_dirs=skip_dirs)
        counts["pixelfed"] = sum(_insert(p, u, l, data_dir, existing) for p, u, l in downloaded)

    return counts
