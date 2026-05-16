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


def _insert(path: Path, source_url: str, source_label: str, data_dir: Path, existing: set, is_nsfw: bool = False) -> bool:
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
        is_nsfw=is_nsfw,
    )
    existing.add(h)
    return True


def classify_inbox(data_dir: Path, weights_path: Path) -> None:
    """Run the trained classifier on all inbox images; move high-confidence ones to corpus/void."""
    from core import brain
    from ratings.utils import move_image

    if not weights_path.exists():
        return

    images = list(Image.objects.filter(location=Image.INBOX, file_deleted=False))
    if not images:
        return

    logger.info("Auto-classifying {} inbox images.", len(images))
    classifier = brain.load_classifier(weights_path)
    encoder = brain.get_encoder()
    transform = brain.get_transform()

    paths = [data_dir / img.file_path for img in images]
    embeddings = brain.encode(encoder, paths, transform=transform)

    to_corpus = to_void = 0
    for img, emb in zip(images, embeddings):
        prob = float(brain.predict_proba(classifier, emb))
        if prob >= 0.75:
            move_image(img, Image.CORPUS, data_dir)
            img.save(update_fields=["file_path", "location"])
            to_corpus += 1
        elif prob <= 0.25:
            move_image(img, Image.VOID, data_dir)
            img.save(update_fields=["file_path", "location"])
            to_void += 1

    remaining = len(images) - to_corpus - to_void
    logger.info("Classified: {} → corpus, {} → void, {} remain in inbox.", to_corpus, to_void, remaining)


def run(config_path: Path, data_dir: Path, weights_path: Path | None = None) -> dict[str, int]:
    """Scrape all enabled sources into data_dir/inbox/. Returns per-source new-image counts."""
    sources = _load_sources(config_path)
    inbox_dir = data_dir / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = [data_dir / d for d in ("corpus", "void", "inbox") if (data_dir / d).exists()]
    existing: set[str] = set(Image.objects.values_list("content_hash", flat=True))
    nsfw_names: set[str] = set(Source.objects.filter(is_nsfw=True).values_list("name", flat=True))
    counts: dict[str, int] = {}

    for board in sources["boards"]:
        logger.info("Scraping 4chan /{}/", board)
        urls = fourchan.iter_image_urls(board)
        downloaded = fourchan.download_images(urls, inbox_dir, board, skip_dirs=skip_dirs)
        counts[f"4chan/{board}"] = sum(_insert(p, u, l, data_dir, existing, board in nsfw_names) for p, u, l in downloaded)

    for topic in sources["topics"]:
        logger.info("Scraping Imgur: {}", topic)
        urls = imgur.iter_image_urls(topic)
        downloaded = imgur.download_images(urls, inbox_dir, topic, skip_dirs=skip_dirs)
        counts[f"imgur/{topic}"] = sum(_insert(p, u, l, data_dir, existing, topic in nsfw_names) for p, u, l in downloaded)

    for blog in sources["blogs"]:
        logger.info("Scraping Tumblr: {}", blog)
        urls = tumblr.iter_image_urls(blog)
        downloaded = tumblr.download_images(urls, inbox_dir, blog, skip_dirs=skip_dirs)
        counts[f"tumblr/{blog}"] = sum(_insert(p, u, l, data_dir, existing, blog in nsfw_names) for p, u, l in downloaded)

    if sources["pixelfed"]:
        logger.info("Scraping Pixelfed: {}", sources["pixelfed"])
        items = pixelfed.iter_image_items(sources["pixelfed"])
        downloaded = pixelfed.download_images(items, inbox_dir, skip_dirs=skip_dirs)
        pf_name = sources["pixelfed"]
        counts["pixelfed"] = sum(_insert(p, u, l, data_dir, existing, pf_name in nsfw_names) for p, u, l in downloaded)

    if weights_path:
        classify_inbox(data_dir, weights_path)

    return counts
