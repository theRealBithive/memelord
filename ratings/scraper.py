"""Orchestrates retina/ scrapers → Django Image inbox."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from core import brain, dedup, nsfw
from ratings.models import Image, Source
from retina import fourchan, imgur, mastodon as mastodon_scraper, pixelfed, tumblr

# Maps Source.type → (toml_section, toml_key, result_key) for list-based sources.
# Pixelfed is kept separate (single URL, not a list).
_SOURCE_MAP = [
    (Source.FOURCHAN, "4chan", "boards", "boards"),
    (Source.IMGUR, "imgur", "topics", "topics"),
    (Source.TUMBLR, "tumblr", "blogs", "blogs"),
]

# Scrapers that share the same iter_image_urls / download_images(urls, dir, name, skip_dirs)
# interface. Pixelfed and Mastodon have different signatures so they stay explicit below.
_SIMPLE_SCRAPERS = [
    (fourchan, "boards", "4chan/{}"),
    (imgur, "topics", "imgur/{}"),
    (tumblr, "blogs", "tumblr/{}"),
]


@dataclass
class VisionConfig:
    """Thresholds and model paths for dedup / classification."""

    phash_max_distance: int = 5
    dino_dedup_threshold: float = 0.92
    nsfw_threshold: float = 0.30
    weights_path: Path | None = None
    nsfw_weights_path: Path | None = None


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
        result = {
            rk: [s.name for s in db_sources if s.type == stype]
            for stype, _, _, rk in _SOURCE_MAP
        }
        result["pixelfed"] = next(
            (s.name for s in db_sources if s.type == Source.PIXELFED), ""
        )
        result["mastodon_accounts"] = [
            s.name for s in db_sources if s.type == Source.MASTODON
        ]
        return result

    logger.info("No sources in DB — falling back to config.toml")
    cfg = _load_config(config_path)
    result = {
        rk: cfg.get(section, {}).get(key, []) for _, section, key, rk in _SOURCE_MAP
    }
    result["pixelfed"] = cfg.get("pixelfed", {}).get("instance_base", "").strip()
    result["mastodon_accounts"] = cfg.get("mastodon", {}).get("accounts", [])
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
    for acct in cfg.get("mastodon", {}).get("accounts", []):
        acct = acct.strip()
        if acct and Source.objects.get_or_create(type=Source.MASTODON, name=acct)[1]:
            created += 1
    return created


def _sha_filter(
    downloaded: list[tuple[Path, str, str]],
    index: dedup.DedupIndex,
) -> list[tuple[Path, str, str, str]]:
    """Apply SHA-256 dedup; return survivors as (path, url, label, content_hash)."""
    survivors: list[tuple[Path, str, str, str]] = []
    for path, source_url, source_label in downloaded:
        try:
            raw = path.read_bytes()
        except (FileNotFoundError, OSError):
            continue
        h = hashlib.sha256(raw).hexdigest()
        if dedup.is_sha_duplicate(h, index):
            path.unlink(missing_ok=True)
            continue
        survivors.append((path, source_url, source_label, h))
    return survivors


def _phash_filter(
    survivors: list[tuple[Path, str, str, str]],
    index: dedup.DedupIndex,
    max_distance: int,
) -> list[tuple[Path, str, str, str, str]]:
    """Apply pHash dedup; return survivors as (path, url, label, hash, phash)."""
    out: list[tuple[Path, str, str, str, str]] = []
    for path, source_url, source_label, h in survivors:
        try:
            dup, ph = dedup.is_phash_duplicate_for_path(path, index, max_distance)
        except OSError:
            continue
        if dup:
            path.unlink(missing_ok=True)
            continue
        out.append((path, source_url, source_label, h, ph))
    return out


def _process_candidates(
    candidates: list[tuple[Path, str, str, str, str]],
    data_dir: Path,
    index: dedup.DedupIndex,
    encoder,
    transform,
    vision: VisionConfig,
    nsfw_clf,
) -> int:
    """DINO dedup, NSFW tag, and DB insert for phash-filtered candidates."""
    if not candidates:
        return 0

    paths = [c[0] for c in candidates]
    embeddings, valid_paths = brain.encode(
        encoder, paths, transform=transform, progress_label="scrape"
    )
    path_to_emb = dict(zip(valid_paths, embeddings))
    inserted = 0

    for path, source_url, source_label, h, ph in candidates:
        emb = path_to_emb.get(path)
        if emb is None:
            continue
        if dedup.is_embedding_duplicate(emb, index, vision.dino_dedup_threshold):
            path.unlink(missing_ok=True)
            continue

        is_nsfw = False
        if nsfw_clf is not None:
            is_nsfw = nsfw.predict_nsfw(nsfw_clf, emb, vision.nsfw_threshold)

        Image.objects.create(
            content_hash=h,
            file_path=str(path.relative_to(data_dir)),
            source_url=source_url or None,
            source_label=source_label,
            location=Image.INBOX,
            is_nsfw=is_nsfw,
            phash=ph,
            embedding=brain.embedding_to_bytes(emb),
        )
        index.add(h, ph, emb)
        inserted += 1

    return inserted


def _process_downloads(
    downloaded: list[tuple[Path, str, str]],
    data_dir: Path,
    index: dedup.DedupIndex,
    encoder,
    transform,
    vision: VisionConfig,
    nsfw_clf,
) -> int:
    """Full dedup pipeline for one source batch."""
    sha_ok = _sha_filter(downloaded, index)
    phash_ok = _phash_filter(sha_ok, index, vision.phash_max_distance)
    if not phash_ok:
        return 0
    return _process_candidates(
        phash_ok, data_dir, index, encoder, transform, vision, nsfw_clf
    )


def classify_inbox(
    data_dir: Path,
    vision: VisionConfig,
    *,
    encoder=None,
    transform=None,
    nsfw_clf=None,
) -> None:
    """Auto-sort inbox with taste classifier; tag NSFW; backfill phash/embedding."""
    from ratings.utils import move_image

    need_vision = vision.weights_path and vision.weights_path.exists()
    need_nsfw = bool(vision.nsfw_weights_path and vision.nsfw_weights_path.exists())
    if not need_vision and not need_nsfw and nsfw_clf is None:
        return

    images = list(Image.objects.filter(location=Image.INBOX, file_deleted=False))
    if not images:
        return

    logger.info("Auto-classifying {} inbox images.", len(images))
    if encoder is None:
        encoder = brain.get_encoder()
    if transform is None:
        transform = brain.get_transform()
    paths = [data_dir / img.file_path for img in images]
    embeddings, valid_paths = brain.encode(
        encoder, paths, transform=transform, progress_label="classify_inbox"
    )
    path_to_emb = dict(zip(valid_paths, embeddings))

    taste_clf = None
    if need_vision:
        taste_clf = brain.load_classifier(vision.weights_path)

    if nsfw_clf is None and need_nsfw:
        nsfw_clf = brain.load_classifier(vision.nsfw_weights_path)

    to_corpus = to_void = nsfw_tagged = 0
    for img, path in zip(images, paths):
        emb = path_to_emb.get(path)
        if emb is None:
            continue
        update_fields: list[str] = []
        if not img.phash:
            from core import phash as phash_mod

            img.phash = phash_mod.compute_phash(path)
            update_fields.append("phash")
        if img.embedding is None:
            img.embedding = brain.embedding_to_bytes(emb)
            update_fields.append("embedding")

        if nsfw_clf is not None and not img.is_nsfw:
            predicted = nsfw.predict_nsfw(nsfw_clf, emb, vision.nsfw_threshold)
            if predicted:
                img.is_nsfw = True
                update_fields.append("is_nsfw")
                nsfw_tagged += 1

        if taste_clf is not None:
            prob = float(brain.predict_proba(taste_clf, emb))
            if prob >= 0.75:
                move_image(img, Image.CORPUS, data_dir)
                update_fields.extend(["file_path", "location"])
                to_corpus += 1
            elif prob <= 0.25:
                move_image(img, Image.VOID, data_dir)
                update_fields.extend(["file_path", "location"])
                to_void += 1

        if update_fields:
            img.save(update_fields=list(dict.fromkeys(update_fields)))

    remaining = len(images) - to_corpus - to_void
    logger.info(
        "Classified: {} → corpus, {} → void, {} NSFW-tagged, {} remain in inbox.",
        to_corpus,
        to_void,
        nsfw_tagged,
        remaining,
    )


def run(
    config_path: Path,
    data_dir: Path,
    vision: VisionConfig | None = None,
) -> dict[str, int]:
    """Scrape all enabled sources into data_dir/inbox/. Returns per-source new-image counts."""
    if vision is None:
        vision = VisionConfig()

    cfg = _load_config(config_path)
    sources = _load_sources(config_path)
    inbox_dir = data_dir / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = [
        data_dir / d for d in ("corpus", "void", "inbox") if (data_dir / d).exists()
    ]
    index = dedup.DedupIndex.from_db()

    encoder = brain.get_encoder()
    transform = brain.get_transform()
    nsfw_clf = None
    if vision.nsfw_weights_path and vision.nsfw_weights_path.exists():
        nsfw_clf = brain.load_classifier(vision.nsfw_weights_path)

    need_classify = (
        vision.weights_path and vision.weights_path.exists()
    ) or nsfw_clf is not None

    counts: dict[str, int] = {}

    for module, sources_key, label_fmt in _SIMPLE_SCRAPERS:
        for name in sources[sources_key]:
            label = label_fmt.format(name)
            logger.info("Scraping {}", label)
            urls = module.iter_image_urls(name)
            downloaded = module.download_images(urls, inbox_dir, name, skip_dirs=skip_dirs)
            counts[label] = _process_downloads(
                downloaded, data_dir, index, encoder, transform, vision, nsfw_clf
            )

    if sources["pixelfed"]:
        logger.info("Scraping Pixelfed: {}", sources["pixelfed"])
        items = pixelfed.iter_image_items(sources["pixelfed"])
        downloaded = pixelfed.download_images(items, inbox_dir, skip_dirs=skip_dirs)
        counts["pixelfed"] = _process_downloads(
            downloaded, data_dir, index, encoder, transform, vision, nsfw_clf
        )

    mastodon_token = cfg.get("mastodon", {}).get("access_token", "").strip() or None
    for account in sources["mastodon_accounts"]:
        logger.info("Scraping Mastodon: {}", account)
        source_obj = Source.objects.filter(type=Source.MASTODON, name=account).first()
        since_id = source_obj.cursor if source_obj else None
        items, new_cursor = mastodon_scraper.iter_image_items(
            account, since_id=since_id, access_token=mastodon_token
        )
        downloaded = mastodon_scraper.download_images(
            items, inbox_dir, account, skip_dirs=skip_dirs
        )
        counts[f"mastodon/{account}"] = _process_downloads(
            downloaded, data_dir, index, encoder, transform, vision, nsfw_clf
        )
        if new_cursor and source_obj:
            source_obj.cursor = new_cursor
            source_obj.save(update_fields=["cursor"])

    if need_classify:
        classify_inbox(data_dir, vision, encoder=encoder, transform=transform, nsfw_clf=nsfw_clf)

    return counts


def vision_config_from_settings() -> VisionConfig:
    """Build VisionConfig from Django settings."""
    from django.conf import settings as dj_settings

    return VisionConfig(
        phash_max_distance=dj_settings.PHASH_MAX_DISTANCE,
        dino_dedup_threshold=dj_settings.DINO_DEDUP_COSINE_THRESHOLD,
        nsfw_threshold=dj_settings.NSFW_THRESHOLD,
        weights_path=Path(dj_settings.WEIGHTS_PATH),
        nsfw_weights_path=Path(dj_settings.NSFW_WEIGHTS_PATH),
    )
