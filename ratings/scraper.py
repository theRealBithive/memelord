"""Orchestrates retina/ scrapers → Django Image store."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from django.db import IntegrityError
from loguru import logger

from core import brain, dedup, nsfw, taste
from ratings import features
from ratings.embeddings import has_current_embedding, stale_images
from ratings.models import Image, Source
from retina import flickr, fourchan, imgur, pixelfed, tumblr
from retina import mastodon as mastodon_scraper

# How many stale library rows classify_images re-encodes inline per run (taste
# contract V20). New downloads are encoded by the dedup step anyway; this cap
# is for rows left over by an encoder change, where the whole library is stale
# at once and the inline pass would run for hours inside the scrape. The rest
# belongs to the re-encode chain (ratings/embeddings.py).
CLASSIFY_BACKFILL_LIMIT = 200

# Maps Source.type → (toml_section, toml_key, result_key) for list-based sources.
_SOURCE_MAP = [
    (Source.FOURCHAN, "4chan", "boards", "boards"),
    (Source.IMGUR, "imgur", "topics", "topics"),
    (Source.TUMBLR, "tumblr", "blogs", "blogs"),
]

# Scrapers that share the same iter_image_urls / download_images(urls, dir, name, skip_dirs)
# interface. 4chan, Pixelfed and Mastodon are cursor-based (incremental) and have
# different signatures, so they stay explicit below.
_SIMPLE_SCRAPERS = [
    (imgur, "topics", "imgur/{}"),
    (tumblr, "blogs", "tumblr/{}"),
]

# Account scrapers over the Mastodon-compatible API: (module, Source.type,
# config section = label prefix = sources-key prefix, service name for the log).
_ACCOUNT_SCRAPERS = [
    (pixelfed, Source.PIXELFED, "pixelfed", "Pixelfed"),
    (mastodon_scraper, Source.MASTODON, "mastodon", "Mastodon"),
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


def _cursor_int(cursor: str | None) -> int | None:
    """Parse a Source.cursor (stored as text) into a 4chan last_modified stamp.

    Returns None on a missing or non-numeric cursor so the board falls back to a
    full scrape rather than crashing on a value written by some other source type.
    """
    if not cursor:
        return None
    try:
        return int(cursor)
    except (TypeError, ValueError):
        return None


def _warn_legacy_pixelfed(cfg: dict) -> None:
    """Warn when a config.toml still uses the removed [pixelfed] instance_base key.

    2.0 replaced the single instance URL with a list of @user@instance account
    handles (``accounts = [...]``). The old key is silently ignored otherwise,
    so a user upgrading would lose Pixelfed scraping with no signal at all.
    """
    pf = cfg.get("pixelfed", {})
    if pf.get("instance_base") and not pf.get("accounts"):
        logger.warning(
            "[pixelfed] instance_base is no longer supported — use "
            'accounts = ["@user@instance"]. Pixelfed scraping stays disabled '
            "until the config is migrated."
        )


def _load_sources(config_path: Path) -> dict:
    """Return scrape sources from the DB, falling back to config.toml if none are configured."""
    db_sources = list(Source.objects.filter(enabled=True))
    if db_sources:
        result = {
            rk: [s.name for s in db_sources if s.type == stype]
            for stype, _, _, rk in _SOURCE_MAP
        }
        result["pixelfed_accounts"] = [
            s.name for s in db_sources if s.type == Source.PIXELFED
        ]
        result["mastodon_accounts"] = [
            s.name for s in db_sources if s.type == Source.MASTODON
        ]
        result["flickr_sources"] = [
            s.name for s in db_sources if s.type == Source.FLICKR
        ]
        return result

    logger.info("No sources in DB — falling back to config.toml")
    cfg = _load_config(config_path)
    _warn_legacy_pixelfed(cfg)
    result = {
        rk: cfg.get(section, {}).get(key, []) for _, section, key, rk in _SOURCE_MAP
    }
    result["pixelfed_accounts"] = [
        a.strip() for a in cfg.get("pixelfed", {}).get("accounts", []) if a.strip()
    ]
    result["mastodon_accounts"] = cfg.get("mastodon", {}).get("accounts", [])
    result["flickr_sources"] = _flickr_sources_from_config(cfg)
    return result


def _flickr_sources_from_config(cfg: dict) -> list[str]:
    """
    The [flickr] sources of config.toml in canonical form.

    config.toml bypasses the add-source view, so its entries go through the
    same normalize_source as typed ones; an invalid entry is logged and dropped.
    """
    names = []
    for entry in cfg.get("flickr", {}).get("sources", []):
        canonical = flickr.normalize_source(str(entry))
        if canonical is None:
            logger.warning("config.toml: {!r} is not a Flickr group or user, skipped", entry)
            continue
        names.append(canonical)
    return names


def import_from_config(config_path: Path) -> int:
    """Read config.toml and create Source records for any not already in the DB. Returns count created."""
    cfg = _load_config(config_path)
    _warn_legacy_pixelfed(cfg)
    created = 0
    for stype, section, key, _ in _SOURCE_MAP:
        for name in cfg.get(section, {}).get(key, []):
            if name and Source.objects.get_or_create(type=stype, name=name)[1]:
                created += 1
    for acct in cfg.get("pixelfed", {}).get("accounts", []):
        acct = acct.strip()
        if acct and Source.objects.get_or_create(type=Source.PIXELFED, name=acct)[1]:
            created += 1
    for acct in cfg.get("mastodon", {}).get("accounts", []):
        acct = acct.strip()
        if acct and Source.objects.get_or_create(type=Source.MASTODON, name=acct)[1]:
            created += 1
    for name in _flickr_sources_from_config(cfg):
        if Source.objects.get_or_create(type=Source.FLICKR, name=name)[1]:
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
    path_to_emb = dict(zip(valid_paths, embeddings, strict=True))
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

        try:
            Image.objects.create(
                content_hash=h,
                file_path=str(path.relative_to(data_dir)),
                source_url=source_url or None,
                source_label=source_label,
                is_nsfw=is_nsfw,
                phash=ph,
                embedding=brain.embedding_to_bytes(emb),
                embedding_model=brain.ENCODER_ID,
            )
        except IntegrityError:
            # A concurrent scrape (gunicorn manual trigger vs. qcluster auto)
            # can insert the same content_hash between our from_db() snapshot
            # and this create. Django runs in autocommit so the failed INSERT
            # auto-rolls-back at the DB level — no atomic() needed. Do NOT
            # unlink path here: filenames are deterministic per URL, so both
            # threads wrote to the same path; the winning record owns the file.
            logger.warning(
                "Duplicate content_hash {} inserted concurrently; skipping.", h[:12]
            )
            continue
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


def classify_images(
    data_dir: Path,
    vision: VisionConfig,
    *,
    encoder=None,
    transform=None,
    nsfw_clf=None,
) -> dict:
    """
    Run taste classifier + NSFW tagger on all unscored images; backfill phash/embedding.

    Reuses each image's stored embedding when available — re-encoding from disk
    is expensive and partial progress is durable because we save per-image.
    predicted_score IS NULL is the "show anyway" signal for fresh images that
    have never been through the classifier; once set it drives the visibility dial.

    The NSFW head only touches rows nobody has decided (NSFW contract N3), in
    both directions: an undecided flag is the head's own earlier guess, so a
    newer head may revise it, while a flag a person set or confirmed stays.
    Returns the counts the "Classify now" report shows (N5).
    """
    counts = {"processed": 0, "nsfw_tagged": 0, "nsfw_untagged": 0}
    need_vision = vision.weights_path and vision.weights_path.exists()
    need_nsfw = bool(vision.nsfw_weights_path and vision.nsfw_weights_path.exists())
    if not need_vision and not need_nsfw and nsfw_clf is None:
        return counts

    images = list(
        Image.objects.filter(is_purged=False, score__isnull=True).order_by("downloaded_at")
    )
    if not images:
        return counts

    logger.info("Auto-classifying {} images.", len(images))

    # Only encode images whose vector is missing or from an older encoder, or
    # that lack a phash — the common case is that everything was encoded at
    # scrape time with the current encoder and we can read from the DB (V3).
    # Oldest first, at most CLASSIFY_BACKFILL_LIMIT per run; the rows beyond
    # the cap are skipped below (no vector, no prediction) and belong to the
    # re-encode chain (taste contract V20).
    backfill = [
        img for img in images if not has_current_embedding(img) or not img.phash
    ]
    left_to_chain = max(0, len(backfill) - CLASSIFY_BACKFILL_LIMIT)
    backfill = backfill[:CLASSIFY_BACKFILL_LIMIT]
    if left_to_chain:
        logger.info(
            "classify_images: {} stale images left to the re-encode chain", left_to_chain
        )
    path_to_emb: dict[Path, object] = {}
    if backfill:
        if encoder is None:
            encoder = brain.get_encoder()
        if transform is None:
            transform = brain.get_transform()
        backfill_paths = [data_dir / img.file_path for img in backfill]
        embeddings, valid_paths = brain.encode(
            encoder,
            backfill_paths,
            transform=transform,
            progress_label="classify_images",
        )
        path_to_emb = dict(zip(valid_paths, embeddings, strict=True))

    taste_model = None
    if need_vision:
        taste_model = taste.load_taste_model(vision.weights_path)

    if nsfw_clf is None and need_nsfw:
        nsfw_clf = brain.load_classifier(vision.nsfw_weights_path)

    nsfw_tagged = 0
    nsfw_untagged = 0
    waiting_for_search_vector = 0
    for img in images:
        path = data_dir / img.file_path
        update_fields: list[str] = []

        if has_current_embedding(img):
            emb = brain.bytes_to_embedding(bytes(img.embedding))
        else:
            emb = path_to_emb.get(path)
            if emb is None:
                continue
            img.embedding = brain.embedding_to_bytes(emb)
            img.embedding_model = brain.ENCODER_ID
            update_fields.extend(["embedding", "embedding_model"])

        if not img.phash:
            from core import phash as phash_mod

            img.phash = phash_mod.compute_phash(path)
            update_fields.append("phash")

        if nsfw_clf is not None and not img.nsfw_judged:
            predicted = nsfw.predict_nsfw(nsfw_clf, emb, vision.nsfw_threshold)
            if predicted != img.is_nsfw:
                # The flag picks the taste category, so the stored prediction
                # came from the wrong model the moment it flips; the block
                # below renews it when the feature is there (N5).
                img.is_nsfw = predicted
                img.predicted_score = None
                update_fields.extend(["is_nsfw", "predicted_score"])
                if predicted:
                    nsfw_tagged += 1
                else:
                    nsfw_untagged += 1

        if taste_model is not None:
            # The DINOv3 half is current by now (read or just encoded above);
            # the SigLIP2 half comes from the index chain, never from here
            # (taste contract V13, V15). Without it the row keeps NULL, which
            # the review queue reads as "show anyway".
            if not features.has_taste_features(img):
                waiting_for_search_vector += 1
            else:
                # Every image is judged by its own category's model, or by the
                # shared one when the category has none (taste contract V5).
                # The category reads is_nsfw, which the NSFW head above has
                # already settled for this row (V25).
                group = taste.taste_group(img.source_label, img.is_nsfw)
                classifier = taste_model.classifier_for(group)
                prob = float(brain.predict_proba(classifier, features.taste_features(img)))
                img.predicted_score = prob
                update_fields.append("predicted_score")

        if update_fields:
            img.save(update_fields=list(dict.fromkeys(update_fields)))

    if waiting_for_search_vector:
        logger.info(
            "classify_images: {} unrated images wait for the search index",
            waiting_for_search_vector,
        )
    logger.info(
        "Classified: {} NSFW-tagged, {} untagged, {} total processed.",
        nsfw_tagged,
        nsfw_untagged,
        len(images),
    )
    counts.update(processed=len(images), nsfw_tagged=nsfw_tagged, nsfw_untagged=nsfw_untagged)
    return counts


def populate_knn_tag_suggestions(
    *,
    refill: bool = False,
    limit: int | None = None,
    k: int = 15,
    max_suggestions: int = 8,
    min_similarity: float = 0.35,
) -> dict[str, int]:
    """
    Fill Image.knn_tag_suggestions by inheriting tags from k visually-similar
    already-tagged images via cosine similarity over DINOv3 embeddings.

    The point: kNN over embeddings is the cheapest way to surface the user's
    own taste tags ("warhammer40k", "cursed") on new images without ever
    retraining a model — every new tag the user adds immediately improves
    what neighbours can inherit.

    Anchor matrix is built once per call so this is O(M*N) over numpy, not
    O(M*N) DB hits. At ~1k images this is milliseconds.
    """
    import numpy as np

    from core import brain

    # Anchors and targets are both restricted to the current encoder (V2);
    # a neighbour from another embedding space would inherit tags by chance.
    anchors = list(
        Image.objects.filter(
            is_purged=False, tags__isnull=False, embedding_model=brain.ENCODER_ID
        )
        .exclude(embedding=None)
        .distinct()
        .prefetch_related("tags")
    )
    if not anchors:
        logger.info("kNN suggestions: no tagged images to inherit from yet.")
        return {"updated": 0, "skipped": 0}

    anchor_embs = np.stack(
        [
            np.asarray(brain.bytes_to_embedding(bytes(a.embedding)), dtype=np.float32)
            for a in anchors
        ]
    )
    anchor_norms = anchor_embs / (
        np.linalg.norm(anchor_embs, axis=1, keepdims=True) + 1e-12
    )
    anchor_tags = [list(a.tags.values_list("name", flat=True)) for a in anchors]
    anchor_hashes = [a.content_hash for a in anchors]

    qs = Image.objects.filter(
        is_purged=False, embedding_model=brain.ENCODER_ID
    ).exclude(embedding=None)
    if not refill:
        qs = qs.filter(knn_tag_suggestions="")
    if limit:
        qs = qs[:limit]
    targets = list(qs.prefetch_related("tags"))
    if not targets:
        return {"updated": 0, "skipped": 0}

    logger.info(
        "kNN suggestions: {} anchor(s) tagged, computing for {} target(s).",
        len(anchors),
        len(targets),
    )
    updated = 0
    for i, img in enumerate(targets, start=1):
        target = np.asarray(
            brain.bytes_to_embedding(bytes(img.embedding)), dtype=np.float32
        )
        t_norm = target / (np.linalg.norm(target) + 1e-12)
        sims = anchor_norms @ t_norm  # cosine similarity, shape (N,)
        order = np.argsort(-sims)

        applied = set(img.tags.values_list("name", flat=True))
        scores: dict[str, float] = {}
        picked = 0
        for idx in order:
            sim = float(sims[idx])
            # Order is descending: once we drop below the threshold every later
            # neighbour is below it too. min_similarity is the cold-start guard
            # — with few anchors, k alone pulls in unrelated images and every
            # tag ends up suggested for every target.
            if sim < min_similarity:
                break
            # Skip the image itself if it happens to be in the anchor set
            # (image is tagged AND we're refilling its own row).
            if anchor_hashes[idx] == img.content_hash:
                continue
            for name in anchor_tags[idx]:
                if name in applied:
                    continue
                scores[name] = scores.get(name, 0.0) + sim
            picked += 1
            if picked >= k:
                break

        top = sorted(scores.items(), key=lambda kv: -kv[1])[:max_suggestions]
        img.knn_tag_suggestions = ",".join(name for name, _ in top)
        img.save(update_fields=["knn_tag_suggestions"])
        updated += 1
        if i % 50 == 0 or i == len(targets):
            logger.info("kNN suggestions progress: {}/{}", i, len(targets))

    logger.info("kNN suggestions: {} updated.", updated)
    return {"updated": updated, "skipped": 0}


@dataclass
class _ScrapeRun:
    """
    Everything one scrape run shares across its sources: where files go, the
    dedup index, the loaded encoder, and the classifier settings. Built once
    in run() so each per-source helper takes one argument instead of seven.
    """

    data_dir: Path
    images_dir: Path
    index: dedup.DedupIndex
    encoder: object
    transform: object
    vision: VisionConfig
    nsfw_clf: object | None


def _prepare_scrape_run(data_dir: Path, vision: VisionConfig) -> _ScrapeRun:
    """Create the images directory, load the dedup index and the models once per run."""
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    index = dedup.DedupIndex.from_db()
    stale = stale_images().exclude(embedding=None).count()
    if stale:
        logger.warning(
            "{} images still carry embeddings from an older encoder; the cosine "
            "dedup layer skips them until they are re-encoded (run Train or "
            "`manage.py reencode_embeddings`).",
            stale,
        )
    nsfw_clf = None
    if vision.nsfw_weights_path and vision.nsfw_weights_path.exists():
        nsfw_clf = brain.load_classifier(vision.nsfw_weights_path)
    return _ScrapeRun(
        data_dir=data_dir,
        images_dir=images_dir,
        index=index,
        encoder=brain.get_encoder(),
        transform=brain.get_transform(),
        vision=vision,
        nsfw_clf=nsfw_clf,
    )


def _ingest(scrape: _ScrapeRun, downloaded: list[tuple[Path, str, str]]) -> int:
    """Run one source's downloads through the dedup pipeline; returns the rows inserted."""
    return _process_downloads(
        downloaded,
        scrape.data_dir,
        scrape.index,
        scrape.encoder,
        scrape.transform,
        scrape.vision,
        scrape.nsfw_clf,
    )


def _scrape_simple_source(module, name: str, scrape: _ScrapeRun) -> int:
    """One imgur topic or tumblr blog: a plain URL list, no cursor."""
    urls = module.iter_image_urls(name)
    downloaded = module.download_images(
        urls, scrape.images_dir, name, skip_dirs=[scrape.images_dir]
    )
    return _ingest(scrape, downloaded)


def _scrape_board(board: str, max_threads: int | None, scrape: _ScrapeRun) -> int:
    """
    One 4chan board, incrementally: the board's Source.cursor holds the
    last_modified high-water mark so only threads touched since the last
    scrape are fetched again. max_threads is an optional global safety valve
    (see fourchan.iter_image_urls). The cursor is written after the downloads
    are in the DB, so a failed batch is retried from the old mark.
    """
    source_obj = Source.objects.filter(type=Source.FOURCHAN, name=board).first()
    since_modified = _cursor_int(source_obj.cursor if source_obj else None)
    urls, new_cursor = fourchan.iter_image_urls(
        board, since_modified=since_modified, max_threads=max_threads
    )
    downloaded = fourchan.download_images(
        urls, scrape.images_dir, board, skip_dirs=[scrape.images_dir]
    )
    inserted = _ingest(scrape, downloaded)
    if new_cursor is not None and source_obj:
        source_obj.cursor = str(new_cursor)
        source_obj.save(update_fields=["cursor"])
    return inserted


def _scrape_account(module, source_type: str, account: str, token: str | None, scrape: _ScrapeRun) -> int:
    """
    One Pixelfed or Mastodon account, incrementally: both speak the same API,
    and the account's Source.cursor is the since_id of the newest status seen.
    """
    source_obj = Source.objects.filter(type=source_type, name=account).first()
    since_id = source_obj.cursor if source_obj else None
    items, new_cursor = module.iter_image_items(
        account, since_id=since_id, access_token=token
    )
    downloaded = module.download_images(
        items, scrape.images_dir, account, skip_dirs=[scrape.images_dir]
    )
    inserted = _ingest(scrape, downloaded)
    if new_cursor and source_obj:
        source_obj.cursor = new_cursor
        source_obj.save(update_fields=["cursor"])
    return inserted


def _scrape_flickr(name: str, api_key: str | None, scrape: _ScrapeRun) -> int:
    """
    One Flickr group pool or photostream. With an API key the source's cursor
    is the newest added/upload time seen and moves only after the downloads
    are in the DB; without a key the feed returns no cursor and the stored one
    stays as it is (see retina/flickr.py).
    """
    source_obj = Source.objects.filter(type=Source.FLICKR, name=name).first()
    since = source_obj.cursor if source_obj else None
    items, new_cursor = flickr.iter_image_items(name, since=since, api_key=api_key)
    downloaded = flickr.download_images(
        items, scrape.images_dir, name, skip_dirs=[scrape.images_dir]
    )
    inserted = _ingest(scrape, downloaded)
    if new_cursor and source_obj:
        source_obj.cursor = new_cursor
        source_obj.save(update_fields=["cursor"])
    return inserted


def run(
    config_path: Path,
    data_dir: Path,
    vision: VisionConfig | None = None,
) -> dict[str, int]:
    """
    Scrape all enabled sources into data_dir/images/. Returns per-source
    new-image counts.

    Every source is wrapped in its own try/except so that a failure in one
    (e.g. a socket read timeout that escapes the scraper's own handlers) is
    logged and skipped rather than aborting every later source in the batch.
    """
    if vision is None:
        vision = VisionConfig()

    cfg = _load_config(config_path)
    sources = _load_sources(config_path)
    scrape = _prepare_scrape_run(data_dir, vision)
    counts: dict[str, int] = {}

    for module, sources_key, label_fmt in _SIMPLE_SCRAPERS:
        for name in sources[sources_key]:
            label = label_fmt.format(name)
            logger.info("Scraping {}", label)
            try:
                counts[label] = _scrape_simple_source(module, name, scrape)
            except Exception as e:
                logger.exception("Scraping {} failed: {}", label, e)

    fourchan_max_threads = cfg.get("4chan", {}).get("max_threads")
    for board in sources["boards"]:
        label = f"4chan/{board}"
        logger.info("Scraping {}", label)
        try:
            counts[label] = _scrape_board(board, fourchan_max_threads, scrape)
        except Exception as e:
            logger.exception("Scraping {} failed: {}", label, e)

    for module, source_type, section, service in _ACCOUNT_SCRAPERS:
        token = cfg.get(section, {}).get("access_token", "").strip() or None
        for account in sources[f"{section}_accounts"]:
            logger.info("Scraping {}: {}", service, account)
            try:
                counts[f"{section}/{account}"] = _scrape_account(
                    module, source_type, account, token, scrape
                )
            except Exception as e:
                logger.exception("Scraping {} {} failed: {}", service, account, e)

    flickr_api_key = _flickr_api_key()
    for name in sources["flickr_sources"]:
        label = f"flickr/{name}"
        logger.info("Scraping {}", label)
        try:
            counts[label] = _scrape_flickr(name, flickr_api_key, scrape)
        except Exception as e:
            logger.exception("Scraping {} failed: {}", label, e)

    has_taste_weights = bool(vision.weights_path and vision.weights_path.exists())
    if has_taste_weights or scrape.nsfw_clf is not None:
        classify_images(
            data_dir,
            vision,
            encoder=scrape.encoder,
            transform=scrape.transform,
            nsfw_clf=scrape.nsfw_clf,
        )

    populate_knn_tag_suggestions()

    return counts


def _flickr_api_key() -> str | None:
    """FLICKR_API_KEY from the environment via settings; None selects the public feed."""
    from django.conf import settings as dj_settings

    if not dj_settings.FLICKR_API_KEY:
        return None
    return dj_settings.FLICKR_API_KEY


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
