"""Script that learns your taste from scored image samples."""

from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from core import brain, nsfw

# Scores >= HIGH_SCORE are positive (good) training samples; scores <= LOW_SCORE
# are negative. The gap is intentional: a strict split between "liked" and
# "disliked" keeps ambiguous middle ground out of training.
HIGH_SCORE = 3
LOW_SCORE = 2

# Per-score sample weights, kept symmetric between the liked and disliked ends.
# Scores 5-6 are boosted 3× so strong favourites outweigh mildly-liked images;
# 3-4 get baseline 1.0. The negative side mirrors this: score 1-2 get 1.0, and
# score 0 ("trash" — garbage the user wouldn't even rate a 1) gets 3× so the
# model learns hard to avoid it, exactly as a 6 pulls toward favourites.
_POSITIVE_WEIGHTS: dict[int, float] = {3: 1.0, 4: 1.0, 5: 3.0, 6: 3.0}
_NEGATIVE_WEIGHTS: dict[int, float] = {0: 3.0, 1: 1.0, 2: 1.0}


def collect_image_paths(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Collect positive and negative image paths via the Django ORM.

    Returns (good_paths, bad_paths) as absolute paths. Only files that exist
    on disk are included. good = score >= HIGH_SCORE, bad = score <= LOW_SCORE.
    """
    from ratings.models import Image

    good_paths: list[Path] = []
    for img in Image.objects.filter(is_purged=False, score__gte=HIGH_SCORE):
        path = data_dir / img.file_path
        if path.exists() and brain.is_image_path(path):
            good_paths.append(path)

    bad_paths: list[Path] = []
    for img in Image.objects.filter(is_purged=False, score__lte=LOW_SCORE):
        path = data_dir / img.file_path
        if path.exists() and brain.is_image_path(path):
            bad_paths.append(path)

    return sorted(good_paths), sorted(bad_paths)


def collect_nsfw_paths(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Return (nsfw_paths, safe_paths) for NSFW classifier training.

    Uses is_nsfw labels across all images regardless of score, so the NSFW
    head is trained on the full available signal.
    """
    from ratings.models import Image

    nsfw_paths: list[Path] = []
    safe_paths: list[Path] = []
    for img in Image.objects.filter(is_purged=False):
        path = data_dir / img.file_path
        if not path.exists() or not brain.is_image_path(path):
            continue
        if img.is_nsfw:
            nsfw_paths.append(path)
        else:
            safe_paths.append(path)
    return sorted(nsfw_paths), sorted(safe_paths)


def _get_sample_weights(data_dir: Path) -> dict[str, float]:
    """
    Return {absolute_path_str: weight} for positive-class images (score >= HIGH_SCORE).

    Negative samples are keyed separately by _get_negative_weights. Keeping the
    two tables apart makes the caller's intent clear and avoids accidentally
    boosting bad samples via the positive-score formula.
    """
    from ratings.models import Image

    result = {}
    for img in Image.objects.filter(is_purged=False, score__gte=HIGH_SCORE):
        path_str = str(data_dir / img.file_path)
        result[path_str] = _POSITIVE_WEIGHTS.get(img.score, 1.0)
    return result


def _get_negative_weights(data_dir: Path) -> dict[str, float]:
    """
    Return {absolute_path_str: weight} for negative-class images (score <= LOW_SCORE).

    Score 0 (trash) is weighted 3× via _NEGATIVE_WEIGHTS; score 1-2 get 1.0.
    Anything outside the table falls back to 1.0 so an unexpected score never
    silently drops a sample to zero weight.
    """
    from ratings.models import Image

    result = {}
    for img in Image.objects.filter(is_purged=False, score__lte=LOW_SCORE):
        path_str = str(data_dir / img.file_path)
        result[path_str] = _NEGATIVE_WEIGHTS.get(img.score, 1.0)
    return result


def _load_cached_embeddings(
    data_dir: Path, paths: list[Path]
) -> dict[str, np.ndarray]:
    """
    Return {absolute_path_str: embedding} for the subset of `paths` whose Image
    row already has a stored DINOv2 vector in the `embedding` column.

    Scoped to the caller's path list (not the whole table) so the returned dict
    matches the training set exactly — downstream phash backfill iterates these
    keys, and widening them would scan rows this run never trains on.

    Re-encoding the whole corpus through DINOv2 on every train run was the
    dominant cost on large libraries — a ViT-B/14 forward pass over tens of
    thousands of images on CPU runs for hours and previously tripped the
    django-q worker timeout. The embeddings are already persisted at scrape time
    and by classify_images, so training reads them back here and only encodes the
    (normally empty) set of rows still missing one.

    Cache validity is model-scoped: the stored vectors are only correct while the
    DINOv2 variant + transform are unchanged. eval()-mode DINOv2 is deterministic
    so cached == freshly-encoded today, but there is no version stamp on the
    column. A bad blob is skipped rather than fatal: bytes_to_embedding raises on
    any vector that isn't exactly EMBEDDING_DIM floats, so an encoder swap that
    changes dimensionality would otherwise crash every train run with no path to
    rebuild the column. Skipping lets the row fall through to the re-encode set
    and self-heal, mirroring brain.encode's "a moved file doesn't kill the job".
    """
    from ratings.models import Image

    # Filter relative file_paths in the query so we touch only rows this run may
    # train on, not every embedded row in the table.
    rel_paths = [str(p.relative_to(data_dir)) for p in paths]
    cache: dict[str, np.ndarray] = {}
    for img in Image.objects.filter(
        is_purged=False, embedding__isnull=False, file_path__in=rel_paths
    ):
        path_str = str(data_dir / img.file_path)
        try:
            cache[path_str] = brain.bytes_to_embedding(bytes(img.embedding))
        except ValueError as exc:
            logger.warning(
                "train: ignoring bad cached embedding for {} ({})", path_str, exc
            )
    return cache


def _backfill_phash_embedding(
    path_to_emb: dict[str, np.ndarray],
    data_dir: Path,
) -> None:
    """
    Persist phash and embedding on Image rows after encoding.

    Only fills rows that don't already have a value — re-saving every row on
    every training run was previously the dominant cost on large libraries
    (one disk read for phash plus one DB write per image, even when both
    fields were already populated from the prior scrape).
    """
    from core import phash as phash_mod
    from ratings.models import Image

    path_to_img = {
        str(data_dir / img.file_path): img
        for img in Image.objects.filter(is_purged=False)
        if (data_dir / img.file_path).exists()
    }
    backfilled_phash = 0
    backfilled_emb = 0
    for path_str, emb in path_to_emb.items():
        img = path_to_img.get(path_str)
        if img is None:
            continue
        update_fields: list[str] = []
        if not img.phash:
            img.phash = phash_mod.compute_phash(Path(path_str))
            update_fields.append("phash")
            backfilled_phash += 1
        if img.embedding is None:
            img.embedding = brain.embedding_to_bytes(emb)
            update_fields.append("embedding")
            backfilled_emb += 1
        if update_fields:
            img.save(update_fields=update_fields)
    if backfilled_phash or backfilled_emb:
        logger.info(
            "Backfilled {} phash and {} embedding rows",
            backfilled_phash, backfilled_emb,
        )
    else:
        logger.info("No phash/embedding backfill needed (all rows up to date)")


def run(
    data_dir: Path = Path("data"),
    weights_path: Path = Path("Janulon_weights.pkl"),
    nsfw_weights_path: Path | None = None,
    nsfw_threshold: float = 0.30,
) -> None:
    """
    Train taste classifier on good (score >= HIGH_SCORE) vs bad (score <= LOW_SCORE)
    samples, and optionally an NSFW head.

    Embeddings are read from the cached `embedding` column (see
    _load_cached_embeddings); only rows still missing one are encoded with
    DINOv2. On a warm cache this skips the encoder entirely and training is a
    matter of seconds — the whole-corpus forward pass it replaces ran for hours
    and tripped the worker timeout. The freshly-encoded vectors are then sliced
    for each classifier so the encoder runs at most once per train.
    """
    good_paths, bad_paths = collect_image_paths(data_dir)
    if not good_paths:
        logger.warning("No good images (score >= {}) found at {}", HIGH_SCORE, data_dir)
    if not bad_paths:
        logger.warning("No bad images (score <= {}) found at {}", LOW_SCORE, data_dir)
    if not good_paths or not bad_paths:
        raise RuntimeError("Need at least one good and one bad image to train.")

    nsfw_paths, safe_paths = collect_nsfw_paths(data_dir)
    train_nsfw = bool(nsfw_paths and safe_paths)
    if nsfw_weights_path and not train_nsfw:
        logger.warning(
            "Skipping NSFW classifier: need at least one is_nsfw=True and one False image."
        )

    all_paths = sorted(
        set(good_paths + bad_paths + (nsfw_paths + safe_paths if train_nsfw else []))
    )

    if train_nsfw:
        logger.info(
            "Training data: {} good + {} bad, NSFW: {} nsfw + {} safe",
            len(good_paths), len(bad_paths), len(nsfw_paths), len(safe_paths),
        )
    else:
        logger.info(
            "Training data: {} good + {} bad", len(good_paths), len(bad_paths)
        )
    path_to_emb = _load_cached_embeddings(data_dir, all_paths)
    missing = [p for p in all_paths if str(p) not in path_to_emb]
    if missing:
        logger.info(
            "Encoding {} of {} images missing a cached embedding…",
            len(missing), len(all_paths),
        )
        encoder = brain.get_encoder()
        transform = brain.get_transform()
        X_missing, valid_missing = brain.encode(
            encoder, missing, transform=transform, progress_label="train"
        )
        for p, emb in zip(valid_missing, X_missing):
            path_to_emb[str(p)] = emb
    else:
        logger.info(
            "All {} training images have cached embeddings; skipping encode",
            len(all_paths),
        )

    # Re-filter each list to paths present in path_to_emb (cached or just-encoded);
    # a scored row with neither a cached embedding nor a readable file is dropped.
    good_paths = [p for p in good_paths if str(p) in path_to_emb]
    bad_paths = [p for p in bad_paths if str(p) in path_to_emb]
    nsfw_paths = [p for p in nsfw_paths if str(p) in path_to_emb]
    safe_paths = [p for p in safe_paths if str(p) in path_to_emb]
    if not good_paths or not bad_paths:
        raise RuntimeError("Need at least one good and one bad image to train.")

    _backfill_phash_embedding(path_to_emb, data_dir)

    X_good = np.array([path_to_emb[str(p)] for p in good_paths])
    X_bad = np.array([path_to_emb[str(p)] for p in bad_paths])
    X = np.concatenate([X_good, X_bad], axis=0)
    y = np.array([1] * len(good_paths) + [0] * len(bad_paths), dtype=np.intp)

    pos_weights_map = _get_sample_weights(data_dir)
    neg_weights_map = _get_negative_weights(data_dir)
    good_weights = np.array(
        [pos_weights_map.get(str(p), 1.0) for p in good_paths],
        dtype=np.float64,
    )
    bad_weights = np.array(
        [neg_weights_map.get(str(p), 1.0) for p in bad_paths],
        dtype=np.float64,
    )
    sample_weight = np.concatenate([good_weights, bad_weights])

    logger.info("Training taste classifier on {} samples", len(y))
    taste_clf = LogisticRegression(max_iter=1000, random_state=42)
    taste_clf.fit(X, y, sample_weight=sample_weight)
    brain.save_classifier(taste_clf, weights_path)
    logger.success("Saved taste classifier to {}", weights_path.resolve())

    if train_nsfw and nsfw_weights_path:
        X_nsfw = np.array([path_to_emb[str(p)] for p in nsfw_paths])
        X_safe = np.array([path_to_emb[str(p)] for p in safe_paths])
        X_n = np.concatenate([X_nsfw, X_safe], axis=0)
        y_n = np.array([1] * len(nsfw_paths) + [0] * len(safe_paths), dtype=np.intp)
        logger.info(
            "Training NSFW classifier on {} NSFW + {} safe samples",
            len(nsfw_paths),
            len(safe_paths),
        )
        nsfw_clf = nsfw.train_nsfw_classifier(X_n, y_n)
        brain.save_classifier(nsfw_clf, nsfw_weights_path)
        logger.success(
            "Saved NSFW classifier to {} (inference threshold={})",
            nsfw_weights_path.resolve(),
            nsfw_threshold,
        )
