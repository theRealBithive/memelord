"""Script that learns your taste from corpus and void samples."""

from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from core import brain, nsfw


def collect_image_paths(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Collect image paths from data_dir/corpus and data_dir/void via the Django ORM.

    Returns (corpus_paths, void_paths) as absolute paths. Only non-deleted files
    that exist on disk are included.
    """
    from ratings.models import Image

    corpus_paths: list[Path] = []
    for img in Image.objects.filter(location=Image.CORPUS, file_deleted=False):
        path = data_dir / img.file_path
        if path.exists() and brain.is_image_path(path):
            corpus_paths.append(path)

    void_paths: list[Path] = []
    for img in Image.objects.filter(location=Image.VOID, file_deleted=False):
        path = data_dir / img.file_path
        if path.exists() and brain.is_image_path(path):
            void_paths.append(path)

    return sorted(corpus_paths), sorted(void_paths)


def collect_nsfw_paths(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Return (nsfw_paths, safe_paths) for NSFW classifier training.

    Uses is_nsfw labels across all images regardless of location, so the NSFW
    head is trained on the full signal available — not just inbox or corpus
    images. A manually flagged void image is just as valid a training sample
    as a corpus one.
    """
    from ratings.models import Image

    nsfw_paths: list[Path] = []
    safe_paths: list[Path] = []
    for img in Image.objects.filter(file_deleted=False):
        path = data_dir / img.file_path
        if not path.exists() or not brain.is_image_path(path):
            continue
        if img.is_nsfw:
            nsfw_paths.append(path)
        else:
            safe_paths.append(path)
    return sorted(nsfw_paths), sorted(safe_paths)


def _get_favourite_weights(data_dir: Path) -> dict[str, float]:
    """
    Return {absolute_path_str: weight} for corpus images.

    Priority: score (1–6 mapped directly) > is_favourite (3.0) > default (1.0).
    Score takes precedence because it's a finer-grained signal than the binary
    fav flag; is_favourite acts as a fallback for images rated before the scoring
    UI was added.
    """
    from ratings.models import Image

    result = {}
    for img in Image.objects.filter(location=Image.CORPUS, file_deleted=False):
        path_str = str(data_dir / img.file_path)
        if img.score is not None:
            result[path_str] = float(img.score)
        elif img.is_favourite:
            result[path_str] = 3.0
        else:
            result[path_str] = 1.0
    return result


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
        for img in Image.objects.filter(file_deleted=False)
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
    Train taste classifier on corpus (1) vs void (0) and optionally an NSFW head.

    All unique image paths are encoded once with DINOv2 in a single forward pass,
    then the embeddings are sliced for each classifier — this avoids running the
    GPU encoder multiple times when training both taste and NSFW heads together.
    """
    corpus_paths, void_paths = collect_image_paths(data_dir)
    if not corpus_paths:
        logger.warning("No corpus images found in DB / on disk at {}", data_dir)
    if not void_paths:
        logger.warning("No void images found in DB / on disk at {}", data_dir)
    if not corpus_paths or not void_paths:
        raise RuntimeError("Need at least one corpus and one void image to train.")

    nsfw_paths, safe_paths = collect_nsfw_paths(data_dir)
    train_nsfw = bool(nsfw_paths and safe_paths)
    if nsfw_weights_path and not train_nsfw:
        logger.warning(
            "Skipping NSFW classifier: need at least one is_nsfw=True and one False image."
        )

    all_paths = sorted(
        set(corpus_paths + void_paths + (nsfw_paths + safe_paths if train_nsfw else []))
    )

    if train_nsfw:
        logger.info(
            "Training data: {} corpus + {} void, NSFW: {} nsfw + {} safe",
            len(corpus_paths), len(void_paths), len(nsfw_paths), len(safe_paths),
        )
    else:
        logger.info(
            "Training data: {} corpus + {} void", len(corpus_paths), len(void_paths)
        )
    logger.info("Loading DINOv2 encoder…")
    encoder = brain.get_encoder()
    transform = brain.get_transform()
    X_all, valid_all_paths = brain.encode(
        encoder, all_paths, transform=transform, progress_label="train"
    )
    path_to_emb = {str(p): X_all[i] for i, p in enumerate(valid_all_paths)}

    # Re-filter each list to paths that were actually encoded (handles files moved mid-run).
    corpus_paths = [p for p in corpus_paths if str(p) in path_to_emb]
    void_paths = [p for p in void_paths if str(p) in path_to_emb]
    nsfw_paths = [p for p in nsfw_paths if str(p) in path_to_emb]
    safe_paths = [p for p in safe_paths if str(p) in path_to_emb]
    if not corpus_paths or not void_paths:
        raise RuntimeError("Need at least one corpus and one void image to train.")

    _backfill_phash_embedding(path_to_emb, data_dir)

    X_corpus = np.array([path_to_emb[str(p)] for p in corpus_paths])
    X_void = np.array([path_to_emb[str(p)] for p in void_paths])
    X = np.concatenate([X_corpus, X_void], axis=0)
    y = np.array([1] * len(corpus_paths) + [0] * len(void_paths), dtype=np.intp)

    favourite_weights = _get_favourite_weights(data_dir)
    corpus_weights = np.array(
        [favourite_weights.get(str(p), 1.0) for p in corpus_paths],
        dtype=np.float64,
    )
    void_weights = np.ones(len(void_paths), dtype=np.float64)
    sample_weight = np.concatenate([corpus_weights, void_weights])

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
