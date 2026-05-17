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
    """Return (nsfw_paths, safe_paths) from Image.is_nsfw labels."""
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
    """Return {absolute_path_str: weight} for corpus images. Favs get 3.0, others 1.0."""
    from ratings.models import Image

    return {
        str(data_dir / img.file_path): 3.0 if img.is_favourite else 1.0
        for img in Image.objects.filter(location=Image.CORPUS, file_deleted=False)
    }


def _path_to_image_map(data_dir: Path) -> dict[str, object]:
    """Map absolute path string to Image ORM instance."""
    from ratings.models import Image

    out: dict[str, object] = {}
    for img in Image.objects.filter(file_deleted=False):
        path = data_dir / img.file_path
        if path.exists():
            out[str(path)] = img
    return out


def _backfill_phash_embedding(
    path_to_emb: dict[str, np.ndarray],
    data_dir: Path,
) -> None:
    """Persist phash and embedding on Image rows after encoding."""
    from core import phash as phash_mod

    path_to_img = _path_to_image_map(data_dir)
    for path_str, emb in path_to_emb.items():
        img = path_to_img.get(path_str)
        if img is None:
            continue
        ph = phash_mod.compute_phash(Path(path_str))
        img.phash = ph
        img.embedding = brain.embedding_to_bytes(emb)
        img.save(update_fields=["phash", "embedding"])


def run(
    data_dir: Path = Path("data"),
    weights_path: Path = Path("Janulon_weights.pkl"),
    nsfw_weights_path: Path | None = None,
    nsfw_threshold: float = 0.30,
) -> None:
    """
    Train taste classifier on corpus (1) vs void (0), optionally NSFW head.

    Encodes each unique image path once and fits classifiers on shared embeddings.
    """
    corpus_paths, void_paths = collect_image_paths(data_dir)
    if not corpus_paths:
        logger.warning("No corpus images found in DB / on disk at {}", data_dir)
    if not void_paths:
        logger.warning("No void images found in DB / on disk at {}", data_dir)
    if not corpus_paths or not void_paths:
        raise SystemExit(1)

    nsfw_paths, safe_paths = collect_nsfw_paths(data_dir)
    train_nsfw = bool(nsfw_paths and safe_paths)
    if nsfw_weights_path and not train_nsfw:
        logger.warning(
            "Skipping NSFW classifier: need at least one is_nsfw=True and one False image."
        )

    all_paths = sorted(
        set(corpus_paths + void_paths + (nsfw_paths + safe_paths if train_nsfw else []))
    )

    logger.info("Encoding {} unique images with DINOv2", len(all_paths))
    encoder = brain.get_encoder()
    transform = brain.get_transform()
    X_all = brain.encode(encoder, all_paths, transform=transform)
    path_to_emb = {str(p): X_all[i] for i, p in enumerate(all_paths)}

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
