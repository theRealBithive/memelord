"""Script that learns your taste from corpus and void samples."""

from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from core import brain


def collect_image_paths(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Collect image paths from data_dir/corpus and data_dir/void via the Django ORM.

    Returns (corpus_paths, void_paths) as absolute paths. Only non-deleted files
    that exist on disk are included.
    """
    from ratings.models import Image

    def _resolve(img: Image) -> Path:
        return data_dir / img.file_path

    corpus_paths = [
        _resolve(img)
        for img in Image.objects.filter(location=Image.CORPUS, file_deleted=False)
        if _resolve(img).exists() and brain.is_image_path(_resolve(img))
    ]
    void_paths = [
        _resolve(img)
        for img in Image.objects.filter(location=Image.VOID, file_deleted=False)
        if _resolve(img).exists() and brain.is_image_path(_resolve(img))
    ]
    return sorted(corpus_paths), sorted(void_paths)


def _get_favourite_weights(data_dir: Path) -> dict[str, float]:
    """Return {absolute_path_str: weight} for corpus images. Favs get 3.0, others 1.0."""
    from ratings.models import Image

    return {
        str(data_dir / img.file_path): 3.0 if img.is_favourite else 1.0
        for img in Image.objects.filter(location=Image.CORPUS, file_deleted=False)
    }


def run(
    data_dir: Path = Path("data"),
    weights_path: Path = Path("Janulon_weights.pkl"),
) -> None:
    """
    Train the taste classifier on corpus (label 1) and void (label 0), then save weights.

    Requires rated images in the DB with location='corpus' or 'void'. Favourite
    corpus images are weighted 3.0; all others are weighted 1.0.
    """
    corpus_paths, void_paths = collect_image_paths(data_dir)
    if not corpus_paths:
        logger.warning("No corpus images found in DB / on disk at {}", data_dir)
    if not void_paths:
        logger.warning("No void images found in DB / on disk at {}", data_dir)
    if not corpus_paths or not void_paths:
        raise SystemExit(1)

    favourite_weights = _get_favourite_weights(data_dir)
    corpus_weights = np.array(
        [favourite_weights.get(str(p), 1.0) for p in corpus_paths],
        dtype=np.float64,
    )
    void_weights = np.ones(len(void_paths), dtype=np.float64)
    sample_weight = np.concatenate([corpus_weights, void_weights])

    logger.info(
        "Encoding {} corpus + {} void images with DINOv2",
        len(corpus_paths),
        len(void_paths),
    )
    encoder = brain.get_encoder()
    transform = brain.get_transform()
    X_corpus = brain.encode(encoder, corpus_paths, transform=transform)
    X_void = brain.encode(encoder, void_paths, transform=transform)
    X = np.concatenate([X_corpus, X_void], axis=0)
    y = np.array([1] * len(corpus_paths) + [0] * len(void_paths), dtype=np.intp)

    logger.info("Training logistic regression on {} samples", len(y))
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X, y, sample_weight=sample_weight)

    brain.save_classifier(classifier, weights_path)
    logger.success("Saved classifier to {}", weights_path.resolve())
