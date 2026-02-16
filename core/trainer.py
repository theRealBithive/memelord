"""Script that learns your taste from corpus and void samples."""

from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from core import brain


def _normalize_path_key(path: Path, data_dir: Path) -> str:
    """Return path relative to data_dir with forward slashes (for engagement map lookup)."""
    try:
        rel = path.relative_to(data_dir)
    except ValueError:
        return path.as_posix()
    return rel.as_posix()


def collect_image_paths(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Collect image paths from data_dir/corpus (positive) and data_dir/void (negative).

    Returns:
        (corpus_paths, void_paths). Only files with supported image extensions are included.
    """
    corpus_dir = data_dir / "corpus"
    void_dir = data_dir / "void"
    corpus_paths = sorted(
        p for p in corpus_dir.glob("*") if p.is_file() and brain.is_image_path(p)
    )
    void_paths = sorted(
        p for p in void_dir.glob("*") if p.is_file() and brain.is_image_path(p)
    )
    return corpus_paths, void_paths


def run(
    data_dir: Path = Path("data"),
    weights_path: Path = Path("Janulon_weights.pkl"),
    db_path: Path | None = None,
) -> None:
    """
    Train the taste classifier on corpus (label 1) and void (label 0), then save weights.

    Requires data_dir/corpus/ and data_dir/void/ to contain images. If db_path is
    given, posted images with engagement data are weighted by: faves (+1),
    replies (+0.5), reblogs (+2); other corpus images use weight 1.0.
    """
    from core import db as db_module

    corpus_paths, void_paths = collect_image_paths(data_dir)
    if not corpus_paths:
        logger.warning("No images in {}/corpus/", data_dir)
    if not void_paths:
        logger.warning("No images in {}/void/", data_dir)
    if not corpus_paths or not void_paths:
        raise SystemExit(1)

    engagement_weights: dict[str, float] = {}
    if db_path is not None and db_path.exists():
        db_module.init_db(db_path)
        engagement_weights = db_module.get_posted_engagement_weights()
        if engagement_weights:
            logger.info(
                "Engagement weighting: {} posted images from DB (faves +0.5*replies +2*reblogs)",
                len(engagement_weights),
            )

    corpus_weights = np.array(
        [
            engagement_weights.get(_normalize_path_key(p, data_dir), 1.0)
            for p in corpus_paths
        ],
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

    logger.info(
        "Training logistic regression on {} samples (with sample weights)",
        len(y),
    )
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X, y, sample_weight=sample_weight)

    brain.save_classifier(classifier, weights_path)
    logger.success("Saved classifier to {}", weights_path.resolve())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Janulon taste classifier on corpus and void."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing corpus/ and void/ subdirs. Default: data",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("Janulon_weights.pkl"),
        help="Output path for classifier weights. Default: Janulon_weights.pkl",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional SQLite DB path. If set, posted images are weighted by engagement (faves +0.5*replies +2*reblogs).",
    )
    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        weights_path=args.weights,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
