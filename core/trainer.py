"""Script that learns your taste from corpus and void samples."""

from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from core import brain


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
) -> None:
    """
    Train the taste classifier on corpus (label 1) and void (label 0), then save weights.

    Requires data_dir/corpus/ and data_dir/void/ to contain images.
    """
    corpus_paths, void_paths = collect_image_paths(data_dir)
    if not corpus_paths:
        logger.warning("No images in {}/corpus/", data_dir)
    if not void_paths:
        logger.warning("No images in {}/void/", data_dir)
    if not corpus_paths or not void_paths:
        raise SystemExit(1)

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
    classifier.fit(X, y)

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
    args = parser.parse_args()
    run(data_dir=args.data_dir, weights_path=args.weights)


if __name__ == "__main__":
    main()
