"""The neural logic: DINOv2 encoder + classifier (taste matrix)."""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression

# ImageNet normalization for DINOv2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

EMBEDDING_DIM = 768


def get_transform() -> T.Compose:
    """
    DINOv2 ViT-B/14 was pretrained with exactly this pipeline (256→224 bicubic
    crop, ImageNet mean/std). Deviating produces out-of-distribution inputs and
    degrades embedding quality, which in turn harms dedup precision and taste
    classifier accuracy.
    """
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_encoder(device: str | torch.device | None = None) -> torch.nn.Module:
    """
    ViT-B/14 is the quality/speed sweet spot for this workload — ViT-S loses
    too much semantic detail for taste discrimination, ViT-L is too slow for
    scrape-time dedup on a single GPU. eval() is required for deterministic
    outputs: stochastic layers in train mode would produce different embeddings
    for the same image, breaking cosine-similarity dedup comparisons.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model.eval()
    return model.to(device)


def encode(
    encoder: torch.nn.Module,
    image_paths: list[Path],
    transform: T.Compose | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 32,
    progress_label: str = "encode",
) -> tuple[np.ndarray, list[Path]]:
    """
    Encode images to 768-d DINOv2 embeddings.

    Per-batch progress is logged via loguru so the in-app log viewer shows the
    job is alive — a silent ViT-B/14 forward pass over thousands of images on
    CPU can take 30+ minutes, which previously looked indistinguishable from a
    crashed worker. progress_label disambiguates concurrent encode passes (e.g.
    "train", "scrape") in the log stream.

    Returns:
        (embeddings, valid_paths) — embeddings shape (N, 768) float32, and the
        subset of image_paths that were successfully read. Files that are missing
        or unreadable at encode time are skipped with a warning so that a file
        moved mid-training does not crash the job.
    """
    import logging
    import time

    from loguru import logger

    if not image_paths:
        return np.zeros((0, 768), dtype=np.float32), []

    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if transform is None:
        transform = get_transform()

    total = len(image_paths)
    n_batches = (total + batch_size - 1) // batch_size
    # Cap to ~20 progress lines for large jobs so the log doesn't drown in noise.
    log_every = max(1, n_batches // 20)
    logger.info(
        "{}: encoding {} images on {} ({} batches of {})",
        progress_label, total, device, n_batches, batch_size,
    )
    t_start = time.monotonic()

    embeddings: list[np.ndarray] = []
    valid_paths: list[Path] = []
    for batch_idx, start in enumerate(range(0, total, batch_size)):
        batch_paths = image_paths[start:start + batch_size]
        tensors: list[torch.Tensor] = []
        batch_valid: list[Path] = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
                batch_valid.append(p)
            except (OSError, FileNotFoundError) as exc:
                logging.warning("encode: skipping unreadable file %s (%s)", p, exc)
        if not tensors:
            continue
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            out = encoder(batch)
        # DINOv2 returns (B, 768) for the CLS token
        if out.dim() == 3:
            out = out[:, 0, :]
        embeddings.append(out.cpu().numpy().astype(np.float32))
        valid_paths.extend(batch_valid)
        done = start + len(batch_paths)
        if (batch_idx + 1) % log_every == 0 or done >= total:
            elapsed = time.monotonic() - t_start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else 0.0
            logger.info(
                "{}: {}/{} encoded ({:.1f} img/s, ETA {:.0f}s)",
                progress_label, done, total, rate, eta,
            )

    elapsed = time.monotonic() - t_start
    logger.info("{}: encoding done in {:.0f}s", progress_label, elapsed)
    if not embeddings:
        return np.zeros((0, 768), dtype=np.float32), []
    return np.vstack(embeddings), valid_paths


def load_classifier(path: Path) -> LogisticRegression:
    """
    Classifiers (taste and NSFW) are trained separately in trainer.py and
    persisted so scrape runs can auto-classify without retraining. Pickle is
    the standard sklearn serialisation format; there is no safer alternative
    for arbitrary estimators. The caller is responsible for checking that
    path exists before calling — see scraper.vision_config_from_settings().
    """
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def save_classifier(classifier: LogisticRegression, path: Path) -> None:
    """
    Writes the fitted estimator to DATA_DIR so the next scrape run can load it
    via load_classifier() without requiring a retraining pass. mkdir is
    included here so the function is safe to call before DATA_DIR is fully
    bootstrapped (e.g. during tests with a temp directory).
    """
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(classifier, f)


def predict_proba(
    classifier: LogisticRegression,
    embedding: np.ndarray,
) -> np.ndarray:
    """
    Returns P(liked) — probability the image belongs in the positive class.
    Used by classify_images() to assign predicted_score; the visibility dial
    thresholds in config are compared against this value to hide low-confidence
    images from the review queue. The 1-d reshape allows callers to pass a
    single (768,) vector without wrapping it in a batch dimension themselves.
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    proba = classifier.predict_proba(embedding)[:, 1]
    return proba if proba.shape[0] > 1 else proba[0]


def is_image_path(path: Path) -> bool:
    """True if path has a supported image extension (case-insensitive)."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """
    SQLite has no native array type, so embeddings are stored as raw float32
    blobs. The dimension check catches shape mismatches at write time rather
    than silently storing a truncated vector that would corrupt cosine-similarity
    comparisons when the DedupIndex is rebuilt from the DB on the next scrape.
    """
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-d embedding, got {arr.shape[0]}")
    return arr.tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    """
    Inverse of embedding_to_bytes(). Called when building the DedupIndex from
    the DB at scrape startup. frombuffer gives a read-only view; callers that
    need to modify the array must copy it first.
    """
    arr = np.frombuffer(data, dtype=np.float32)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM} floats, got {arr.shape[0]}")
    return arr


def cosine_similarity_matrix(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """
    Cosine similarity is the right metric here because DINOv2 embeddings lie on
    a hypersphere — angular distance is meaningful, Euclidean distance is not.
    Zero-norm guard prevents division-by-zero on degenerate images (solid-colour
    fills) that produce all-zero activations. Returns shape (M,) when query is
    1-d so is_embedding_duplicate() can call np.max() directly without squeezing.
    """
    if bank.size == 0:
        if query.ndim == 1:
            return np.array([], dtype=np.float32)
        return np.zeros((query.shape[0], 0), dtype=np.float32)

    q = np.atleast_2d(query.astype(np.float32))
    b = bank.astype(np.float32)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    q_norm = np.where(q_norm == 0, 1, q_norm)
    b_norm = np.where(b_norm == 0, 1, b_norm)
    sims = (q / q_norm) @ (b / b_norm).T
    if query.ndim == 1:
        return sims[0]
    return sims
