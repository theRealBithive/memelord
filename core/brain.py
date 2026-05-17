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

# Supported image extensions for corpus/void
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

EMBEDDING_DIM = 768


def get_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_encoder(device: str | torch.device | None = None) -> torch.nn.Module:
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
) -> tuple[np.ndarray, list[Path]]:
    """
    Encode images to 768-d DINOv2 embeddings.

    Returns:
        (embeddings, valid_paths) — embeddings shape (N, 768) float32, and the
        subset of image_paths that were successfully read. Files that are missing
        or unreadable at encode time are skipped with a warning so that a file
        moved mid-training does not crash the job.
    """
    import logging

    if not image_paths:
        return np.zeros((0, 768), dtype=np.float32), []

    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if transform is None:
        transform = get_transform()

    embeddings: list[np.ndarray] = []
    valid_paths: list[Path] = []
    for start in range(0, len(image_paths), batch_size):
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

    if not embeddings:
        return np.zeros((0, 768), dtype=np.float32), []
    return np.vstack(embeddings), valid_paths


def load_classifier(path: Path) -> LogisticRegression:
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def save_classifier(classifier: LogisticRegression, path: Path) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(classifier, f)


def predict_proba(
    classifier: LogisticRegression,
    embedding: np.ndarray,
) -> np.ndarray:
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    proba = classifier.predict_proba(embedding)[:, 1]
    return proba if proba.shape[0] > 1 else proba[0]


def is_image_path(path: Path) -> bool:
    """True if path has a supported image extension (case-insensitive)."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-d embedding, got {arr.shape[0]}")
    return arr.tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.float32)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM} floats, got {arr.shape[0]}")
    return arr


def cosine_similarity_matrix(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between query row(s) and bank rows.

    Args:
        query: Shape (768,) or (N, 768).
        bank: Shape (M, 768). Empty bank returns empty array.

    Returns:
        Shape (N, M) or (M,) when query is 1-d.
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
