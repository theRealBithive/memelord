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


def get_transform() -> T.Compose:
    """DINOv2 preprocessing: resize 256, center crop 224, ImageNet normalize."""
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_encoder(device: str | torch.device | None = None) -> torch.nn.Module:
    """Load DINOv2 ViT-B/14 from torch hub; returns model in eval mode."""
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
) -> np.ndarray:
    """
    Encode images to 768-d DINOv2 embeddings.

    Args:
        encoder: DINOv2 model from get_encoder().
        image_paths: List of paths to image files.
        transform: Preprocessing transform; uses get_transform() if None.
        device: Device for the encoder; inferred if None.
        batch_size: Batch size for forward passes.

    Returns:
        Array of shape (N, 768), float32.
    """
    if not image_paths:
        return np.zeros((0, 768), dtype=np.float32)

    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if transform is None:
        transform = get_transform()

    embeddings: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        end = start + batch_size
        batch_paths = image_paths[start:end]
        tensors: list[torch.Tensor] = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            t = transform(img)
            tensors.append(t)
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            out = encoder(batch)
        # DINOv2 returns (B, 768) for the CLS token
        if out.dim() == 3:
            out = out[:, 0, :]
        embeddings.append(out.cpu().numpy().astype(np.float32))

    return np.vstack(embeddings)


def load_classifier(path: Path) -> LogisticRegression:
    """Load a trained classifier from a pickle file."""
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def save_classifier(classifier: LogisticRegression, path: Path) -> None:
    """Save a trained classifier to a pickle file."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(classifier, f)


def predict_proba(
    classifier: LogisticRegression,
    embedding: np.ndarray,
) -> np.ndarray:
    """
    Predict P(positive) for one or more embeddings.

    Args:
        classifier: Trained sklearn classifier (e.g. LogisticRegression).
        embedding: Shape (768,) or (N, 768).

    Returns:
        Probability of class 1, shape () or (N,).
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    proba = classifier.predict_proba(embedding)[:, 1]
    return proba if proba.shape[0] > 1 else proba[0]


def is_image_path(path: Path) -> bool:
    """True if path has a supported image extension (case-insensitive)."""
    return path.suffix.lower() in IMAGE_EXTENSIONS
