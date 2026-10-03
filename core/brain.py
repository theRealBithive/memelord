"""The neural logic: DINOv3 encoder + classifier (taste matrix)."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

EMBEDDING_DIM = 768

# The encoder sees every image at INPUT_SIZE x INPUT_SIZE pixels. Meta's
# processor config says 224; DINOv3 uses RoPE and derives its position
# encoding from the actual patch grid, and the LVD-1689M checkpoint went
# through a high-resolution adaptation phase, so 448 is inside what the
# weights were trained for. The point is fine structure that 224 px flattens:
# brush work on a miniature, the texture of a wallpaper. A 28 mm figure that
# fills 300 px of its photo is ~45 px wide at 224 and ~90 px at 448. Measured
# on a 20-core CPU: 0.41 s/img at 224, 1.38 s/img at 448 (3.4x), which is why
# the library is re-encoded by a sliced background chain
# (ratings/embeddings.py) and never inline in a scrape.
INPUT_SIZE = 448

# Name of the embedding generation every stored vector is stamped with
# (Image.embedding_model). A vector stamped with anything else was produced by
# a different encoder (or the same one at another resolution) and lives in a
# different space, so it is never compared with current ones. Bumping this
# constant is the whole "invalidate all embeddings" switch; ratings/embeddings.py
# is how stale rows then self-heal.
ENCODER_ID = "dinov3_vitb16_448"

# Gated repo: the operator accepts Meta's DINOv3 licence once on this page and
# provides a read token via the HF_TOKEN environment variable. huggingface_hub
# reads that variable itself, so no code here ever sees the token.
HF_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

ImageTransform = Callable[[Image.Image], torch.Tensor]


class EncoderUnavailableError(RuntimeError):
    """The DINOv3 weights could not be loaded; the message says what to do."""


class _PooledEncoder(torch.nn.Module):
    """
    Adapter that gives the Hugging Face model the shape of the old torch.hub
    one: a Module whose forward takes a (B, 3, H, W) batch and returns the
    (B, 768) CLS embedding. encode() and every test double were written against
    that contract, so the adapter keeps the encoder swap contained to this file.
    pooler_output is the CLS token after the final LayerNorm, which is exactly
    what Meta's own hub model returns in eval mode (forward -> x_norm_clstoken).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output


def _load_from_hub(loader, model_id: str):
    """
    Translate the ways a gated download fails into one actionable message.

    transformers surfaces "no token / licence not accepted" and plain network
    failures alike as OSError with a multi-paragraph text (huggingface_hub's
    own HTTP errors are OSError subclasses too). The scrape and train jobs show
    str(exc) in the UI, so what the operator needs is a short line that names
    HF_TOKEN and the licence page (contract V9), not the stack of hub internals.
    """
    try:
        return loader.from_pretrained(model_id)
    except OSError as exc:
        text = str(exc)
        looks_gated = "gated" in text.lower() or "401" in text or "403" in text
        if looks_gated:
            raise EncoderUnavailableError(
                f"{model_id} is a gated model: accept the licence on "
                f"https://huggingface.co/{model_id} and set HF_TOKEN to a read token."
            ) from exc
        first_line = text.splitlines()[0] if text else repr(exc)
        raise EncoderUnavailableError(
            f"Could not load {model_id} (network or cache problem): {first_line}"
        ) from exc


def get_transform() -> ImageTransform:
    """
    The image processor shipped with the checkpoint encodes exactly the
    preprocessing DINOv3 was evaluated with (resize geometry, ImageNet
    mean/std). Re-implementing it as a torchvision pipeline would be a second
    source of truth that drifts silently and produces out-of-distribution
    inputs, which degrades dedup precision and classifier accuracy. So the
    processor is wrapped into the PIL -> tensor callable that encode() expects,
    one image at a time; encode() stacks the results into a batch.

    The one thing overridden is the output size: the processor squashes every
    image to a square (no centre crop) and would use 224 from its config, the
    call asks for INPUT_SIZE instead. Squashing stays as it was, because a
    batch needs equal shapes and the published preprocessing squashes too.
    """
    from transformers import AutoImageProcessor

    processor = _load_from_hub(AutoImageProcessor, HF_MODEL_ID)
    size = {"height": INPUT_SIZE, "width": INPUT_SIZE}

    def to_tensor(image: Image.Image) -> torch.Tensor:
        return processor(images=image, return_tensors="pt", size=size)["pixel_values"][0]

    return to_tensor


def get_encoder(device: str | torch.device | None = None) -> torch.nn.Module:
    """
    ViT-B/16 is the quality/speed sweet spot for this workload: ViT-S loses
    too much semantic detail for taste discrimination, ViT-L is too slow for
    scrape-time dedup on a single GPU. The LVD-1689M checkpoint is the web-image
    one (the SAT variant is satellite imagery). eval() is required for
    deterministic outputs: stochastic layers in train mode would produce
    different embeddings for the same image, breaking cosine-similarity dedup.
    Loaded through transformers rather than torch.hub so the model code is
    pinned by the package version instead of fetched from a GitHub branch at
    runtime.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel

    model = _load_from_hub(AutoModel, HF_MODEL_ID)
    model.eval()
    return _PooledEncoder(model).to(device)


def encode(
    encoder: torch.nn.Module,
    image_paths: list[Path],
    transform: ImageTransform | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 16,
    progress_label: str = "encode",
) -> tuple[np.ndarray, list[Path]]:
    """
    Encode images to 768-d DINOv3 embeddings.

    Per-batch progress is logged via loguru so the in-app log viewer shows the
    job is alive — a silent ViT-B/16 forward pass over thousands of images on
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


def load_classifier(path: Path) -> LogisticRegression | None:
    """
    Classifiers (taste and NSFW) are trained separately in trainer.py and
    persisted so scrape runs can auto-classify without retraining. Pickle is
    the standard sklearn serialisation format; there is no safer alternative
    for arbitrary estimators. The caller is responsible for checking that
    path exists before calling — see scraper.vision_config_from_settings().

    Returns None, with a warning, unless the file records that its estimator
    was fitted on the current encoder (contract V5). A classifier is a
    hyperplane in one embedding space and yields meaningless probabilities in
    another, which would silently steer the review queue. None makes every
    caller behave as if no classifier existed, the pre-training state the UI
    already handles. A bare estimator is the pre-DINOv3 file format and carries
    no stamp, so it counts as foreign too.
    """
    import pickle

    from loguru import logger

    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        logger.warning(
            "{}: classifier predates encoder stamping; ignored until the next train run",
            path,
        )
        return None
    trained_on = payload.get("encoder")
    if trained_on != ENCODER_ID:
        logger.warning(
            "{}: classifier was trained on {} but the encoder is {}; "
            "ignored until the next train run",
            path, trained_on, ENCODER_ID,
        )
        return None
    return payload["classifier"]


def save_classifier(classifier: LogisticRegression, path: Path) -> None:
    """
    Writes the fitted estimator to DATA_DIR so the next scrape run can load it
    via load_classifier() without requiring a retraining pass. The estimator is
    wrapped together with the encoder name it was fitted on, so load_classifier
    can refuse to apply it to vectors from another encoder (contract V5). mkdir
    is included here so the function is safe to call before DATA_DIR is fully
    bootstrapped (e.g. during tests with a temp directory).
    """
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"encoder": ENCODER_ID, "classifier": classifier}
    with path.open("wb") as f:
        pickle.dump(payload, f)


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
    Cosine similarity is the right metric here because DINOv3 embeddings lie on
    a hypersphere — angular distance is meaningful, Euclidean distance is not.
    Zero-norm guard prevents division-by-zero on degenerate images (solid-colour
    fills) that produce all-zero activations. Returns shape (M,) when query is
    1-d so is_embedding_duplicate() can call np.max() directly without squeezing.
    """
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
