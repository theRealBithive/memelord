"""
The search encoder: SigLIP2 ViT-B/16 at 224 px, for text-to-image search.

Why a second encoder next to DINOv3 (core/brain.py): DINOv3 is image-only, so
it cannot embed a query string. SigLIP2 puts images and text into one space,
which is what makes "a red car" findable.

Why SigLIP2 and not CLIP: it is multilingual, so German queries work, and its
retrieval quality is better than CLIP ViT-B/32.

Why base/16 at 224: it costs the same per image as the DINOv3 ViT-B/16 that
every image already pays for (about 2 img/s on a CPU), so the extra encoder pass
does not change how long a scrape takes by an order of magnitude.

Why the fixed-resolution checkpoint: its model_type is "siglip", so the mature
Siglip* classes in transformers apply. The NaFlex variants are "siglip2" and
need a different, patch-sequence input pipeline.

This module has no Django imports and imports transformers lazily, so merely
importing it is cheap and loads no model.
"""

import threading

import numpy as np
import torch
from PIL import Image

from core import brain

# _load_from_hub is imported by its underscore name on purpose: it turns gated
# and network OSErrors into brain.EncoderUnavailableError with an actionable
# message. Renaming it in brain.py would trigger a 35-minute mutation run for a
# cosmetic change, so it stays private there and is reused here.
from core.brain import _load_from_hub

# Stamp written to Image.search_embedding_model, the counterpart of
# brain.ENCODER_ID for the search vectors.
SEARCH_ENCODER_ID = "siglip2_b16_224"
SEARCH_MODEL_ID = "google/siglip2-base-patch16-224"
SEARCH_DIM = 768
TEXT_MAX_TOKENS = 64

_text_encoder_lock = threading.Lock()
_cached_text_encoder: tuple[torch.nn.Module, object] | None = None


class _SiglipImageEncoder(torch.nn.Module):
    """
    Adapter that gives the SigLIP vision tower the contract brain.encode()
    expects: forward takes a (B, 3, H, W) batch and returns (B, 768). That way
    batching, the progress log and the skipping of unreadable files are reused
    unchanged.

    The output is pooler_output, which for SigLIP is the attention-pooling head
    and exactly what SiglipModel.get_image_features returns, i.e. the vector that
    lives in the same space as the text vectors. last_hidden_state[:, 0] would be
    WRONG: SigLIP has no CLS token, so position 0 is just the first image patch.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output


def get_image_encoder(device: str | torch.device | None = None) -> torch.nn.Module:
    """
    Only the vision tower is loaded: the text tower is ~1.1 GB because of the
    256k-token vocabulary and is useless for encoding images. eval() is required
    so the same image always gets the same vector.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import SiglipVisionModel

    model = _load_from_hub(SiglipVisionModel, SEARCH_MODEL_ID)
    model.eval()
    return _SiglipImageEncoder(model).to(device)


def get_image_transform() -> brain.ImageTransform:
    """
    Google's published preprocessing (resize to 224, mean/std 0.5) comes from the
    checkpoint's own processor. Re-implementing it would be a second source of
    truth that drifts silently and degrades search quality without any error.
    """
    from transformers import AutoImageProcessor

    processor = _load_from_hub(AutoImageProcessor, SEARCH_MODEL_ID)

    def to_tensor(image: Image.Image) -> torch.Tensor:
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    return to_tensor


def get_text_encoder() -> tuple[torch.nn.Module, object]:
    """
    Returns (SiglipTextModel, tokenizer), loaded once and then kept (contract
    V11: the web process loads this on the first search only, never per
    request). Only the text tower is loaded because the full checkpoint also
    carries the vision tower, which the web process does not need.

    The model stays on CPU: a single query is one short sequence, and the web
    process must not claim GPU memory that the scrape/train jobs need.

    A plain global plus a lock instead of functools.lru_cache: the Django dev
    server is multi-threaded, and without the lock two first searches would load
    1.1 GB twice. gunicorn sync workers are single-threaded, so there the lock
    is just cheap.
    """
    global _cached_text_encoder
    with _text_encoder_lock:
        if _cached_text_encoder is not None:
            return _cached_text_encoder
        from transformers import AutoTokenizer, SiglipTextModel

        model = _load_from_hub(SiglipTextModel, SEARCH_MODEL_ID)
        model.eval()
        tokenizer = _load_from_hub(AutoTokenizer, SEARCH_MODEL_ID)
        _cached_text_encoder = (model, tokenizer)
        return _cached_text_encoder


def encode_text(text: str) -> np.ndarray:
    """
    Returns a (768,) float32 vector with unit L2 norm, comparable by dot product
    with the L2-normalised image vectors.

    padding="max_length" with 64 tokens is mandatory: SigLIP was trained on
    fixed, 64-token padded inputs. With dynamic padding the model does not crash,
    it just produces degraded embeddings, which is a silent quality bug that no
    error message would reveal. Only input_ids are passed on, because training
    used no attention mask either.

    A zero vector is left as it is instead of divided by its norm, so the result
    is never NaN.
    """
    model, tokenizer = get_text_encoder()
    tokens = tokenizer(
        [text],
        padding="max_length",
        max_length=TEXT_MAX_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(input_ids=tokens["input_ids"])
    vector = output.pooler_output[0].cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def reset_text_encoder_cache() -> None:
    """Forget the cached text tower. Tests only: production keeps it for the process lifetime."""
    global _cached_text_encoder
    with _text_encoder_lock:
        _cached_text_encoder = None
