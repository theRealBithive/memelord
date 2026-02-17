"""Local image captioning for alt text using Moondream2."""

import os
from pathlib import Path

import torch
from PIL import Image

MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-06-21"


def _detect_device() -> str:
    """Return the best available torch device name."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_model():
    """Lazy-load Moondream2 model (singleton). Uses cache when available."""
    import logging

    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as tf_logging

    tf_logging.set_verbosity_error()
    prev = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    device = _detect_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MOONDREAM_MODEL_ID,
                revision=MOONDREAM_REVISION,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        except (OSError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(
                MOONDREAM_MODEL_ID,
                revision=MOONDREAM_REVISION,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        model = model.to(device)
        model.eval()
        return model
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev
        tf_logging.set_verbosity(logging.WARNING)


_model = None


def describe_for_alt(
    image_path: Path | str,
    max_length: int = 125,
) -> str:
    """
    Generate a short description of the image for use as alt text.

    Uses Moondream2 (vikhyatk/moondream2) locally. Output is trimmed to
    max_length characters to suit Mastodon's alt field.

    Args:
        image_path: Path to the image file.
        max_length: Max character length of the returned string. Default 125.

    Returns:
        A single-sentence description, truncated to max_length.
    """
    global _model
    if _model is None:
        _model = _get_model()

    path = Path(image_path)
    img = Image.open(path).convert("RGB")

    result = _model.caption(img, length="short")
    caption = (result.get("caption") or "").strip()
    if len(caption) > max_length:
        caption = caption[: max_length - 3].rsplit(" ", 1)[0] + "..."
    return caption or "Image"
