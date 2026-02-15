"""Local image captioning for alt text using BLIP."""

from pathlib import Path

import torch
from PIL import Image

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"


def _get_model():
    """Lazy-load BLIP processor and model (singleton)."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    model.eval()
    return processor, model


_processor = None
_model = None


def describe_for_alt(
    image_path: Path | str,
    max_length: int = 125,
) -> str:
    """
    Generate a short description of the image for use as alt text.

    Uses BLIP (Salesforce/blip-image-captioning-base) locally. Output is
    trimmed to max_length characters to suit Mastodon's alt field.

    Args:
        image_path: Path to the image file.
        max_length: Maximum character length of the returned string. Default 125.

    Returns:
        A single-sentence description, truncated to max_length.
    """
    global _processor, _model
    if _processor is None or _model is None:
        _processor, _model = _get_model()

    path = Path(image_path)
    img = Image.open(path).convert("RGB")

    inputs = _processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=50)

    caption = _processor.decode(out[0], skip_special_tokens=True).strip()
    if len(caption) > max_length:
        caption = caption[: max_length - 3].rsplit(" ", 1)[0] + "..."
    return caption or "Image"
