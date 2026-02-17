"""Local image captioning for alt text using SmolVLM2."""

import os
from pathlib import Path

import torch

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
ALT_TEXT_PROMPT = "Describe this image in one sentence for alt text."


def _detect_device() -> str:
    """Return the best available torch device name."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_model():
    """Lazy-load SmolVLM2 model and processor (singleton). Uses cache when available."""
    import logging

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.utils import logging as tf_logging

    tf_logging.set_verbosity_error()
    prev = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    device = _detect_device()
    dtype = torch.float32
    attn = "eager"
    try:
        try:
            processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                local_files_only=True,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                _attn_implementation=attn,
                local_files_only=True,
            )
        except (OSError, ValueError):
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                _attn_implementation=attn,
            )
        model = model.to(device)
        model.eval()
        return model, processor
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev
        tf_logging.set_verbosity(logging.WARNING)


_model = None
_processor = None


def describe_for_alt(
    image_path: Path | str,
    max_length: int = 125,
) -> str:
    """
    Generate a short description of the image for use as alt text.

    Uses SmolVLM2 (HuggingFaceTB/SmolVLM2-500M-Video-Instruct) locally.
    Output is trimmed to max_length characters to suit Mastodon's alt field.

    Args:
        image_path: Path to the image file.
        max_length: Max character length of the returned string. Default 125.

    Returns:
        A single-sentence description, truncated to max_length.
    """
    global _model, _processor
    if _model is None:
        _model, _processor = _get_model()

    path = Path(image_path)
    # Ensure path is absolute so the processor can load the image
    path = path.resolve()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(path)},
                {"type": "text", "text": ALT_TEXT_PROMPT},
            ],
        },
    ]
    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_model.device, dtype=torch.float32)

    generated_ids = _model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=128,
    )
    generated_texts = _processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    caption = (generated_texts[0] or "").strip()
    if len(caption) > max_length:
        caption = caption[: max_length - 3].rsplit(" ", 1)[0] + "..."
    return caption or "Image"
