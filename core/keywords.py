"""Florence-2 keyword suggestion for images via the native <OD> task."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/Florence-2-base"

_model = None
_processor = None


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_model():
    """
    Lazy-load Florence-2-base with trust_remote_code (required — Florence-2 ships
    custom modelling code). local_files_only is attempted first so offline installs
    don't stall on a network probe; falls back to a HuggingFace download if the
    model isn't cached yet. float32 on CPU; bfloat16 is unsupported on most CPUs
    and produces garbage outputs.
    """
    import logging

    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.utils import logging as tf_logging

    tf_logging.set_verbosity_error()
    device = _detect_device()
    try:
        try:
            processor = AutoProcessor.from_pretrained(
                MODEL_ID, trust_remote_code=True, local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError):
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        model = model.to(device).eval()
        return model, processor
    finally:
        tf_logging.set_verbosity(logging.WARNING)


def suggest_keywords(image_path: Path | str, max_keywords: int = 8) -> list[str]:
    """
    Return up to max_keywords tag suggestions using Florence-2's <OD> task.

    <OD> (object detection) returns labels for objects found in the image —
    "dog", "person", "tree", etc. There is no native tagging task in Florence-2
    (<GENERATE_TAGS> is not a real task token; the model emits garbage when
    given it), so OD's label set is the closest fit for free-form #tags.

    Returns [] on any error (unreadable image, missing model, etc.) so callers
    can safely skip without try/except.
    """
    global _model, _processor
    try:
        if _model is None:
            _model, _processor = _get_model()

        from PIL import Image as PILImage

        path = Path(image_path).resolve()
        image = PILImage.open(path).convert("RGB")

        prompt = "<OD>"
        inputs = _processor(text=prompt, images=image, return_tensors="pt").to(
            _model.device, dtype=torch.float32
        )

        generated_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            do_sample=False,
            num_beams=3,
        )
        raw = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = _processor.post_process_generation(
            raw, task=prompt, image_size=(image.width, image.height)
        )
        # post_process_generation for <OD> returns {"<OD>": {"bboxes": [...], "labels": [...]}}.
        labels = parsed.get(prompt, {}).get("labels", []) or []
        tags = [str(l).strip().lower() for l in labels if str(l).strip()]

        seen: set[str] = set()
        unique: list[str] = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique[:max_keywords]
    except Exception as e:
        # Logged (not silently swallowed) so future breakage — like the
        # transformers 5.x / Florence-2 forced_bos_token_id incompatibility —
        # surfaces in the log viewer instead of just producing empty UI.
        logger.warning("suggest_keywords(%s) failed: %s", image_path, e, exc_info=True)
        return []
