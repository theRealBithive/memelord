"""Tests for core.caption: SmolVLM2-based alt text generation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from core import caption
from tests.conftest import minimal_png_bytes


# ---------------------------------------------------------------------------
# describe_for_alt
# ---------------------------------------------------------------------------


def test_describe_for_alt_returns_string_within_max_length(tmp_path: Path) -> None:
    """describe_for_alt returns a non-empty string of at most max_length chars."""
    img = tmp_path / "tiny.png"
    img.write_bytes(minimal_png_bytes())
    max_len = 125
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["A scenic landscape."]
    with (
        patch.object(caption, "_model", None),
        patch.object(caption, "_processor", None),
    ):
        with patch("core.caption._get_model") as mock_get:
            mock_get.return_value = (mock_model, mock_processor)
            result = caption.describe_for_alt(img, max_length=max_len)
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result) <= max_len
    assert result == "A scenic landscape."


def test_describe_for_alt_fallback_when_empty() -> None:
    """When caption is empty after strip, describe_for_alt returns 'Image'."""
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["   \n  "]
    with (
        patch.object(caption, "_model", None),
        patch.object(caption, "_processor", None),
    ):
        with patch("core.caption._get_model") as mock_get:
            mock_get.return_value = (mock_model, mock_processor)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(minimal_png_bytes())
                p = Path(f.name)
            try:
                out = caption.describe_for_alt(p, max_length=125)
            finally:
                p.unlink(missing_ok=True)
    assert out == "Image"


def test_describe_for_alt_truncates_long_caption(tmp_path: Path) -> None:
    """Captions exceeding max_length are truncated at a word boundary with '...'."""
    img = tmp_path / "tiny.png"
    img.write_bytes(minimal_png_bytes())
    long_text = "A " + "word " * 40
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = MagicMock()
    mock_processor.batch_decode.return_value = [long_text]
    with (
        patch.object(caption, "_model", None),
        patch.object(caption, "_processor", None),
    ):
        with patch("core.caption._get_model") as mock_get:
            mock_get.return_value = (mock_model, mock_processor)
            result = caption.describe_for_alt(img, max_length=30)
    assert result.endswith("...")
    assert len(result) <= 30


def test_describe_for_alt_decodes_only_new_tokens(tmp_path: Path) -> None:
    """batch_decode is called with only the newly generated tokens, not the full prompt."""
    img = tmp_path / "tiny.png"
    img.write_bytes(minimal_png_bytes())
    input_length = 100
    num_new_tokens = 20
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs["input_ids"].shape = (1, input_length)
    mock_processor.apply_chat_template.return_value.to.return_value = mock_inputs
    mock_model.generate.return_value = torch.zeros(
        1, input_length + num_new_tokens, dtype=torch.long
    )
    mock_processor.batch_decode.return_value = ["A scenic landscape."]
    with (
        patch.object(caption, "_model", None),
        patch.object(caption, "_processor", None),
    ):
        with patch("core.caption._get_model") as mock_get:
            mock_get.return_value = (mock_model, mock_processor)
            caption.describe_for_alt(img, max_length=125)
    mock_processor.batch_decode.assert_called_once()
    (decode_arg,) = mock_processor.batch_decode.call_args[0]
    assert decode_arg.shape == (1, num_new_tokens)


# ---------------------------------------------------------------------------
# _detect_device
# ---------------------------------------------------------------------------


def test_detect_device_returns_cuda_when_available() -> None:
    """_detect_device returns 'cuda' when CUDA is available."""
    with patch("core.caption.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        assert caption._detect_device() == "cuda"


def test_detect_device_returns_cpu_when_no_cuda() -> None:
    """_detect_device falls back to 'cpu' when CUDA is not available."""
    with patch("core.caption.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        assert caption._detect_device() == "cpu"


# ---------------------------------------------------------------------------
# _get_model
# ---------------------------------------------------------------------------


def _run_get_model(
    *,
    device: str = "cpu",
    local_fails: bool = False,
) -> tuple[MagicMock, MagicMock, MagicMock, object]:
    """Helper: run _get_model with mocked transformers and device detection.

    Returns (model from_pretrained mock, processor from_pretrained mock,
             model mock, (model, processor) result).
    """
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_processor = MagicMock()

    with (
        patch("core.caption._detect_device", return_value=device),
        patch(
            "transformers.AutoModelForImageTextToText.from_pretrained",
        ) as mock_model_fp,
        patch(
            "transformers.AutoProcessor.from_pretrained",
        ) as mock_processor_fp,
    ):
        if local_fails:
            mock_processor_fp.side_effect = [OSError("no cache"), mock_processor]
            mock_model_fp.return_value = mock_model
        else:
            mock_processor_fp.return_value = mock_processor
            mock_model_fp.return_value = mock_model

        result = caption._get_model()
        return (
            mock_model_fp,
            mock_processor_fp,
            mock_model,
            result,
        )


def test_get_model_returns_model_and_processor() -> None:
    """_get_model returns (model, processor) tuple."""
    _, _, _, result = _run_get_model()
    assert isinstance(result, tuple)
    assert len(result) == 2
    model, processor = result
    assert model is not None
    assert processor is not None


def test_get_model_does_not_pass_device_map() -> None:
    """_get_model must not pass device_map to from_pretrained."""
    model_from_pretrained, _, _, _ = _run_get_model()
    for c in model_from_pretrained.call_args_list:
        assert "device_map" not in c.kwargs, "device_map should not be passed"


def test_get_model_moves_to_detected_device() -> None:
    """_get_model calls model.to() with the detected device."""
    _, _, mock_model, _ = _run_get_model(device="cpu")
    mock_model.to.assert_called_once_with("cpu")


def test_get_model_calls_eval() -> None:
    """_get_model puts the model in eval mode."""
    _, _, mock_model, _ = _run_get_model()
    mock_model.eval.assert_called_once()


def test_get_model_falls_back_to_online_on_cache_miss() -> None:
    """When local_files_only fails for processor, _get_model retries without it."""
    model_fp, proc_fp, _, result = _run_get_model(local_fails=True)
    assert proc_fp.call_count == 2
    first_proc_kw = proc_fp.call_args_list[0].kwargs
    second_proc_kw = proc_fp.call_args_list[1].kwargs
    assert first_proc_kw.get("local_files_only") is True
    assert "local_files_only" not in second_proc_kw
    assert model_fp.call_count == 1
    assert result is not None


def test_get_model_uses_eager_attention() -> None:
    """_get_model uses eager attention for CPU compatibility."""
    model_from_pretrained, _, _, _ = _run_get_model()
    first_call_kwargs = model_from_pretrained.call_args_list[0].kwargs
    assert first_call_kwargs.get("_attn_implementation") == "eager"


# ---------------------------------------------------------------------------
# Integration tests — real model, no mocks
# Run with: pytest --run-integration -m integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smolvlm_model_and_processor():
    """Load the real SmolVLM2 model and processor once for all integration tests."""
    caption._model = None
    caption._processor = None
    model, processor = caption._get_model()
    yield model, processor
    caption._model = None
    caption._processor = None


@pytest.mark.integration
def test_get_model_loads_real_model(smolvlm_model_and_processor) -> None:
    """_get_model loads the real SmolVLM2 model and processor without errors."""
    model, processor = smolvlm_model_and_processor
    assert model is not None
    assert processor is not None
    assert hasattr(processor, "apply_chat_template")
    assert hasattr(model, "generate")


@pytest.mark.integration
def test_describe_for_alt_end_to_end(tmp_path: Path) -> None:
    """Full end-to-end: describe_for_alt returns a valid string."""
    from PIL import Image

    img_path = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(img_path)

    caption._model = None
    caption._processor = None
    try:
        result = caption.describe_for_alt(img_path)
    finally:
        caption._model = None
        caption._processor = None

    assert isinstance(result, str)
    assert len(result) <= 125
    assert result == "Image" or len(result) > 0
