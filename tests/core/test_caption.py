"""Tests for core.caption: Moondream2-based alt text generation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    with patch.object(caption, "_model", None):
        with patch("core.caption._get_model") as mock_get:
            mock_model = MagicMock()
            mock_model.caption.return_value = {
                "caption": "A scenic landscape.",
            }
            mock_get.return_value = mock_model
            caption._model = mock_model

            result = caption.describe_for_alt(img, max_length=max_len)

    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result) <= max_len
    assert result == "A scenic landscape."


def test_describe_for_alt_fallback_when_empty() -> None:
    """When caption is empty after strip, describe_for_alt returns 'Image'."""
    with patch.object(caption, "_model", None):
        with patch("core.caption._get_model") as mock_get:
            mock_model = MagicMock()
            mock_model.caption.return_value = {"caption": "   \n  "}
            mock_get.return_value = mock_model
            caption._model = mock_model

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
    with patch.object(caption, "_model", None):
        mock_model = MagicMock()
        mock_model.caption.return_value = {"caption": long_text}
        caption._model = mock_model

        result = caption.describe_for_alt(img, max_length=30)

    assert result.endswith("...")
    assert len(result) <= 30


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
) -> tuple[MagicMock, MagicMock, object]:
    """Helper: run _get_model with mocked transformers and device detection.

    Returns (from_pretrained mock, model mock, result).
    """
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with (
        patch("core.caption._detect_device", return_value=device),
        patch("transformers.AutoModelForCausalLM") as mock_auto,
    ):
        if local_fails:
            mock_auto.from_pretrained.side_effect = [OSError("no cache"), mock_model]
        else:
            mock_auto.from_pretrained.return_value = mock_model

        result = caption._get_model()
        return mock_auto.from_pretrained, mock_model, result


def test_get_model_does_not_pass_device_map() -> None:
    """_get_model must not pass device_map to from_pretrained."""
    from_pretrained, _, _ = _run_get_model()

    for c in from_pretrained.call_args_list:
        assert "device_map" not in c.kwargs, "device_map should not be passed"


def test_get_model_moves_to_detected_device() -> None:
    """_get_model calls model.to() with the detected device."""
    _, mock_model, _ = _run_get_model(device="cpu")
    mock_model.to.assert_called_once_with("cpu")


def test_get_model_calls_eval() -> None:
    """_get_model puts the model in eval mode."""
    _, mock_model, _ = _run_get_model()
    mock_model.eval.assert_called_once()


def test_get_model_uses_float16_on_cuda() -> None:
    """_get_model uses float16 dtype when running on CUDA."""
    from_pretrained, _, _ = _run_get_model(device="cuda")

    first_call_kwargs = from_pretrained.call_args_list[0].kwargs
    assert first_call_kwargs["torch_dtype"] is torch.float16


def test_get_model_uses_float32_on_cpu() -> None:
    """_get_model uses float32 dtype when running on CPU."""
    from_pretrained, _, _ = _run_get_model(device="cpu")

    first_call_kwargs = from_pretrained.call_args_list[0].kwargs
    assert first_call_kwargs["torch_dtype"] is torch.float32


def test_get_model_falls_back_to_online_on_cache_miss() -> None:
    """When local_files_only fails, _get_model retries without it."""
    from_pretrained, _, result = _run_get_model(local_fails=True)

    assert from_pretrained.call_count == 2
    first_kwargs = from_pretrained.call_args_list[0].kwargs
    second_kwargs = from_pretrained.call_args_list[1].kwargs
    assert first_kwargs.get("local_files_only") is True
    assert "local_files_only" not in second_kwargs
    assert result is not None


def test_get_model_passes_trust_remote_code() -> None:
    """_get_model always passes trust_remote_code=True."""
    from_pretrained, _, _ = _run_get_model()

    for c in from_pretrained.call_args_list:
        assert c.kwargs.get("trust_remote_code") is True
