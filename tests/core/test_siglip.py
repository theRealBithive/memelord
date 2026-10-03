"""core.siglip: the SigLIP2 search encoder, tested with fakes (nothing is downloaded).

Contract (confirmed by the operator; the full list lives with the search module):

V9 Der Such-Encoder ist SigLIP2 ViT-B/16 bei 224 px mit Googles veröffentlichter
   Vorverarbeitung. Suchtexte werden so tokenisiert, wie das Modell trainiert
   wurde (auf 64 Token aufgefüllt). Such-Vektoren sind 768-dimensionale
   float32-Vektoren im selben Speicherformat wie die Taste-Vektoren.
V10 Können die SigLIP2-Gewichte nicht geladen werden, endet der Aufruf mit einer
   einzigen klaren Meldung (EncoderUnavailableError), nie still.
V11 Das Textmodell wird im Webprozess erst bei der ersten Suche geladen und
   danach behalten, nie pro Anfrage neu.

Risks: R1 = padding other than max_length/64 silently degrades embeddings;
R2 = CLS slice instead of pooler_output silently gives wrong vectors.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from core import brain, siglip


@pytest.fixture(autouse=True)
def _clean_text_cache():
    siglip.reset_text_encoder_cache()
    yield
    siglip.reset_text_encoder_cache()


def _non_unit_vector() -> torch.Tensor:
    return torch.arange(1, 769, dtype=torch.float32).reshape(1, 768)


class _FakeTextModel:
    def __init__(self, pooled: torch.Tensor | None = None) -> None:
        self.pooled = _non_unit_vector() if pooled is None else pooled
        self.eval_called = False
        self.forward_kwargs: dict = {}

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **kwargs):
        self.forward_kwargs = kwargs
        return SimpleNamespace(pooler_output=self.pooled)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {
            "input_ids": torch.zeros(1, siglip.TEXT_MAX_TOKENS, dtype=torch.long),
            "attention_mask": torch.ones(1, siglip.TEXT_MAX_TOKENS, dtype=torch.long),
        }


def _hub_loader(model: _FakeTextModel, tokenizer: _FakeTokenizer):
    """A _load_from_hub double that hands out the text model or the tokenizer by class name."""
    def load(loader, model_id):
        assert model_id == siglip.SEARCH_MODEL_ID
        if loader.__name__ == "SiglipTextModel":
            return model
        assert loader.__name__ == "AutoTokenizer"
        return tokenizer

    return mock.Mock(side_effect=load)


def _patched_hub(model=None, tokenizer=None):
    model = model if model is not None else _FakeTextModel()
    tokenizer = tokenizer if tokenizer is not None else _FakeTokenizer()
    loader = _hub_loader(model, tokenizer)
    return mock.patch.object(siglip, "_load_from_hub", loader), loader, model, tokenizer


# --- image tower ----------------------------------------------------------------


def test_adapter_returns_pooler_output_not_the_first_patch() -> None:
    """Contract: V9 (Risk R2: SigLIP has no CLS token, position 0 is a patch)"""
    pooled = torch.full((2, 768), 0.25)
    hidden = torch.arange(2 * 5 * 768, dtype=torch.float32).reshape(2, 5, 768)
    wrapped = mock.Mock(
        return_value=SimpleNamespace(pooler_output=pooled, last_hidden_state=hidden)
    )

    batch = torch.zeros(2, 3, 224, 224)
    out = siglip._SiglipImageEncoder(wrapped)(batch)

    assert tuple(out.shape) == (2, 768)
    assert torch.equal(out, pooled)
    assert not torch.equal(out, hidden[:, 0])
    wrapped.assert_called_once()
    assert wrapped.call_args.kwargs["pixel_values"] is batch


def test_image_encoder_loads_vision_tower_in_eval_mode_on_cpu() -> None:
    """Contract: V9"""
    fake = mock.MagicMock(spec=["eval", "to", "parameters"])
    with mock.patch.object(siglip, "_load_from_hub", return_value=fake) as load, \
         mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch.object(
             siglip._SiglipImageEncoder, "to", autospec=True, side_effect=lambda self, d: self
         ) as to:
        encoder = siglip.get_image_encoder()

    assert load.call_args.args[0].__name__ == "SiglipVisionModel"
    assert load.call_args.args[1] == siglip.SEARCH_MODEL_ID
    fake.eval.assert_called_once()
    assert isinstance(encoder, siglip._SiglipImageEncoder)
    assert [call.args[1] for call in to.call_args_list] == ["cpu"]
    # The adapter must wrap the tower that was loaded: a batch goes to it and
    # its pooled output comes back.
    batch = torch.zeros(1, 3, 224, 224)
    assert encoder(batch) is fake.return_value.pooler_output
    assert fake.call_args.kwargs["pixel_values"] is batch


def test_image_encoder_prefers_cuda_when_available_and_honours_explicit_device() -> None:
    """Contract: V9 (device selection is part of loading the encoder)"""
    with mock.patch.object(siglip, "_load_from_hub", return_value=mock.MagicMock()), \
         mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch.object(
             siglip._SiglipImageEncoder, "to", autospec=True, side_effect=lambda self, d: self
         ) as to:
        siglip.get_image_encoder()
        siglip.get_image_encoder("cpu")

    assert [call.args[1] for call in to.call_args_list] == ["cuda", "cpu"]


def test_image_encoder_surfaces_unavailable_error() -> None:
    """Contract: V10"""
    failure = brain.EncoderUnavailableError("could not load")
    with mock.patch.object(siglip, "_load_from_hub", side_effect=failure):
        with pytest.raises(brain.EncoderUnavailableError):
            siglip.get_image_encoder("cpu")


def test_image_transform_is_googles_processor_applied_per_image() -> None:
    """Contract: V9"""
    processor = mock.Mock(return_value={"pixel_values": torch.zeros(1, 3, 224, 224)})
    with mock.patch.object(siglip, "_load_from_hub", return_value=processor) as load:
        transform = siglip.get_image_transform()

    image = Image.new("RGB", (100, 60), color="red")
    tensor = transform(image)

    assert load.call_args.args[0].__name__ == "AutoImageProcessor"
    assert load.call_args.args[1] == siglip.SEARCH_MODEL_ID
    assert processor.call_args.kwargs["images"] is image
    assert processor.call_args.kwargs["return_tensors"] == "pt"
    assert tuple(tensor.shape) == (3, 224, 224)


def test_search_constants_name_siglip2_b16_224() -> None:
    """Contract: V9"""
    assert siglip.SEARCH_ENCODER_ID == "siglip2_b16_224"
    assert siglip.SEARCH_MODEL_ID == "google/siglip2-base-patch16-224"
    assert siglip.SEARCH_DIM == 768
    assert siglip.TEXT_MAX_TOKENS == 64


# --- text tower -----------------------------------------------------------------


def test_text_is_tokenised_padded_to_64_tokens() -> None:
    """Contract: V9 (Risk R1: dynamic padding silently degrades embeddings)"""
    patch, _, _, tokenizer = _patched_hub()
    with patch:
        siglip.encode_text("a red square")

    args, kwargs = tokenizer.calls[0]
    assert args == (["a red square"],)
    assert kwargs == {
        "padding": "max_length",
        "max_length": 64,
        "truncation": True,
        "return_tensors": "pt",
    }


def test_text_model_receives_only_input_ids() -> None:
    """Contract: V9 (training used no attention mask)"""
    patch, _, model, _ = _patched_hub()
    with patch:
        siglip.encode_text("x")

    assert list(model.forward_kwargs) == ["input_ids"]
    # ... and those are the tokenizer's padded ids, not a placeholder.
    padded = torch.zeros(1, siglip.TEXT_MAX_TOKENS, dtype=torch.long)
    assert torch.equal(model.forward_kwargs["input_ids"], padded)


def test_text_vector_is_768_float32_with_unit_norm() -> None:
    """Contract: V9"""
    patch, _, _, _ = _patched_hub()
    with patch:
        vector = siglip.encode_text("a red square")

    assert vector.shape == (768,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
    expected = _non_unit_vector()[0].numpy()
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(vector, expected, rtol=1e-5)


def test_zero_output_stays_zero_without_nan() -> None:
    """Contract: V9 (a degenerate output must not poison the search with NaN)"""
    patch, _, _, _ = _patched_hub(model=_FakeTextModel(pooled=torch.zeros(1, 768)))
    with patch:
        vector = siglip.encode_text("anything")

    assert vector.shape == (768,)
    assert not np.isnan(vector).any()
    assert not vector.any()


def test_same_text_gives_identical_vectors() -> None:
    """Contract: V9 (determinism; the search module's ranking depends on it)"""
    patch, _, _, _ = _patched_hub()
    with patch:
        first = siglip.encode_text("ein rotes Quadrat")
        second = siglip.encode_text("ein rotes Quadrat")

    np.testing.assert_array_equal(first, second)


def test_text_encoder_is_loaded_once_and_reloaded_after_reset() -> None:
    """Contract: V11"""
    patch, loader, model, tokenizer = _patched_hub()
    with patch:
        first = siglip.get_text_encoder()
        siglip.get_text_encoder()
        siglip.get_text_encoder()
        siglip.encode_text("one")
        siglip.encode_text("two")

        assert loader.call_count == 2
        loaded_classes = sorted(call.args[0].__name__ for call in loader.call_args_list)
        assert loaded_classes == ["AutoTokenizer", "SiglipTextModel"]
        assert first == (model, tokenizer)
        assert model.eval_called

        siglip.reset_text_encoder_cache()
        siglip.get_text_encoder()

        assert loader.call_count == 4


def test_text_encoder_surfaces_unavailable_error_and_does_not_cache_failure() -> None:
    """Contract: V10 (and V11: a failed load must be retried, not remembered)"""
    failure = brain.EncoderUnavailableError("could not load")
    with mock.patch.object(siglip, "_load_from_hub", side_effect=failure):
        with pytest.raises(brain.EncoderUnavailableError):
            siglip.encode_text("x")

    patch, loader, _, _ = _patched_hub()
    with patch:
        siglip.encode_text("x")
    assert loader.call_count == 2


@given(text=st.text(min_size=1, max_size=300), zero_output=st.booleans())
def test_text_vector_is_always_finite_768_float32_unit_or_zero(
    text: str, zero_output: bool
) -> None:
    """Contract: V9

    The generator reaches short strings, strings beyond 200 characters (the
    truncation path of a real tokenizer), arbitrary unicode and whitespace-only
    text; zero_output switches between the unit-norm branch and the zero-vector
    branch, so both are asserted without a guard on the implementation.
    """
    pooled = torch.zeros(1, 768) if zero_output else _non_unit_vector()
    patch, _, _, _ = _patched_hub(model=_FakeTextModel(pooled=pooled))
    with patch:
        vector = siglip.encode_text(text)
    siglip.reset_text_encoder_cache()

    assert vector.shape == (768,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    expected_norm = 0.0 if zero_output else 1.0
    assert abs(float(np.linalg.norm(vector)) - expected_norm) < 1e-5


# --- import hygiene ---------------------------------------------------------------


def test_importing_the_module_does_not_import_transformers() -> None:
    """Contract: V11 (nothing heavy happens before the first search)"""
    import subprocess

    code = "import sys, core.siglip; print('transformers' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
