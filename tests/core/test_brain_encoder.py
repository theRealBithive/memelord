"""core.brain: encoder loading and classifier files.

Contract (confirmed by the operator on 2026-10-02; the full list V1–V9 lives in
tests/ratings/test_embedding_generation.py):

V5 Ein Taste- oder NSFW-Klassifikator merkt sich den Encoder, auf dem er
   trainiert wurde, und wird nur auf Embeddings dieses Encoders angewendet.
   Eine Klassifikator-Datei ohne diese Angabe gilt als fremd und wird nicht
   angewendet.
V8 Der neue Encoder ist DINOv3 ViT-B/16 mit Metas veröffentlichter
   Vorverarbeitung. Embeddings bleiben 768-dimensionale float32-Vektoren, das
   Speicherformat ändert sich nicht.
V9 Können die Modellgewichte nicht bezogen werden, scheitert der Scrape- oder
   Trainingslauf mit einer einzigen klaren Meldung, die sagt, was zu tun ist.
   Er läuft nie still ohne Embeddings weiter.
"""

import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image
from sklearn.linear_model import LogisticRegression

from core import brain


class _FakeHFModel:
    """Stands in for the transformers model: records eval(), returns a CLS batch."""

    def __init__(self) -> None:
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, pixel_values: torch.Tensor):
        batch = pixel_values.shape[0]
        return SimpleNamespace(pooler_output=torch.full((batch, 768), 0.5))


def _fitted_classifier() -> LogisticRegression:
    X = np.vstack([np.full(768, 0.1), np.full(768, 0.9)])
    return LogisticRegression().fit(X, [0, 1])


# --- V8 -----------------------------------------------------------------------


def test_encoder_id_and_dimension_name_dinov3_vitb16_at_448() -> None:
    """Contract: V8; taste V18 (the stamp names the resolution, so every 224 px vector is stale)"""
    assert brain.ENCODER_ID == "dinov3_vitb16_448"
    assert brain.INPUT_SIZE == 448
    assert brain.HF_MODEL_ID == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert brain.EMBEDDING_DIM == 768


def test_encoder_returns_one_cls_vector_per_image_in_eval_mode() -> None:
    """Contract: V8"""
    fake = _FakeHFModel()
    with mock.patch("transformers.AutoModel") as auto_model:
        auto_model.from_pretrained.return_value = fake
        encoder = brain.get_encoder("cpu")

    out = encoder(torch.zeros(2, 3, brain.INPUT_SIZE, brain.INPUT_SIZE))

    auto_model.from_pretrained.assert_called_once_with(brain.HF_MODEL_ID)
    assert fake.eval_called
    assert tuple(out.shape) == (2, brain.EMBEDDING_DIM)


def test_transform_is_metas_processor_applied_per_image_at_448() -> None:
    """Contract: V8; taste V18

    _load_from_hub is patched rather than the transformers module attribute:
    transformers resolves AutoImageProcessor lazily, so a module-level patch is
    not what `from transformers import AutoImageProcessor` picks up.
    """
    processor = mock.Mock(return_value={"pixel_values": torch.zeros(1, 3, 448, 448)})
    with mock.patch.object(brain, "_load_from_hub", return_value=processor) as load:
        transform = brain.get_transform()

    image = Image.new("RGB", (100, 60), color="red")
    tensor = transform(image)

    load.assert_called_once()
    assert load.call_args.args[1] == brain.HF_MODEL_ID
    assert load.call_args.args[0].__name__ == "AutoImageProcessor"
    processor.assert_called_once()
    assert processor.call_args.kwargs["images"] is image
    assert processor.call_args.kwargs["return_tensors"] == "pt"
    assert processor.call_args.kwargs["size"] == {"height": 448, "width": 448}
    assert tuple(tensor.shape) == (3, 448, 448)


def test_storage_format_round_trips_768_float32() -> None:
    """Contract: V8"""
    vector = np.arange(768, dtype=np.float32) / 768
    blob = brain.embedding_to_bytes(vector)
    assert len(blob) == 768 * 4
    np.testing.assert_array_equal(brain.bytes_to_embedding(blob), vector)


# --- V9 -----------------------------------------------------------------------


def test_gated_repo_failure_names_token_and_licence_page() -> None:
    """Contract: V9"""
    loader = SimpleNamespace(
        from_pretrained=mock.Mock(
            side_effect=OSError(
                "You are trying to access a gated repo.\nMake sure to have access..."
            )
        )
    )
    with pytest.raises(brain.EncoderUnavailableError) as excinfo:
        brain._load_from_hub(loader, brain.HF_MODEL_ID)
    loader.from_pretrained.assert_called_once_with(brain.HF_MODEL_ID)
    message = str(excinfo.value)
    assert "HF_TOKEN" in message
    assert f"https://huggingface.co/{brain.HF_MODEL_ID}" in message
    assert "\n" not in message


def test_network_failure_is_one_line_with_cause() -> None:
    """Contract: V9"""
    original = OSError("Connection error: offline\nTraceback internals follow")
    loader = SimpleNamespace(from_pretrained=mock.Mock(side_effect=original))
    with pytest.raises(brain.EncoderUnavailableError) as excinfo:
        brain._load_from_hub(loader, brain.HF_MODEL_ID)
    message = str(excinfo.value)
    assert message == (
        f"Could not load {brain.HF_MODEL_ID} (network or cache problem): "
        "Connection error: offline"
    )
    assert excinfo.value.__cause__ is original


def test_network_failure_without_text_falls_back_to_the_exception_repr() -> None:
    """Contract: V9 (an empty OSError must still leave a visible cause)"""
    loader = SimpleNamespace(from_pretrained=mock.Mock(side_effect=OSError()))
    with pytest.raises(brain.EncoderUnavailableError) as excinfo:
        brain._load_from_hub(loader, brain.HF_MODEL_ID)
    assert str(excinfo.value).endswith("OSError()")


@given(
    gated_word=st.sampled_from(["", "gated", "Gated", "GATED"]),
    has_401=st.booleans(),
    has_403=st.booleans(),
    noise=st.text(alphabet="bcefhijklmnopqrsuvwxyz \n", max_size=40),
)
def test_gated_detection_depends_only_on_the_three_hub_signals(
    gated_word: str, has_401: bool, has_403: bool, noise: str
) -> None:
    """Contract: V9

    The hub reports a missing licence as "gated" text or as HTTP 401/403, so
    exactly those three signals must route to the licence message; anything
    else is a network/cache problem and must quote its first line instead.
    The noise alphabet contains no "g" and no digits, so it cannot fake a
    signal.
    """
    parts = [noise, gated_word, "401" if has_401 else "", noise, "403" if has_403 else ""]
    text = " ".join(part for part in parts if part)
    loader = SimpleNamespace(from_pretrained=mock.Mock(side_effect=OSError(text)))

    with pytest.raises(brain.EncoderUnavailableError) as excinfo:
        brain._load_from_hub(loader, brain.HF_MODEL_ID)

    message = str(excinfo.value)
    expect_licence_message = bool(gated_word) or has_401 or has_403
    if expect_licence_message:
        assert message.startswith(f"{brain.HF_MODEL_ID} is a gated model")
    else:
        assert message.startswith(f"Could not load {brain.HF_MODEL_ID}")


def test_encoder_failure_is_a_runtime_error_so_jobs_fail_loudly() -> None:
    """Contract: V9 (tasks.run_scrape/run_train report str(exc) to the UI)"""
    assert issubclass(brain.EncoderUnavailableError, RuntimeError)


def test_get_encoder_propagates_unavailable_error() -> None:
    """Contract: V9"""
    with mock.patch("transformers.AutoModel") as auto_model:
        auto_model.from_pretrained.side_effect = OSError("403 Client Error")
        with pytest.raises(brain.EncoderUnavailableError):
            brain.get_encoder("cpu")


# --- V5 -----------------------------------------------------------------------


def test_saved_classifier_is_stamped_with_current_encoder(tmp_path: Path) -> None:
    """Contract: V5"""
    path = tmp_path / "w.pkl"
    brain.save_classifier(_fitted_classifier(), path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    assert payload["encoder"] == brain.ENCODER_ID
    assert isinstance(payload["classifier"], LogisticRegression)
    assert isinstance(brain.load_classifier(path), LogisticRegression)


def test_legacy_bare_estimator_file_is_not_applied(tmp_path: Path) -> None:
    """Contract: V5"""
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as f:
        pickle.dump(_fitted_classifier(), f)
    assert brain.load_classifier(path) is None


def test_classifier_from_other_encoder_is_not_applied(tmp_path: Path) -> None:
    """Contract: V5"""
    path = tmp_path / "foreign.pkl"
    with path.open("wb") as f:
        pickle.dump({"encoder": "dinov2_vitb14", "classifier": _fitted_classifier()}, f)
    assert brain.load_classifier(path) is None


@given(
    encoder=st.one_of(
        st.just(brain.ENCODER_ID),
        st.text(min_size=0, max_size=40),
        st.none(),
    )
)
def test_classifier_applies_iff_trained_on_current_encoder(encoder) -> None:
    """Contract: V5 (property over the stamp: exactly one value unlocks the file).

    The generator reaches both states: the current id directly, and arbitrary
    text (which only by chance equals the current id), plus a missing stamp.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.pkl"
        with path.open("wb") as f:
            pickle.dump({"encoder": encoder, "classifier": _fitted_classifier()}, f)
        loaded = brain.load_classifier(path)
    if encoder == brain.ENCODER_ID:
        assert isinstance(loaded, LogisticRegression)
    else:
        assert loaded is None


def test_encoder_picks_cpu_when_no_cuda_is_available() -> None:
    """Contract: V8 (device selection is part of loading the encoder)"""
    fake = _FakeHFModel()
    with mock.patch("transformers.AutoModel") as auto_model, \
         mock.patch("torch.cuda.is_available", return_value=False):
        auto_model.from_pretrained.return_value = fake
        encoder = brain.get_encoder()
    out = encoder(torch.zeros(1, 3, brain.INPUT_SIZE, brain.INPUT_SIZE))
    assert out.device.type == "cpu"
    assert tuple(out.shape) == (1, brain.EMBEDDING_DIM)


def test_explicit_device_wins_over_cuda_autodetection() -> None:
    """Contract: V8 (device selection is part of loading the encoder)

    A caller's device must be honoured even when CUDA exists, and CUDA must be
    chosen when the caller leaves it open. `.to` is intercepted because this
    machine has no GPU, so the requested device is the only observable.
    """
    fake = _FakeHFModel()
    with mock.patch("transformers.AutoModel") as auto_model, \
         mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch.object(
             brain._PooledEncoder, "to", autospec=True, side_effect=lambda self, device: self
         ) as to:
        auto_model.from_pretrained.return_value = fake
        brain.get_encoder("cpu")
        brain.get_encoder()

    requested_devices = [call.args[1] for call in to.call_args_list]
    assert requested_devices == ["cpu", "cuda"]


def test_encode_uses_the_hub_processor_when_no_transform_is_given(tmp_path: Path) -> None:
    """Contract: V8 (the default preprocessing is Meta's processor, not an ad-hoc pipeline)"""
    Image.new("RGB", (6, 6), color="blue").save(tmp_path / "img.png")
    calls = {"n": 0}

    def processor_transform(image):
        calls["n"] += 1
        return torch.zeros(3, 8, 8)

    class MockEncoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.size(0), 768)

    with mock.patch.object(brain, "get_transform", return_value=processor_transform) as get_transform:
        result, valid = brain.encode(MockEncoder(), [tmp_path / "img.png"])

    get_transform.assert_called_once()
    assert calls["n"] == 1
    assert result.shape == (1, 768)
    assert valid == [tmp_path / "img.png"]


def test_autodetection_asks_for_cpu_when_cuda_is_absent() -> None:
    """Contract: V8 (device selection is part of loading the encoder)

    Complements test_explicit_device_wins_over_cuda_autodetection: without a
    GPU the loader must request "cpu". `.to` is intercepted because a Module
    without real tensors would silently accept any device string.
    """
    fake = _FakeHFModel()
    with mock.patch("transformers.AutoModel") as auto_model, \
         mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch.object(
             brain._PooledEncoder, "to", autospec=True, side_effect=lambda self, device: self
         ) as to:
        auto_model.from_pretrained.return_value = fake
        brain.get_encoder()

    assert [call.args[1] for call in to.call_args_list] == ["cpu"]
