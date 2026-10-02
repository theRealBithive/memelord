"""Tests for core.brain (DINOv3 + classifier)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from core import brain


def test_brain_module_exposes_encoder_interface() -> None:
    """Module exposes the encoder/classifier interface the pipeline relies on."""
    assert callable(brain.get_encoder)
    assert callable(brain.get_transform)
    assert callable(brain.encode)


def test_is_image_path_true_for_image_extensions() -> None:
    """is_image_path returns True for supported extensions."""
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"):
        assert brain.is_image_path(Path(f"x{ext}")) is True


def test_is_image_path_false_for_non_image() -> None:
    """is_image_path returns False for non-image extensions."""
    assert brain.is_image_path(Path("x.txt")) is False
    assert brain.is_image_path(Path("x")) is False


def _dummy_transform(image: Image.Image) -> torch.Tensor:
    """Stand-in for the DINOv3 processor: encode()'s batching does not depend on it.

    The real transform is downloaded from a gated Hugging Face repo, so these
    encode() tests, which only exercise batching and skip logic, inject a
    constant-shape tensor instead.
    """
    return torch.zeros(3, 8, 8)


def test_encode_empty_paths_returns_empty_array() -> None:
    """encode with no paths returns (0, 768) float32 and empty path list."""
    mock_encoder = torch.nn.Linear(3, 768)  # unused, we only check shape
    result, valid = brain.encode(mock_encoder, [])
    assert result.shape == (0, 768)
    assert result.dtype == np.float32
    assert valid == []


def test_encode_returns_shape_n_768(tmp_path: Path) -> None:
    """encode with a mock encoder and one image returns (1, 768) and the path."""
    Image.new("RGB", (224, 224), color="blue").save(tmp_path / "img.png")
    path = tmp_path / "img.png"

    class MockEncoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.size(0), 768, device=x.device, dtype=x.dtype)

    encoder = MockEncoder()
    result, valid = brain.encode(encoder, [path], transform=_dummy_transform)
    assert result.shape == (1, 768)
    assert result.dtype == np.float32
    assert valid == [path]


def test_encode_skips_missing_file(tmp_path: Path) -> None:
    """encode silently skips a path that no longer exists on disk."""
    Image.new("RGB", (4, 4)).save(tmp_path / "real.png")
    real = tmp_path / "real.png"
    missing = tmp_path / "gone.png"

    class MockEncoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.size(0), 768)

    result, valid = brain.encode(MockEncoder(), [real, missing], transform=_dummy_transform)
    assert result.shape == (1, 768)
    assert valid == [real]


def test_save_classifier_load_classifier_roundtrip(tmp_path: Path) -> None:
    """save_classifier and load_classifier roundtrip."""
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(np.random.randn(5, 768), [0, 1, 0, 1, 0])
    path = tmp_path / "weights.pkl"
    brain.save_classifier(clf, path)
    assert path.exists()
    loaded = brain.load_classifier(path)
    assert loaded.predict_proba(np.random.randn(1, 768)).shape == (1, 2)


def test_predict_proba_single_embedding_returns_scalar() -> None:
    """predict_proba with (768,) returns a scalar probability."""
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(np.random.randn(4, 768), [0, 1, 0, 1])
    emb = np.random.randn(768).astype(np.float32)
    proba = brain.predict_proba(clf, emb)
    assert np.isscalar(proba) or proba.shape == ()
    assert 0 <= float(proba) <= 1


def test_predict_proba_batch_returns_array() -> None:
    """predict_proba with (N, 768) returns shape (N,)."""
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(np.random.randn(4, 768), [0, 1, 0, 1])
    emb = np.random.randn(3, 768).astype(np.float32)
    proba = brain.predict_proba(clf, emb)
    assert proba.shape == (3,)
    assert np.all((proba >= 0) & (proba <= 1))


def _garbage_file(path: Path) -> Path:
    path.write_bytes(b"this is not an image")
    return path


def _solid_png(path: Path, mode: str = "RGB") -> Path:
    Image.new(mode, (8, 8)).save(path)
    return path


class _ZeroEncoder(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros(batch.size(0), 768)


def test_encode_yields_exactly_one_vector_per_image_across_batches(tmp_path: Path) -> None:
    paths = [_solid_png(tmp_path / f"{i}.png") for i in range(3)]

    result, valid = brain.encode(_ZeroEncoder(), paths, transform=_dummy_transform, batch_size=2)

    assert result.shape == (3, 768)
    assert valid == paths


def test_encode_continues_after_a_batch_with_only_unreadable_files(tmp_path: Path) -> None:
    first = _solid_png(tmp_path / "first.png")
    broken = _garbage_file(tmp_path / "broken.png")
    last = _solid_png(tmp_path / "last.png")

    result, valid = brain.encode(
        _ZeroEncoder(), [first, broken, last], transform=_dummy_transform, batch_size=1
    )

    assert result.shape == (2, 768)
    assert valid == [first, last]


def test_encode_returns_an_empty_result_when_every_file_is_unreadable(tmp_path: Path) -> None:
    broken = [_garbage_file(tmp_path / f"broken{i}.png") for i in range(2)]

    result, valid = brain.encode(_ZeroEncoder(), broken, transform=_dummy_transform)

    assert result.shape == (0, 768)
    assert result.dtype == np.float32
    assert valid == []


def test_encode_feeds_rgb_images_to_the_transform(tmp_path: Path) -> None:
    """Grayscale and RGBA files must reach the processor as RGB; DINOv3 expects 3 channels."""
    paths = [_solid_png(tmp_path / "grey.png", mode="L"), _solid_png(tmp_path / "alpha.png", mode="RGBA")]
    seen_modes: list[str] = []

    def recording_transform(image: Image.Image) -> torch.Tensor:
        seen_modes.append(image.mode)
        return torch.zeros(3, 8, 8)

    brain.encode(_ZeroEncoder(), paths, transform=recording_transform)

    assert seen_modes == ["RGB", "RGB"]


def _unit_vector(axis: int) -> np.ndarray:
    vector = np.zeros(768, dtype=np.float32)
    vector[axis] = 1.0
    return vector


def test_cosine_similarity_returns_float32_for_float64_inputs() -> None:
    query = np.ones((2, 768), dtype=np.float64)
    bank = np.ones((3, 768), dtype=np.float64)

    assert brain.cosine_similarity_matrix(query, bank).dtype == np.float32


def test_cosine_similarity_normalises_each_row_on_its_own() -> None:
    """Rows of different length must still score 1.0 against themselves."""
    rows = np.stack([_unit_vector(0), 3.0 * _unit_vector(1)])

    sims = brain.cosine_similarity_matrix(rows, rows)

    np.testing.assert_allclose(sims, np.eye(2, dtype=np.float32), atol=1e-6)


def test_cosine_similarity_treats_zero_vectors_as_dissimilar_not_nan() -> None:
    query = np.stack([np.zeros(768, dtype=np.float32), _unit_vector(0)])
    bank = np.stack([_unit_vector(0), np.zeros(768, dtype=np.float32)])

    sims = brain.cosine_similarity_matrix(query, bank)

    assert np.isfinite(sims).all()
    assert sims[0].tolist() == [0.0, 0.0]
    assert sims[1].tolist() == [1.0, 0.0]


def test_cosine_similarity_with_empty_bank_has_zero_columns() -> None:
    """is_embedding_duplicate relies on the (M,) shape for 1-d queries."""
    empty_bank = np.zeros((0, 768), dtype=np.float32)

    assert brain.cosine_similarity_matrix(np.ones((2, 768)), empty_bank).shape == (2, 0)
    assert brain.cosine_similarity_matrix(np.ones(768), empty_bank).shape == (0,)


def test_predict_proba_two_embeddings_stay_a_batch() -> None:
    """Only a single row collapses to a scalar; two rows are still a batch."""
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(np.random.randn(4, 768), [0, 1, 0, 1])
    proba = brain.predict_proba(clf, np.random.randn(2, 768).astype(np.float32))
    assert proba.shape == (2,)


def test_embedding_to_bytes_rejects_wrong_dimension() -> None:
    """A truncated vector must fail at write time, not corrupt the dedup index later."""
    with pytest.raises(ValueError, match="768-d"):
        brain.embedding_to_bytes(np.zeros(767, dtype=np.float32))


def test_bytes_to_embedding_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="768 floats"):
        brain.bytes_to_embedding(np.zeros(10, dtype=np.float32).tobytes())


# --- storage format: shape/dtype normalisation and diagnostics ----------------


def test_embedding_to_bytes_normalises_dtype_and_batch_dim_to_768_float32() -> None:
    """The blob is 768 float32 whatever the caller holds: encode() hands out
    float32 rows, but a (1, 768) slice or a float64 vector from numpy maths must
    produce the same bytes, or the DedupIndex would read garbage back."""
    base = np.arange(768, dtype=np.float32)
    blob = brain.embedding_to_bytes(base)
    assert len(blob) == 768 * 4
    assert brain.embedding_to_bytes(base.astype(np.float64)) == blob
    assert brain.embedding_to_bytes(base.reshape(1, 768)) == blob


def test_dimension_errors_name_expected_and_actual_length() -> None:
    """A DB row or a vector of the wrong size is a data problem the operator has
    to chase, so the error says both numbers instead of a bare ValueError."""
    with pytest.raises(ValueError) as excinfo:
        brain.embedding_to_bytes(np.zeros(5, dtype=np.float32))
    assert str(excinfo.value) == "Expected 768-d embedding, got 5"
    with pytest.raises(ValueError) as excinfo:
        brain.bytes_to_embedding(np.zeros(3, dtype=np.float32).tobytes())
    assert str(excinfo.value) == "Expected 768 floats, got 3"


def test_save_classifier_creates_missing_parent_directories(tmp_path: Path) -> None:
    """save_classifier runs before DATA_DIR is bootstrapped (fresh volume, tests
    with a temp dir), so a missing directory chain must not be an error."""
    X = np.vstack([np.full(768, 0.1), np.full(768, 0.9)])
    classifier = LogisticRegression().fit(X, [0, 1])
    path = tmp_path / "nested" / "deeper" / "weights.pkl"
    brain.save_classifier(classifier, path)
    assert brain.load_classifier(path) is not None


# --- device placement ----------------------------------------------------------


class _DeviceRecordingEncoder(torch.nn.Module):
    """Records which device each input batch arrived on.

    Its single parameter decides the encoder's home device. "meta" is used as
    the second device because it exists on every host (it holds no data), so a
    CPU-only machine can still observe whether encode() moved the batch to the
    encoder's device or to the device the caller asked for.
    """

    def __init__(self, home: str) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.empty(1, device=home))
        self.seen_devices: list[str] = []

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.seen_devices.append(batch.device.type)
        return torch.zeros(batch.size(0), 768)


def test_encode_moves_batches_to_where_the_encoder_lives(tmp_path: Path) -> None:
    """A GPU-resident encoder must receive GPU batches without the caller
    naming the device: the trainer and the scraper only pass the encoder."""
    encoder = _DeviceRecordingEncoder("meta")
    path = _solid_png(tmp_path / "a.png")
    brain.encode(encoder, [path], transform=_dummy_transform)
    assert encoder.seen_devices == ["meta"]


def test_encode_honours_an_explicit_device_over_the_encoder_home(tmp_path: Path) -> None:
    encoder = _DeviceRecordingEncoder("cpu")
    path = _solid_png(tmp_path / "a.png")
    brain.encode(encoder, [path], transform=_dummy_transform, device="meta")
    assert encoder.seen_devices == ["meta"]
