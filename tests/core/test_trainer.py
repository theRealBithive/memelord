"""Tests for core.trainer."""

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import django
import numpy as np
from django.test import TestCase, override_settings
from loguru import logger
from PIL import Image

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from core import brain, siglip, taste, trainer
from ratings.models import Image as ImageModel


def _search_fields(seed: int) -> dict:
    """A current SigLIP2 vector for a rated fixture row, so run() needs no search encoder (V13)."""
    vector = np.random.default_rng(1000 + seed).standard_normal(768).astype(np.float32)
    return {
        "search_embedding": brain.embedding_to_bytes(vector),
        "search_embedding_model": siglip.SEARCH_ENCODER_ID,
    }


class TrainerTests(TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmpdir.name)
        (self.data_dir / "images").mkdir()
        self.settings_override = override_settings(DATA_DIR=self.data_dir)
        self.settings_override.enable()

    def tearDown(self) -> None:
        self.settings_override.disable()
        self._tmpdir.cleanup()

    def _write_image(self, rel: str) -> Path:
        path = self.data_dir / rel
        Image.new("RGB", (10, 10), color="red").save(path)
        return path

    def test_collect_image_paths_empty(self) -> None:
        """collect_image_paths returns empty when no DB rows."""
        good, bad = trainer.collect_image_paths(self.data_dir)
        self.assertEqual(good, [])
        self.assertEqual(bad, [])

    def test_trashed_image_collected_as_negative(self) -> None:
        """Score 0 (trash) is kept on disk and collected into the negative set."""
        h = uuid.uuid4().hex
        path = f"images/{h}.jpg"
        self._write_image(path)
        ImageModel.objects.create(
            content_hash=h, file_path=path, source_label="t", score=0
        )
        good, bad = trainer.collect_image_paths(self.data_dir)
        self.assertEqual(good, [])
        self.assertIn(str(self.data_dir / path), [str(p) for p in bad])

    def _make_scored(self, score: int) -> str:
        h = uuid.uuid4().hex
        path = f"images/{h}.jpg"
        self._write_image(path)
        ImageModel.objects.create(
            content_hash=h, file_path=path, source_label="t", score=score, **_search_fields(score)
        )
        return h

    def test_positive_sample_weights_formula(self) -> None:
        """Score 5-6 → 3.0, score 3-4 → 1.0; negatives are not in the positive map."""
        for score in (3, 4, 5, 6):
            self._make_scored(score)
        for score in (0, 1, 2):
            self._make_scored(score)

        weights = trainer._get_sample_weights(self.data_dir)

        for img in ImageModel.objects.filter(score__gte=3):
            w = weights[str(self.data_dir / img.file_path)]
            expected = 3.0 if img.score >= 5 else 1.0
            self.assertAlmostEqual(w, expected, msg=f"score={img.score}")

        for img in ImageModel.objects.filter(score__lte=2):
            self.assertNotIn(str(self.data_dir / img.file_path), weights)

    def test_negative_sample_weights_formula(self) -> None:
        """Score 0 (trash) → 3.0, score 1-2 → 1.0; positives not in the negative map."""
        for score in (0, 1, 2):
            self._make_scored(score)
        for score in (3, 5):
            self._make_scored(score)

        weights = trainer._get_negative_weights(self.data_dir)

        for img in ImageModel.objects.filter(score__lte=2):
            w = weights[str(self.data_dir / img.file_path)]
            expected = 3.0 if img.score == 0 else 1.0
            self.assertAlmostEqual(w, expected, msg=f"score={img.score}")

        for img in ImageModel.objects.filter(score__gte=3):
            self.assertNotIn(str(self.data_dir / img.file_path), weights)

    @patch.object(brain, "get_transform")
    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_saves_taste_weights(self, mock_encode, mock_get_encoder, mock_get_transform) -> None:
        """run() fits taste classifier and saves weights."""
        pos = self._write_image("images/pos.png")
        neg = self._write_image("images/neg.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
            **_search_fields(1),
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
            **_search_fields(2),
        )
        mock_get_encoder.return_value = None
        embeddings = np.array([[0.1] * 768, [0.2] * 768], dtype=np.float32)
        mock_encode.return_value = (embeddings, [pos, neg])

        weights_path = self.data_dir / "weights.pkl"
        trainer.run(data_dir=self.data_dir, weights_path=weights_path)
        self.assertTrue(weights_path.exists())

    @patch.object(brain, "get_transform")
    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_uses_cached_embeddings_without_encoding(
        self, mock_encode, mock_get_encoder, mock_get_transform
    ) -> None:
        """With every row's embedding cached, run() skips the encoder entirely."""
        self._write_image("images/pos.png")
        self._write_image("images/neg.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
            embedding=brain.embedding_to_bytes(
                np.full(768, 0.1, dtype=np.float32)
            ),
            embedding_model=brain.ENCODER_ID,
            **_search_fields(1),
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
            embedding=brain.embedding_to_bytes(
                np.full(768, 0.2, dtype=np.float32)
            ),
            embedding_model=brain.ENCODER_ID,
            **_search_fields(2),
        )

        weights_path = self.data_dir / "weights.pkl"
        trainer.run(data_dir=self.data_dir, weights_path=weights_path)

        self.assertTrue(weights_path.exists())
        mock_encode.assert_not_called()
        mock_get_encoder.assert_not_called()

    @patch.object(brain, "get_transform")
    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_encodes_only_uncached_rows(
        self, mock_encode, mock_get_encoder, mock_get_transform
    ) -> None:
        """With a partial cache, run() encodes exactly the rows missing an embedding."""
        self._write_image("images/pos.png")
        neg = self._write_image("images/neg.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
            embedding=brain.embedding_to_bytes(
                np.full(768, 0.1, dtype=np.float32)
            ),
            embedding_model=brain.ENCODER_ID,
            **_search_fields(1),
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
            **_search_fields(2),
        )
        mock_get_encoder.return_value = None
        mock_encode.return_value = (
            np.full((1, 768), 0.2, dtype=np.float32),
            [neg],
        )

        weights_path = self.data_dir / "weights.pkl"
        trainer.run(data_dir=self.data_dir, weights_path=weights_path)

        self.assertTrue(weights_path.exists())
        mock_encode.assert_called_once()
        # encode() must receive only the uncached path, not the whole training set.
        encoded_paths = mock_encode.call_args.args[1]
        self.assertEqual(encoded_paths, [neg])

    @patch.object(brain, "get_transform")
    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_reencodes_corrupt_cached_embedding(
        self, mock_encode, mock_get_encoder, mock_get_transform
    ) -> None:
        """A wrong-dimension cached blob must not crash train; it falls through to encode."""
        self._write_image("images/pos.png")
        bad = self._write_image("images/neg.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
            embedding=brain.embedding_to_bytes(
                np.full(768, 0.1, dtype=np.float32)
            ),
            embedding_model=brain.ENCODER_ID,
            **_search_fields(1),
        )
        # A 512-d blob bypasses embedding_to_bytes' dimension guard but raises
        # ValueError in bytes_to_embedding — simulates an encoder-dimension swap.
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
            embedding=np.full(512, 0.2, dtype=np.float32).tobytes(),
            embedding_model=brain.ENCODER_ID,
            **_search_fields(2),
        )
        mock_get_encoder.return_value = None
        mock_encode.return_value = (
            np.full((1, 768), 0.2, dtype=np.float32),
            [bad],
        )

        weights_path = self.data_dir / "weights.pkl"
        trainer.run(data_dir=self.data_dir, weights_path=weights_path)

        self.assertTrue(weights_path.exists())
        # The corrupt row must be re-encoded, not propagated as a crash.
        mock_encode.assert_called_once()
        encoded_paths = mock_encode.call_args.args[1]
        self.assertEqual(encoded_paths, [bad])

    @patch.object(brain, "get_transform")
    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_saves_nsfw_weights_when_labels_exist(
        self, mock_encode, mock_get_encoder, mock_get_transform
    ) -> None:
        """run() saves NSFW classifier when both classes exist."""
        self._write_image("images/pos.png")
        self._write_image("images/neg.png")
        self._write_image("images/safe.png")
        self._write_image("images/nsfw.png")
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/pos.png",
            source_label="t",
            score=5,
            **_search_fields(1),
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/neg.png",
            source_label="t",
            score=1,
            **_search_fields(2),
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/safe.png",
            source_label="t",
            is_nsfw=False,
        )
        ImageModel.objects.create(
            content_hash=uuid.uuid4().hex,
            file_path="images/nsfw.png",
            source_label="t",
            is_nsfw=True,
        )
        mock_get_encoder.return_value = None
        all_paths = [
            self.data_dir / "images/pos.png",
            self.data_dir / "images/neg.png",
            self.data_dir / "images/safe.png",
            self.data_dir / "images/nsfw.png",
        ]
        mock_encode.return_value = (np.random.randn(4, 768).astype(np.float32), all_paths)

        taste_path = self.data_dir / "taste.pkl"
        nsfw_path = self.data_dir / "nsfw.pkl"
        trainer.run(
            data_dir=self.data_dir,
            weights_path=taste_path,
            nsfw_weights_path=nsfw_path,
        )
        self.assertTrue(taste_path.exists())
        self.assertTrue(nsfw_path.exists())
        # Taste contract V12, V17: the taste model judges the 1536-d feature,
        # the NSFW head stays on the 768-d DINOv3 vector alone.
        self.assertEqual(taste.load_taste_model(taste_path).shared.n_features_in_, taste.FEATURE_DIM)
        self.assertEqual(brain.load_classifier(nsfw_path).n_features_in_, brain.EMBEDDING_DIM)

    def _make_cached(
        self, label: str, score: int | None, seed: int, *, search: bool = True, is_nsfw: bool = False
    ) -> Path:
        """A row with a current DINOv3 vector (and, by default, a search vector), so run() needs no encoder."""
        h = uuid.uuid4().hex
        rel = f"images/{h}.png"
        path = self._write_image(rel)
        vector = np.random.default_rng(seed).standard_normal(768).astype(np.float32)
        ImageModel.objects.create(
            content_hash=h, file_path=rel, source_label=label, score=score, is_nsfw=is_nsfw,
            embedding=brain.embedding_to_bytes(vector), embedding_model=brain.ENCODER_ID,
            **(_search_fields(seed) if search else {}),
        )
        return path

    @patch.object(siglip, "get_image_transform")
    @patch.object(siglip, "get_image_encoder")
    @patch.object(brain, "encode")
    def test_run_fetches_missing_search_vectors_for_rated_rows_only(
        self, mock_encode, mock_siglip_encoder, mock_siglip_transform
    ) -> None:
        """Taste contract: V15 — the rated row gets its SigLIP2 vector inline, the unrated one waits for the chain."""
        rated_without = self._make_cached("t", 5, 1, search=False)
        self._make_cached("t", 1, 2)
        unrated_without = self._make_cached("t", None, 3, search=False)
        siglip_encoder = object()
        mock_siglip_encoder.return_value = siglip_encoder

        def encode_each(encoder, image_paths, **kwargs):
            vectors = np.stack([np.full(768, 0.5, dtype=np.float32) for _ in image_paths])
            return vectors, list(image_paths)

        mock_encode.side_effect = encode_each
        weights_path = self.data_dir / "weights.pkl"

        trainer.run(data_dir=self.data_dir, weights_path=weights_path)

        mock_encode.assert_called_once()
        self.assertIs(mock_encode.call_args.args[0], siglip_encoder)
        self.assertEqual(mock_encode.call_args.args[1], [rated_without])
        rated_row = ImageModel.objects.get(file_path=str(rated_without.relative_to(self.data_dir)))
        unrated_row = ImageModel.objects.get(file_path=str(unrated_without.relative_to(self.data_dir)))
        self.assertEqual(rated_row.search_embedding_model, siglip.SEARCH_ENCODER_ID)
        self.assertIsNone(unrated_row.search_embedding)
        self.assertTrue(weights_path.exists())

    @patch.object(brain, "encode")
    def test_a_bad_cached_search_blob_is_skipped_with_a_warning(self, mock_encode) -> None:
        """Taste contract: V13 — a search blob of the wrong size is not a feature; the row drops out, the run goes on."""
        self._make_cached("t", 6, 1)
        self._make_cached("t", 1, 2)
        h = uuid.uuid4().hex
        rel = f"images/{h}.png"
        self._write_image(rel)
        ImageModel.objects.create(
            content_hash=h, file_path=rel, source_label="t", score=5,
            embedding=brain.embedding_to_bytes(np.full(768, 0.3, dtype=np.float32)),
            embedding_model=brain.ENCODER_ID,
            search_embedding=np.full(512, 0.2, dtype=np.float32).tobytes(),
            search_embedding_model=siglip.SEARCH_ENCODER_ID,
        )
        weights_path = self.data_dir / "weights.pkl"

        messages = self._run_capturing_log(weights_path)

        mock_encode.assert_not_called()
        bad_path = str(self.data_dir / rel)
        self.assertTrue(
            any(m.startswith(f"train: ignoring bad cached search embedding for {bad_path} (") for m in messages),
            messages,
        )
        self.assertIn("train: 1 rated image(s) skipped: no search vector yet", messages)
        self.assertIn("Training taste classifier on 2 samples", messages)

    @patch.object(siglip, "get_image_transform")
    @patch.object(siglip, "get_image_encoder")
    @patch.object(brain, "encode")
    def test_run_refuses_when_the_only_good_row_has_no_search_vector(
        self, mock_encode, mock_siglip_encoder, mock_siglip_transform
    ) -> None:
        """Taste contract: V13 — a side that lost all its rows to the search gate stops the run, like an empty side."""
        self._make_cached("t", 5, 1, search=False)
        self._make_cached("t", 1, 2)
        mock_encode.return_value = (np.zeros((0, 768), dtype=np.float32), [])
        weights_path = self.data_dir / "weights.pkl"

        with self.assertRaisesRegex(RuntimeError, "Need at least one good and one bad image"):
            trainer.run(data_dir=self.data_dir, weights_path=weights_path)
        self.assertFalse(weights_path.exists())

    @patch.object(siglip, "get_image_transform")
    @patch.object(siglip, "get_image_encoder")
    @patch.object(brain, "encode")
    def test_a_rated_row_the_search_encoder_cannot_read_is_skipped_and_counted(
        self, mock_encode, mock_siglip_encoder, mock_siglip_transform
    ) -> None:
        """Taste contract: V13 — no search vector, no training row; the log says how many."""
        self._make_cached("t", 6, 1)
        self._make_cached("t", 5, 2, search=False)
        self._make_cached("t", 1, 3)
        mock_encode.return_value = (np.zeros((0, 768), dtype=np.float32), [])
        weights_path = self.data_dir / "weights.pkl"

        messages = self._run_capturing_log(weights_path)

        self.assertIn("train: 1 rated image(s) skipped: no search vector yet", messages)
        self.assertIn("Training taste classifier on 2 samples", messages)
        self.assertTrue(weights_path.exists())

    def _run_capturing_log(self, weights_path: Path) -> list[str]:
        messages: list[str] = []
        sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
        try:
            trainer.run(data_dir=self.data_dir, weights_path=weights_path)
        finally:
            logger.remove(sink_id)
        return messages

    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_run_fits_a_model_per_source_past_the_threshold_and_says_so(
        self, mock_encode, mock_get_encoder
    ) -> None:
        """Taste contract: V2, V8, V9 — tg (10/10) gets its own model, wsg (3/2) falls back."""
        seed = 0
        for score in (3, 4, 5, 6, 3, 4, 5, 6, 3, 6):
            seed += 1
            self._make_cached("tg", score, seed)
        for score in (0, 1, 2, 0, 1, 2, 0, 1, 2, 0):
            seed += 1
            self._make_cached("tg", score, seed)
        for score in (4, 5, 6):
            seed += 1
            self._make_cached("wsg", score, seed)
        for score in (1, 0):
            seed += 1
            self._make_cached("wsg", score, seed)
        weights_path = self.data_dir / "weights.pkl"

        messages = self._run_capturing_log(weights_path)

        mock_encode.assert_not_called()
        self.assertIn("taste: own model for 'tg' (10 good, 10 bad)", messages)
        self.assertIn("taste: 'wsg' uses the shared model (3 good, 2 bad, needs 10/10)", messages)
        model = taste.load_taste_model(weights_path)
        self.assertEqual(set(model.per_source), {"tg"})
        self.assertEqual(sorted(p.name for p in self.data_dir.glob("*.pkl")), ["weights.pkl"])

    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_flagged_rows_train_the_nsfw_group_and_leave_their_sources(
        self, mock_encode, mock_get_encoder
    ) -> None:
        """Taste contract: V24, V27 — flagged rows of two sources make one '(nsfw)' model; tg without them is 10/9 and falls back; wsg, flagged only, gets no line."""
        seed = 0
        for score in (3, 4, 5, 6, 3, 4, 5, 6, 3, 6):
            seed += 1
            self._make_cached("tg", score, seed)
        for score in (0, 1, 2, 0, 1, 2, 0, 1, 2):
            seed += 1
            self._make_cached("tg", score, seed)
        # Ten liked and ten disliked flagged rows, alternating between the two
        # sources. Under the old per-source key tg would have been 15/14.
        for index, score in enumerate((3, 4, 5, 6, 3, 4, 5, 6, 3, 6)):
            seed += 1
            self._make_cached("tg" if index % 2 == 0 else "wsg", score, seed, is_nsfw=True)
        for index, score in enumerate((0, 1, 2, 0, 1, 2, 0, 1, 2, 0)):
            seed += 1
            self._make_cached("tg" if index % 2 == 0 else "wsg", score, seed, is_nsfw=True)
        weights_path = self.data_dir / "weights.pkl"

        messages = self._run_capturing_log(weights_path)

        mock_encode.assert_not_called()
        self.assertIn("taste: own model for '(nsfw)' (10 good, 10 bad)", messages)
        self.assertIn("taste: 'tg' uses the shared model (10 good, 9 bad, needs 10/10)", messages)
        self.assertFalse(any("'wsg'" in message for message in messages))
        model = taste.load_taste_model(weights_path)
        self.assertEqual(set(model.per_source), {taste.NSFW_GROUP})

    @patch.object(brain, "get_transform")
    @patch.object(brain, "get_encoder")
    @patch.object(brain, "encode")
    def test_a_row_the_encoder_cannot_read_leaves_its_sources_count(
        self, mock_encode, mock_get_encoder, mock_get_transform
    ) -> None:
        """Taste contract: V2 (R1) — counts are of rows that train, after the unreadable row dropped."""
        seed = 0
        for score in (3, 4, 5, 6, 3, 4, 5, 6, 3, 6):
            seed += 1
            self._make_cached("tg", score, seed)
        for score in (0, 1, 2, 0, 1, 2, 0, 1, 2):
            seed += 1
            self._make_cached("tg", score, seed)
        # The tenth disliked row has a file but no vector, and the encoder
        # cannot read it: brain.encode returns nothing for it.
        h = uuid.uuid4().hex
        self._write_image(f"images/{h}.png")
        ImageModel.objects.create(
            content_hash=h, file_path=f"images/{h}.png", source_label="tg", score=1, **_search_fields(99)
        )
        mock_get_encoder.return_value = None
        mock_encode.return_value = (np.zeros((0, 768), dtype=np.float32), [])
        weights_path = self.data_dir / "weights.pkl"

        messages = self._run_capturing_log(weights_path)

        self.assertIn("taste: 'tg' uses the shared model (10 good, 9 bad, needs 10/10)", messages)
        self.assertEqual(taste.load_taste_model(weights_path).per_source, {})

    def test_backfill_skips_vectors_whose_file_has_no_row(self) -> None:
        """Contract: V3 — only rows get stamped; a stray encoded file is ignored, the rest still lands."""
        self._write_image("images/known.png")
        self._write_image("images/stray.png")
        known = ImageModel.objects.create(
            content_hash=uuid.uuid4().hex, file_path="images/known.png", source_label="t", score=5
        )
        vectors = {
            str(self.data_dir / "images/known.png"): np.full(768, 0.3, dtype=np.float32),
            str(self.data_dir / "images/stray.png"): np.full(768, 0.4, dtype=np.float32),
        }
        trainer._backfill_phash_embedding(vectors, self.data_dir)
        known.refresh_from_db()
        self.assertEqual(known.embedding_model, brain.ENCODER_ID)
        self.assertTrue(known.phash)
        np.testing.assert_array_equal(brain.bytes_to_embedding(bytes(known.embedding)), vectors[str(self.data_dir / "images/known.png")])

    def test_backfill_writes_nothing_when_every_row_is_current(self) -> None:
        """Contract: V3 — a current stamp is left alone (re-saving every row was the old dominant cost)."""
        self._write_image("images/cur.png")
        stored = np.full(768, 0.1, dtype=np.float32)
        row = ImageModel.objects.create(
            content_hash=uuid.uuid4().hex, file_path="images/cur.png", source_label="t", score=5,
            phash="abcd", embedding=brain.embedding_to_bytes(stored), embedding_model=brain.ENCODER_ID,
        )
        vectors = {str(self.data_dir / "images/cur.png"): np.full(768, 0.9, dtype=np.float32)}
        with patch.object(ImageModel, "save") as save:
            trainer._backfill_phash_embedding(vectors, self.data_dir)
        save.assert_not_called()
        row.refresh_from_db()
        np.testing.assert_array_equal(brain.bytes_to_embedding(bytes(row.embedding)), stored)
