"""Embedding generations across scrape, train, dedup, kNN and the views.

Contract (confirmed by the operator on 2026-10-02):

V1 Jedes gespeicherte Embedding trägt den Namen des Encoders, der es erzeugt
   hat. Ein Embedding ohne bekannten Erzeuger gilt nie als aktuell.
V2 Nur Embeddings des aktuellen Encoders nehmen an Ähnlichkeitsrechnungen teil:
   Duplikaterkennung beim Scrape, ähnliche Bilder, kNN-Tag-Vorschläge, Taste-
   und NSFW-Vorhersage. Ältere Embeddings werden dort ignoriert, nie verglichen
   oder gemischt. (The "ähnliche Bilder" strip was removed from the UI in the
   2026-10-02 overhaul, so no test covers it any more.)
V3 Ein Bild mit fehlendem oder älterem Embedding wird beim nächsten
   Encoder-Lauf über dieses Bild neu kodiert. Das Ergebnis ersetzt den alten
   Vektor zusammen mit der neuen Encoder-Kennung. Jedes Bild ist gespeichert,
   sobald es kodiert ist.
V4 Nach einem vollständigen Trainingslauf trägt jedes nicht gelöschte Bild mit
   lesbarer Datei ein aktuelles Embedding. Gelöschte Bilder behalten ihren
   Content-Hash für die Download-Sperre und fallen aus der Embedding-Schicht
   heraus. Erhaltungssatz: Für jedes nicht gelöschte Bild gilt „hat Embedding“
   genau dann, wenn „hat Encoder-Kennung“.
V5 Ein Taste- oder NSFW-Klassifikator merkt sich den Encoder, auf dem er
   trainiert wurde, und wird nur auf Embeddings dieses Encoders angewendet.
   Eine Klassifikator-Datei ohne diese Angabe gilt als fremd und wird nicht
   angewendet. (tests/core/test_brain_encoder.py)
V6 Die Migration kennzeichnet alle vorhandenen Embeddings als DINOv2. Sie
   löscht kein Embedding und berührt keine Datei.
V7 Exakte und perzeptuelle Duplikaterkennung funktionieren für alle Bilder
   unabhängig von der Embedding-Generation. Das Upgrade lässt kein bekanntes
   Bild durch diese beiden Schichten.
V8 Der neue Encoder ist DINOv3 ViT-B/16 mit Metas veröffentlichter
   Vorverarbeitung. Embeddings bleiben 768-dimensionale float32-Vektoren, das
   Speicherformat ändert sich nicht. (tests/core/test_brain_encoder.py)
V9 Können die Modellgewichte nicht bezogen werden, scheitert der Scrape- oder
   Trainingslauf mit einer einzigen klaren Meldung, die sagt, was zu tun ist.
   Er läuft nie still ohne Embeddings weiter. (tests/core/test_brain_encoder.py)
"""

import hashlib
import importlib
import os
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import django
import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.apps import apps as django_apps
from django.test import TestCase
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.django import TestCase as HypothesisTestCase
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression

from core import brain, dedup, trainer
from ratings import scraper, views
from ratings.embeddings import (
    has_current_embedding,
    reencode_stale_embeddings,
    stale_images,
)
from ratings.models import Image, Tag

CURRENT = brain.ENCODER_ID
LEGACY = "dinov2_vitb14"
PHASH = "a1b2c3d4e5f6a7b8"


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(768).astype(np.float32)


def _blob(seed: int) -> bytes:
    return brain.embedding_to_bytes(_vector(seed))


def _row(
    *,
    generation: str = CURRENT,
    embedding: bool = True,
    phash: str = PHASH,
    purged: bool = False,
    score: int | None = None,
    seed: int = 0,
    file_path: str | None = None,
) -> Image:
    """Insert an Image row; `generation` is the stamp a stored vector carries."""
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=file_path or f"images/{h}.png",
        source_label="t",
        phash=phash,
        embedding=_blob(seed) if embedding else None,
        embedding_model=generation if embedding else "",
        is_purged=purged,
        score=score,
    )


def _write_png(data_dir: Path, rel: str) -> Path:
    path = data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (4, 4), color=(200, 20, 20)).save(path)
    return path


def _seed_for(path: Path) -> int:
    return int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16)


def _fake_encode(encoder, image_paths, transform=None, device=None, batch_size=32, progress_label=""):
    """Deterministic stand-in for brain.encode: one vector per path, derived from the path.

    Paths containing "unreadable" are dropped, mirroring how encode() skips files
    it cannot open.
    """
    valid = [p for p in image_paths if "unreadable" not in str(p)]
    if not valid:
        return np.zeros((0, 768), dtype=np.float32), []
    return np.stack([_vector(_seed_for(p)) for p in valid]), valid


def _expected_blob(path: Path) -> bytes:
    return brain.embedding_to_bytes(_vector(_seed_for(path)))


def _assert_conservation(testcase: TestCase, data_dir: Path) -> None:
    """V4 conservation law over every non-purged row whose file the encoder can read.

    Scope correction after a red property run on 2026-10-02: the first draft
    applied the law to every non-purged row, and hypothesis produced a row that
    carries a blob with an empty stamp and has no file on disk. No run can
    repair such a row (there is nothing to encode) and deleting its blob would
    silently destroy data, which V6 forbids. V4's own wording scopes the
    guarantee to "nicht gelöschte Bilder mit lesbarer Datei"; the law is a
    statement about the encoder's reach, so the helper now uses that scope.
    Rows outside the reach must come out byte-identical, which the property
    test asserts separately and unguarded.
    """
    for img in Image.objects.filter(is_purged=False):
        readable = (data_dir / img.file_path).exists() and "unreadable" not in img.file_path
        if not readable:
            continue
        has_embedding = img.embedding is not None
        has_stamp = img.embedding_model != ""
        testcase.assertEqual(
            has_embedding, has_stamp, f"{img.content_hash}: embedding/stamp disagree"
        )
        testcase.assertEqual(img.embedding_model, CURRENT, img.content_hash)


def _save_fitted_classifier(path: Path) -> None:
    X = np.vstack([_vector(1), _vector(2)])
    brain.save_classifier(LogisticRegression().fit(X, [1, 0]), path)


class StaleDefinitionTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    def test_embedding_without_known_producer_is_never_current(self) -> None:
        """Contract: V1"""
        unstamped = _row(generation="")
        self.assertIsNotNone(unstamped.embedding)
        self.assertFalse(has_current_embedding(unstamped))
        self.assertIn(unstamped, stale_images())

    def test_stale_set_is_missing_or_foreign_and_never_purged(self) -> None:
        """Contract: V1, V4"""
        legacy = _row(generation=LEGACY)
        missing = _row(embedding=False)
        current = _row()
        purged_legacy = _row(generation=LEGACY, purged=True)

        stale = set(stale_images().values_list("content_hash", flat=True))

        self.assertEqual(stale, {legacy.content_hash, missing.content_hash})
        self.assertTrue(has_current_embedding(current))
        self.assertNotIn(purged_legacy.content_hash, stale)

    def test_current_stamp_without_a_vector_is_stale_and_stays_out_of_the_index(self) -> None:
        """Contract: V1, V2, V3

        A stamp proves nothing without a vector: the row is re-encoded like a
        missing one, and the dedup index must neither crash on it nor count it.
        """
        h = uuid.uuid4().hex
        stamped_empty = Image.objects.create(
            content_hash=h, file_path=f"images/{h}.png", source_label="t",
            phash=PHASH, embedding=None, embedding_model=CURRENT,
        )
        self.assertIn(stamped_empty, stale_images())
        index = dedup.DedupIndex.from_db()
        self.assertEqual(index.embeddings.shape, (0, 768))
        self.assertIn(h, index.content_hashes)


class DedupIndexGenerationTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    def test_only_current_vectors_enter_cosine_layer_but_all_hashes_stay(self) -> None:
        """Contract: V2, V7"""
        current = _row(seed=3)
        _row(generation=LEGACY, seed=4)
        _row(generation="", seed=5)
        _row(embedding=False)
        _row(generation=LEGACY, purged=True, seed=6)

        index = dedup.DedupIndex.from_db()

        self.assertEqual(len(index.content_hashes), 5)
        self.assertEqual(len(index.phash_ints), 5)
        self.assertEqual(index.embeddings.shape, (1, 768))
        np.testing.assert_array_equal(index.embeddings[0], brain.bytes_to_embedding(current.embedding))


_ROW_SPEC = st.fixed_dictionaries(
    {
        "generation": st.sampled_from([CURRENT, LEGACY, "", "other_v9"]),
        "embedding": st.booleans(),
        "phash": st.booleans(),
        "purged": st.booleans(),
    }
)


class DedupIndexGenerationProperty(HypothesisTestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    @settings(max_examples=40, deadline=None)
    @given(specs=st.lists(_ROW_SPEC, max_size=8))
    def test_index_counts_follow_generation_exactly(self, specs) -> None:
        """Contract: V2, V7 (property).

        The generator reaches every mix of stamped/unstamped, current/legacy,
        purged/live rows, with and without phash, so both the "kept" and the
        "dropped" branches of the index build are exercised in one example.
        """
        for i, spec in enumerate(specs):
            _row(
                generation=spec["generation"],
                embedding=spec["embedding"],
                phash=PHASH if spec["phash"] else "",
                purged=spec["purged"],
                seed=i,
            )

        index = dedup.DedupIndex.from_db()

        expected_vectors = sum(1 for s in specs if s["embedding"] and s["generation"] == CURRENT)
        self.assertEqual(len(index.content_hashes), len(specs))
        self.assertEqual(len(index.phash_ints), sum(1 for s in specs if s["phash"]))
        self.assertEqual(index.embeddings.shape[0], expected_vectors)


class KnnGenerationTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self.tag = Tag.objects.create(name=f"warhammer-{uuid.uuid4().hex[:6]}")

    def test_legacy_anchor_lends_no_tags_and_legacy_target_is_untouched(self) -> None:
        """Contract: V2"""
        legacy_anchor = _row(generation=LEGACY, seed=1)
        legacy_anchor.tags.add(self.tag)
        target_current = _row(seed=1)
        target_legacy = _row(generation=LEGACY, seed=1)

        scraper.populate_knn_tag_suggestions(refill=True)

        target_current.refresh_from_db()
        target_legacy.refresh_from_db()
        self.assertEqual(target_current.knn_tag_suggestions, "")
        self.assertEqual(target_legacy.knn_tag_suggestions, "")

    def test_current_anchor_lends_tags_to_current_target(self) -> None:
        """Contract: V2 (the positive branch, so the filter is not just an empty query)"""
        anchor = _row(seed=1)
        anchor.tags.add(self.tag)
        target = _row(seed=1)

        scraper.populate_knn_tag_suggestions(refill=True)

        target.refresh_from_db()
        self.assertIn(self.tag.name, target.knn_tag_suggestions)


class TastePredictionGenerationTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    def test_fallback_never_applies_classifier_to_legacy_vector(self) -> None:
        """Contract: V2"""
        legacy = _row(generation=LEGACY, seed=1)
        with mock.patch.object(views, "_get_taste_clf", return_value=object()), \
             mock.patch.object(brain, "predict_proba", return_value=0.7) as predict:
            self.assertIsNone(views._taste_prediction(legacy))
        predict.assert_not_called()
        legacy.refresh_from_db()
        self.assertIsNone(legacy.predicted_score)

    def test_fallback_predicts_and_persists_for_current_vector(self) -> None:
        """Contract: V2 (positive branch)"""
        current = _row(seed=1)
        with mock.patch.object(views, "_get_taste_clf", return_value=object()), \
             mock.patch.object(brain, "predict_proba", return_value=0.7):
            self.assertEqual(views._taste_prediction(current), 70)
        current.refresh_from_db()
        self.assertAlmostEqual(current.predicted_score, 0.7)

    def test_stored_prediction_is_kept_regardless_of_generation(self) -> None:
        """Operator decision 2026-10-02: old predictions stay until the next train run."""
        legacy = _row(generation=LEGACY, seed=1)
        Image.objects.filter(pk=legacy.pk).update(predicted_score=0.42)
        legacy.refresh_from_db()
        self.assertEqual(views._taste_prediction(legacy), 42)


class ClassifyImagesGenerationTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.weights = self.data_dir / "w.pkl"
        _save_fitted_classifier(self.weights)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_legacy_unscored_row_is_reencoded_not_predicted_from_old_vector(self) -> None:
        """Contract: V2, V3"""
        current = _row(seed=1)
        legacy = _row(generation=LEGACY, seed=2)
        _write_png(self.data_dir, current.file_path)
        legacy_path = _write_png(self.data_dir, legacy.file_path)
        vision = scraper.VisionConfig(weights_path=self.weights)

        with mock.patch.object(brain, "encode", side_effect=_fake_encode) as encode:
            scraper.classify_images(self.data_dir, vision, encoder=object(), transform=object())

        encode.assert_called_once()
        self.assertEqual(encode.call_args.args[1], [legacy_path])
        legacy.refresh_from_db()
        current.refresh_from_db()
        self.assertEqual(legacy.embedding_model, CURRENT)
        self.assertEqual(bytes(legacy.embedding), _expected_blob(legacy_path))
        self.assertIsNotNone(legacy.predicted_score)
        self.assertIsNotNone(current.predicted_score)
        _assert_conservation(self, self.data_dir)


class ReencodeTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_counts_replacement_and_conservation(self) -> None:
        """Contract: V3, V4"""
        legacy = _row(generation=LEGACY, seed=1)
        missing = _row(embedding=False)
        unstamped = _row(generation="", seed=2)
        current = _row(seed=7)
        no_file = _row(generation=LEGACY, seed=3)
        purged = _row(generation=LEGACY, purged=True, seed=4)
        unreadable = _row(generation=LEGACY, seed=5, file_path="images/unreadable.png")
        for img in (legacy, missing, unstamped, current, purged, unreadable):
            _write_png(self.data_dir, img.file_path)

        with mock.patch.object(brain, "encode", side_effect=_fake_encode):
            result = reencode_stale_embeddings(
                self.data_dir, encoder=object(), transform=object(), chunk_size=2
            )

        self.assertEqual(result, {"encoded": 3, "missing_file": 1, "unreadable": 1})
        for img in (legacy, missing, unstamped):
            img.refresh_from_db()
            self.assertEqual(img.embedding_model, CURRENT)
            self.assertEqual(bytes(img.embedding), _expected_blob(self.data_dir / img.file_path))
        current.refresh_from_db()
        self.assertEqual(bytes(current.embedding), _blob(7))
        for untouched, generation in ((no_file, LEGACY), (purged, LEGACY), (unreadable, LEGACY)):
            untouched.refresh_from_db()
            self.assertEqual(untouched.embedding_model, generation)
        _assert_conservation(self, self.data_dir)

    def test_each_chunk_is_durable_before_the_next_starts(self) -> None:
        """Contract: V3 (resumability)"""
        first = _row(generation=LEGACY, seed=1)
        second = _row(generation=LEGACY, seed=2)
        Image.objects.filter(pk=second.pk).update(
            downloaded_at=first.downloaded_at.replace(year=first.downloaded_at.year + 1)
        )
        for img in (first, second):
            _write_png(self.data_dir, img.file_path)
        calls = {"n": 0}

        def encode_then_crash(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("worker killed")
            return _fake_encode(*args, **kwargs)

        with mock.patch.object(brain, "encode", side_effect=encode_then_crash):
            with self.assertRaises(RuntimeError):
                reencode_stale_embeddings(
                    self.data_dir, encoder=object(), transform=object(), chunk_size=1
                )

        first.refresh_from_db()
        second.refresh_from_db()
        self.assertEqual(first.embedding_model, CURRENT)
        self.assertEqual(second.embedding_model, LEGACY)

    def test_reencode_hands_its_encoder_transform_and_batch_size_to_the_encoder_pass(self) -> None:
        """Contract: V3

        The pass encodes with the model it was given. Falling back to a fresh
        load inside brain.encode would download and build a second copy of the
        encoder for every chunk, so the objects must arrive unchanged.
        """
        legacy = _row(generation=LEGACY, seed=1)
        _write_png(self.data_dir, legacy.file_path)
        encoder, transform = object(), object()

        with mock.patch.object(brain, "encode", side_effect=_fake_encode) as encode:
            reencode_stale_embeddings(
                self.data_dir, encoder=encoder, transform=transform, batch_size=7
            )

        self.assertIs(encode.call_args.args[0], encoder)
        self.assertIs(encode.call_args.kwargs["transform"], transform)
        self.assertEqual(encode.call_args.kwargs["batch_size"], 7)

    def test_every_unreadable_file_in_a_chunk_is_counted_and_the_rest_is_still_encoded(self) -> None:
        """Contract: V3, V4

        Two unreadable files ahead of a good one in the same chunk: the report
        counts both, and the good file behind them still gets its vector.
        """
        first_bad = _row(generation=LEGACY, seed=1, file_path="images/unreadable-1.png")
        second_bad = _row(generation=LEGACY, seed=2, file_path="images/unreadable-2.png")
        good = _row(generation=LEGACY, seed=3)
        for position, img in enumerate((first_bad, second_bad, good)):
            _write_png(self.data_dir, img.file_path)
            Image.objects.filter(pk=img.pk).update(
                downloaded_at=img.downloaded_at.replace(year=2000 + position)
            )

        with mock.patch.object(brain, "encode", side_effect=_fake_encode):
            result = reencode_stale_embeddings(
                self.data_dir, encoder=object(), transform=object(), chunk_size=3
            )

        self.assertEqual(result, {"encoded": 1, "missing_file": 0, "unreadable": 2})
        good.refresh_from_db()
        self.assertEqual(good.embedding_model, CURRENT)
        _assert_conservation(self, self.data_dir)

    def test_a_rating_given_while_the_pass_runs_survives(self) -> None:
        """Contract: V3

        Re-encoding runs in the background while the operator keeps rating.
        The pass may only write the vector and its stamp; writing the whole
        row back would overwrite a score given in the meantime with the stale
        value it loaded minutes earlier.
        """
        img = _row(generation=LEGACY, seed=1)
        _write_png(self.data_dir, img.file_path)

        def encode_and_rate_meanwhile(*args, **kwargs):
            Image.objects.filter(pk=img.pk).update(score=5)
            return _fake_encode(*args, **kwargs)

        with mock.patch.object(brain, "encode", side_effect=encode_and_rate_meanwhile):
            reencode_stale_embeddings(self.data_dir, encoder=object(), transform=object())

        img.refresh_from_db()
        self.assertEqual(img.score, 5)
        self.assertEqual(img.embedding_model, CURRENT)

    def test_encoder_is_not_loaded_when_nothing_is_stale(self) -> None:
        """Contract: V3 (no work, no model download)"""
        _row(seed=1)
        with mock.patch.object(brain, "get_encoder") as get_encoder:
            result = reencode_stale_embeddings(self.data_dir)
        get_encoder.assert_not_called()
        self.assertEqual(result["encoded"], 0)


_REENCODE_SPEC = st.fixed_dictionaries(
    {
        "generation": st.sampled_from([CURRENT, LEGACY, ""]),
        "embedding": st.booleans(),
        "file_exists": st.booleans(),
        "purged": st.booleans(),
    }
)


class ReencodeProperty(HypothesisTestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    @settings(max_examples=40, deadline=None)
    @given(specs=st.lists(_REENCODE_SPEC, max_size=6))
    def test_reencode_restores_conservation_and_touches_only_stale_rows(self, specs) -> None:
        """Contract: V3, V4 (property).

        Rows cover current/legacy/unstamped vectors, missing vectors, missing
        files and purged rows, so the run must both skip and replace in the
        same example. Rows that were already current, purged, or without a file
        must come out byte-identical.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            rows = []
            for i, spec in enumerate(specs):
                img = _row(
                    generation=spec["generation"],
                    embedding=spec["embedding"],
                    purged=spec["purged"],
                    seed=100 + i,
                )
                if spec["file_exists"]:
                    _write_png(data_dir, img.file_path)
                rows.append((img, spec, img.embedding, img.embedding_model))

            with mock.patch.object(brain, "encode", side_effect=_fake_encode):
                result = reencode_stale_embeddings(
                    data_dir, encoder=object(), transform=object(), chunk_size=2
                )

            expected_encoded = sum(
                1
                for _, spec, _, _ in rows
                if not spec["purged"]
                and spec["file_exists"]
                and not (spec["embedding"] and spec["generation"] == CURRENT)
            )
            self.assertEqual(result["encoded"], expected_encoded)
            for img, spec, blob_before, stamp_before in rows:
                img.refresh_from_db()
                was_current = spec["embedding"] and spec["generation"] == CURRENT
                if spec["purged"] or not spec["file_exists"] or was_current:
                    self.assertEqual(img.embedding, blob_before)
                    self.assertEqual(img.embedding_model, stamp_before)
                else:
                    self.assertEqual(img.embedding_model, CURRENT)
                    self.assertEqual(bytes(img.embedding), _expected_blob(data_dir / img.file_path))
            _assert_conservation(self, data_dir)


class MigrationLabelTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    def test_existing_vectors_are_labelled_dinov2_and_nothing_is_deleted(self) -> None:
        """Contract: V6"""
        migration = importlib.import_module("ratings.migrations.0021_image_embedding_model")
        with_vector = _row(generation="", seed=1)
        without_vector = _row(embedding=False)
        blob_before = bytes(with_vector.embedding)

        migration.label_legacy_embeddings(django_apps, None)

        with_vector.refresh_from_db()
        without_vector.refresh_from_db()
        self.assertEqual(with_vector.embedding_model, migration.LEGACY_ENCODER_ID)
        self.assertEqual(bytes(with_vector.embedding), blob_before)
        self.assertEqual(without_vector.embedding_model, "")
        self.assertIsNone(without_vector.embedding)
        self.assertEqual(Image.objects.count(), 2)


class WritePathsStampTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_scrape_insert_stamps_current_encoder(self) -> None:
        """Contract: V1 (scrape-time write path)"""
        path = _write_png(self.data_dir, "images/new.png")
        index = dedup.DedupIndex()
        candidates = [(path, "http://x/new.png", "4chan/wg", "h" * 64, PHASH)]

        with mock.patch.object(brain, "encode", side_effect=_fake_encode):
            inserted = scraper._process_candidates(
                candidates, self.data_dir, index, object(), object(), scraper.VisionConfig(), None
            )

        self.assertEqual(inserted, 1)
        row = Image.objects.get(content_hash="h" * 64)
        self.assertEqual(row.embedding_model, CURRENT)
        self.assertEqual(bytes(row.embedding), _expected_blob(path))
        self.assertEqual(index.embeddings.shape, (1, 768))

    def test_trainer_backfill_replaces_legacy_vector_and_keeps_current_one(self) -> None:
        """Contract: V1, V3 (trainer write path)"""
        legacy = _row(generation=LEGACY, seed=1)
        current = _row(seed=7)
        legacy_path = _write_png(self.data_dir, legacy.file_path)
        current_path = _write_png(self.data_dir, current.file_path)
        path_to_emb = {str(legacy_path): _vector(50), str(current_path): _vector(60)}

        trainer._backfill_phash_embedding(path_to_emb, self.data_dir)

        legacy.refresh_from_db()
        current.refresh_from_db()
        self.assertEqual(legacy.embedding_model, CURRENT)
        self.assertEqual(bytes(legacy.embedding), brain.embedding_to_bytes(_vector(50)))
        self.assertEqual(bytes(current.embedding), _blob(7))


class FullTrainRunMigratesLibraryTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_one_train_run_plus_classification_brings_every_live_row_current(self) -> None:
        """Contract: V4 (and V5: the new classifier file is accepted afterwards)"""
        liked = _row(generation=LEGACY, score=5, seed=1)
        disliked = _row(generation=LEGACY, score=1, seed=2)
        unrated = _row(generation=LEGACY, seed=3)
        purged = _row(generation=LEGACY, purged=True, seed=4)
        for img in (liked, disliked, unrated, purged):
            _write_png(self.data_dir, img.file_path)
        weights = self.data_dir / "w.pkl"

        with mock.patch.object(brain, "get_encoder", return_value=object()), \
             mock.patch.object(brain, "get_transform", return_value=object()), \
             mock.patch.object(brain, "encode", side_effect=_fake_encode):
            trainer.run(data_dir=self.data_dir, weights_path=weights)
            scraper.classify_images(
                self.data_dir,
                scraper.VisionConfig(weights_path=weights),
                encoder=object(),
                transform=object(),
            )

        for img in (liked, disliked, unrated):
            img.refresh_from_db()
            self.assertEqual(img.embedding_model, CURRENT)
        purged.refresh_from_db()
        self.assertEqual(purged.embedding_model, LEGACY)
        self.assertIn(purged.content_hash, dedup.DedupIndex.from_db().content_hashes)
        self.assertIsNotNone(brain.load_classifier(weights))
        _assert_conservation(self, self.data_dir)


class ReencodeLimitAndLazyLoadTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_limit_caps_the_run_and_the_encoder_is_loaded_once(self) -> None:
        """Contract: V3 (a partial run still leaves every touched row current)"""
        rows = [_row(generation=LEGACY, seed=i) for i in range(3)]
        for img in rows:
            _write_png(self.data_dir, img.file_path)

        with mock.patch.object(brain, "get_encoder", return_value=object()) as get_encoder, \
             mock.patch.object(brain, "get_transform", return_value=object()) as get_transform, \
             mock.patch.object(brain, "encode", side_effect=_fake_encode):
            result = reencode_stale_embeddings(self.data_dir, limit=2)

        self.assertEqual(result["encoded"], 2)
        get_encoder.assert_called_once()
        get_transform.assert_called_once()
        self.assertEqual(stale_images().count(), 1)
        _assert_conservation_partial = [
            img for img in Image.objects.filter(embedding_model=CURRENT)
        ]
        self.assertEqual(len(_assert_conservation_partial), 2)


class ReencodeCommandTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_dry_run_reports_without_encoding(self) -> None:
        """Contract: V3 (operator tooling reports the stale count, touches nothing)"""
        from io import StringIO

        from django.core.management import call_command

        legacy = _row(generation=LEGACY, seed=1)
        out = StringIO()
        with mock.patch.object(brain, "get_encoder") as get_encoder:
            call_command("reencode_embeddings", "--dry-run", stdout=out)
        get_encoder.assert_not_called()
        self.assertIn("1 image(s) would be re-encoded", out.getvalue())
        legacy.refresh_from_db()
        self.assertEqual(legacy.embedding_model, LEGACY)

    def test_command_reencodes_and_reports_counts(self) -> None:
        """Contract: V3"""
        from io import StringIO

        from django.core.management import call_command
        from django.test import override_settings

        legacy = _row(generation=LEGACY, seed=1)
        _write_png(self.data_dir, legacy.file_path)
        _row(generation=LEGACY, seed=2)  # no file on disk
        out = StringIO()
        with override_settings(DATA_DIR=self.data_dir), \
             mock.patch.object(brain, "get_encoder", return_value=object()), \
             mock.patch.object(brain, "get_transform", return_value=object()), \
             mock.patch.object(brain, "encode", side_effect=_fake_encode):
            call_command("reencode_embeddings", "--chunk-size", "1", stdout=out)
        self.assertIn("Re-encoded 1 image(s); 1 file(s) missing, 0 unreadable.", out.getvalue())
        legacy.refresh_from_db()
        self.assertEqual(legacy.embedding_model, CURRENT)


class TasteClassifierCacheTests(TestCase):
    """The view-level classifier cache must honour V5 without re-reading a foreign file per render."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.weights = Path(self._tmp.name) / "w.pkl"
        views._taste_clf_cache = None
        views._taste_clf_mtime = None

    def tearDown(self) -> None:
        views._taste_clf_cache = None
        views._taste_clf_mtime = None
        self._tmp.cleanup()

    def test_foreign_file_yields_none_and_is_read_once(self) -> None:
        """Contract: V5"""
        import pickle

        with self.weights.open("wb") as f:
            pickle.dump({"encoder": LEGACY, "classifier": LogisticRegression()}, f)
        with mock.patch.object(views, "WEIGHTS_PATH", self.weights), \
             mock.patch.object(brain, "load_classifier", wraps=brain.load_classifier) as load:
            self.assertIsNone(views._get_taste_clf())
            self.assertIsNone(views._get_taste_clf())
        load.assert_called_once()

    def test_current_file_is_applied(self) -> None:
        """Contract: V5 (positive branch)"""
        _save_fitted_classifier(self.weights)
        with mock.patch.object(views, "WEIGHTS_PATH", self.weights):
            self.assertIsInstance(views._get_taste_clf(), LogisticRegression)

    def test_missing_file_yields_none(self) -> None:
        with mock.patch.object(views, "WEIGHTS_PATH", self.weights):
            self.assertIsNone(views._get_taste_clf())
