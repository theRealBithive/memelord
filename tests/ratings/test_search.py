"""
Text search over images with SigLIP2: the vector generation, the ranking and
the gallery view.

Contract (confirmed by the operator on 2026-10-03):

V1 Jedes Bild kann neben dem Taste-Vektor einen Such-Vektor tragen. Der
   Such-Vektor trägt den Namen des Encoders, der ihn erzeugt hat. Ein
   Such-Vektor ohne oder mit fremdem Namen gilt nie als durchsuchbar.
V2 Such-Vektoren und Taste-Vektoren werden nie miteinander verglichen oder
   vermischt. Duplikaterkennung, kNN-Tag-Vorschläge, Taste- und NSFW-Vorhersage
   und „ähnliche Bilder" rechnen ausschließlich mit DINOv3-Vektoren des
   aktuellen Encoders. Die Textsuche rechnet ausschließlich mit SigLIP2.
V3 Ein nicht gelöschtes Bild mit lesbarer Datei ohne aktuellen Such-Vektor
   bekommt ihn durch den Index-Job. Der Index-Job startet von selbst nach jedem
   Hintergrund-Scrape, auf Knopfdruck in der Konfiguration oder per Kommando.
   Er arbeitet bewertete Bilder zuerst ab (zuletzt bewertete vorn), dann
   unbewertete in Download-Reihenfolge. Jedes Bild ist gespeichert, sobald es
   kodiert ist; ein Abbruch verliert höchstens das laufende Stück und der
   nächste Lauf macht dort weiter.
V4 Erhaltungssatz: Für jedes nicht gelöschte Bild gilt „hat Such-Vektor" genau
   dann, wenn „hat Such-Encoder-Kennung". Gelöschte Bilder werden nie kodiert
   und fallen aus Suche und Ähnlichkeit heraus.
V5 Die Textsuche ordnet Bilder nach Ähnlichkeit zwischen Suchtext und
   Bildinhalt, höchste zuerst, und zeigt höchstens eine feste Anzahl bester
   Treffer (100). Die Kandidatenmenge bestimmt der Umfang: „rated" (Standard)
   sind die bewerteten Bilder genau wie die Galerie mit denselben Filtern (min
   score, Tag, NSFW-Schalter, nicht gelöscht); „all" nimmt zusätzlich alle
   unbewerteten, nicht gelöschten Bilder dazu, NSFW-Schalter und Tag-Filter
   gelten weiter, min score gilt nur für bewertete (Trash, Score 0, bleibt
   draußen). Erhaltungssatz: Jedes Suchergebnis liegt in der Kandidatenmenge;
   die Suche zeigt nie ein Bild, das der Filter versteckt.
V6 Die Reihenfolge der Treffer hängt nur vom Suchtext und den Such-Vektoren
   ab, nicht von Downloadzeit, Score, Tag oder Zufall. Dieselbe Suche liefert
   zweimal dieselbe Reihenfolge.
V7 Bilder ohne aktuellen Such-Vektor erscheinen in keinem Suchergebnis. Die
   Galerie nennt ihre Anzahl innerhalb der aktuellen Kandidatenmenge und sagt,
   ob der Index-Job gerade läuft oder gestartet werden muss.
V8 Ein leerer Suchtext (nur Leerraum) ist keine Suche: die Galerie zeigt die
   normale Reihenfolge. Suchtext wird auf 200 Zeichen gekürzt, geht nur an den
   Tokenizer (nie in SQL, nie in eine Shell) und erscheint auf der Seite nur
   als Text, nie als HTML (OWASP A03 Injection, Prüfung im View und Autoescape
   im Template). Umfang und Ähnlichkeits-Anker werden gegen Whitelist bzw.
   Datenbank geprüft, ein unbekannter Wert fällt auf den Standard zurück bzw.
   ergibt 404. Suchen und Indexieren können nur angemeldete Nutzer auslösen
   (OWASP A01, `login_required`; der Index-Knopf zusätzlich nur per POST).
V9 Der Such-Encoder ist SigLIP2 ViT-B/16 bei 224 px mit Googles
   veröffentlichter Vorverarbeitung. Suchtexte werden so tokenisiert, wie das
   Modell trainiert wurde (auf 64 Token aufgefüllt). Such-Vektoren sind
   768-dimensionale float32-Vektoren im selben Speicherformat wie die
   Taste-Vektoren. Deutsche und englische Suchbegriffe finden dasselbe Motiv
   (nur Integrationstest, echtes Modell). (tests/core/test_siglip.py)
V10 Können die SigLIP2-Gewichte nicht geladen werden, endet der Index-Job mit
   einer einzigen klaren Meldung und reiht sich nicht erneut ein; Scrape und
   Train sind davon nie betroffen (sie kodieren kein SigLIP, der Scrape reiht
   nur den Job ein). Eine Suche in der Galerie meldet in diesem Fall einen
   Fehler und zeigt die normale Galerie, nie eine stille leere Trefferliste.
   „Ähnliche Bilder" braucht kein SigLIP und funktioniert dann weiter.
V11 Das Textmodell wird im Webprozess erst bei der ersten Suche geladen und
   danach behalten, nie pro Anfrage neu. Seitenaufrufe ohne Suchtext laden
   kein Modell und bleiben so schnell wie heute. Der Hintergrund-Worker lädt
   nur den Bildturm, nie den Textturm. (tests/core/test_siglip.py)
V12 Die Migration fügt nur die zwei neuen Spalten hinzu. Sie verändert keine
   bestehende Zeile, keinen Taste-Vektor und keine Datei. Fresh Start entfernt
   Such-Vektoren mit den Bildzeilen, ohne eigenen Code.
V13 Der Index-Job belegt den Hintergrund-Worker nie länger als eine Scheibe am
   Stück (1000 Bilder, etwa 10 Minuten). Bleibt danach etwas übrig, reiht er
   sich selbst erneut ein; zwischen zwei Scheiben kommen wartende Scrape- oder
   Train-Jobs dran. Es ist nie mehr als eine Index-Kette gleichzeitig
   eingereiht: ein zweiter Start (Knopf, Scrape-Ende) während einer laufenden
   Kette tut nichts. (tests/ratings/test_search_index_job.py)
V14 Der Index-Job endet von selbst: wenn nichts mehr aussteht, oder wenn eine
   ganze Scheibe kein einziges Bild kodieren konnte (fehlende oder unlesbare
   Dateien). Im zweiten Fall nennt er die Anzahl und verweist auf
   `repair_orphans`. Er dreht nie endlos. (tests/ratings/test_search_index_job.py)
V15 Die Konfigurationsseite zeigt „x von y Bildern indexiert" und den
   Start-Knopf. Während der Kette zeigt sie den Fortschritt (verbleibende
   Anzahl), die Navigation zeigt den laufenden Job wie bei Scrape und Train,
   und Fresh Start ist gesperrt wie bei Scrape und Train. Dieser Zustand kommt
   aus der Datenbank, nicht aus der Browser-Session: ein Neuladen, ein zweiter
   Browser oder der zweite gunicorn-Worker sehen denselben Stand.
   (tests/ratings/test_search_index_job.py)
V16 Suche und Ähnlichkeit lesen nicht bei jeder Anfrage alle Vektoren aus der
   Datenbank. Jeder Webprozess hält je Encoder eine Matrix und erneuert sie,
   sobald sich die Zahl der Bilder mit aktuellem Vektor oder die Zahl der
   gelöschten Bilder geändert hat. Erhaltungssatz: Das Ergebnis mit Matrix ist
   bei jedem Datenbankzustand identisch mit dem Ergebnis ohne Matrix. Ein
   gelöschtes Bild erscheint nie, ein neu indexiertes spätestens bei der
   nächsten Anfrage. (tests/ratings/test_vector_bank.py)
V17 Jedes nicht gelöschte Bild mit aktuellem Taste-Vektor bietet „ähnliche
   Bilder": die ähnlichsten anderen Bilder nach Kosinus über DINOv3, höchste
   zuerst, höchstens 100, ohne das Bild selbst, mit derselben Kandidatenmenge
   und denselben Filtern wie die Textsuche (Umfang Standard hier „all", weil
   der Anker meist im Rückstand steht). Erreichbar aus der Review-Karte und
   aus der Galerie-Lightbox. Ein Bild ohne aktuellen Taste-Vektor sagt das,
   statt eine leere Liste zu zeigen. (tests/ratings/test_similar.py)
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from unittest import mock

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

import numpy as np
from django.contrib.auth import get_user_model
from django.db import connection
from django.test import TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from hypothesis import given
from hypothesis import settings as hsettings
from hypothesis import strategies as st
from hypothesis.extra.django import TestCase as HypothesisTestCase
from loguru import logger
from PIL import Image as PILImage

from core import brain, siglip
from core.brain import EncoderUnavailableError
from ratings import search
from ratings.models import Image, Tag

CURRENT = siglip.SEARCH_ENCODER_ID
FOREIGN = "clip_vitb32"


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(768).astype(np.float32)


def _blob(seed: int) -> bytes:
    return brain.embedding_to_bytes(_vector(seed))


def _row(
    *,
    stamp: str | None = CURRENT,
    score: int | None = 3,
    nsfw: bool = False,
    purged: bool = False,
    seed: int = 0,
    rated_at=None,
    taste_seed: int | None = None,
    file_path: str | None = None,
) -> Image:
    """Insert an Image row; `stamp=None` means no search vector at all."""
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=file_path or f"images/{h}.png",
        source_label="t",
        score=score,
        rated_at=rated_at,
        is_nsfw=nsfw,
        is_purged=purged,
        search_embedding=_blob(seed) if stamp is not None else None,
        search_embedding_model=stamp or "",
        embedding=_blob(taste_seed) if taste_seed is not None else None,
        embedding_model=brain.ENCODER_ID if taste_seed is not None else "",
    )


def _write_png(data_dir: Path, rel: str) -> Path:
    path = data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (4, 4), color=(20, 200, 20)).save(path)
    return path


def _seed_for(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def _fake_encode(encoder, image_paths, transform=None, device=None, batch_size=32, progress_label=""):
    """Deterministic stand-in for brain.encode; paths containing "unreadable" are dropped."""
    valid = [p for p in image_paths if "unreadable" not in str(p)]
    if not valid:
        return np.zeros((0, 768), dtype=np.float32), []
    return np.stack([_vector(_seed_for(str(p))) for p in valid]), valid


def _fake_encode_text(text: str) -> np.ndarray:
    vector = _vector(_seed_for(text))
    return vector / np.linalg.norm(vector)


def _expected_ranking(query: str, candidates) -> list[str]:
    """What rank_by_text must return, computed without the bank (V5, V6)."""
    rows = list(
        candidates.filter(search_embedding_model=CURRENT)
        .exclude(search_embedding=None)
        .values_list("content_hash", "search_embedding")
    )
    scored = []
    for content_hash, blob in rows:
        vector = brain.bytes_to_embedding(bytes(blob))
        similarity = float(brain.cosine_similarity_matrix(_fake_encode_text(query), vector[None, :])[0])
        scored.append((-similarity, content_hash))
    return [h for _, h in sorted(scored)]


# ── stale definition and counts ──────────────────────────────────────────────


class StaleDefinitionTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    def test_vector_without_known_producer_is_never_searchable(self) -> None:
        """Contract: V1"""
        unstamped = _row(stamp="")
        foreign = _row(stamp=FOREIGN)
        self.assertIsNotNone(unstamped.search_embedding)
        self.assertFalse(search.has_search_embedding(unstamped))
        self.assertFalse(search.has_search_embedding(foreign))
        self.assertTrue(search.has_search_embedding(_row()))

    def test_stale_set_is_missing_or_foreign_and_never_purged(self) -> None:
        """Contract: V1, V4"""
        foreign = _row(stamp=FOREIGN)
        missing = _row(stamp=None)
        _row()
        _row(stamp=FOREIGN, purged=True)
        stale = set(search.stale_search_images().values_list("content_hash", flat=True))
        self.assertEqual(stale, {foreign.content_hash, missing.content_hash})

    def test_current_stamp_without_a_vector_is_stale(self) -> None:
        """Contract: V1, V4 (a stamp is a claim about a vector; without the vector it is repaired, not trusted)"""
        half = _row()
        Image.objects.filter(pk=half.pk).update(search_embedding=None)
        half.refresh_from_db()
        self.assertFalse(search.has_search_embedding(half))
        stale = set(search.stale_search_images().values_list("content_hash", flat=True))
        self.assertIn(half.content_hash, stale)

    def test_index_counts_cover_non_purged_rows_only(self) -> None:
        """Contract: V7, V15"""
        _row()
        _row(stamp=None)
        _row(stamp=FOREIGN)
        _row(purged=True)
        _row(stamp=None, purged=True)
        self.assertEqual(search.index_counts(), (1, 3))


# ── the backfill loop ────────────────────────────────────────────────────────


class EncodeStaleTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _encode(self, **kwargs):
        with mock.patch.object(brain, "encode", side_effect=_fake_encode):
            return search.encode_stale_search_embeddings(
                self.data_dir, encoder=object(), transform=object(), **kwargs
            )

    def test_rated_images_come_first_most_recently_rated_in_front(self) -> None:
        """Contract: V3 (risk R13)"""
        now = timezone.now()
        unrated_old = _row(stamp=None, score=None)
        rated_earlier = _row(stamp=None, score=4, rated_at=now - timedelta(days=2))
        rated_latest = _row(stamp=None, score=2, rated_at=now - timedelta(hours=1))
        for img in (unrated_old, rated_earlier, rated_latest):
            _write_png(self.data_dir, img.file_path)

        self.assertEqual(self._encode(limit=1)["encoded"], 1)
        rated_latest.refresh_from_db()
        rated_earlier.refresh_from_db()
        unrated_old.refresh_from_db()
        self.assertTrue(search.has_search_embedding(rated_latest))
        self.assertFalse(search.has_search_embedding(rated_earlier))
        self.assertFalse(search.has_search_embedding(unrated_old))

        self.assertEqual(self._encode(limit=1)["encoded"], 1)
        rated_earlier.refresh_from_db()
        unrated_old.refresh_from_db()
        self.assertTrue(search.has_search_embedding(rated_earlier))
        self.assertFalse(search.has_search_embedding(unrated_old))

    def test_unrated_images_follow_download_order_not_insertion_order(self) -> None:
        """Contract: V3 (unrated rows go in download order, even when the table was filled out of order)"""
        inserted_first = _row(stamp=None, score=None)
        inserted_second = _row(stamp=None, score=None)
        Image.objects.filter(pk=inserted_second.pk).update(
            downloaded_at=timezone.now() - timedelta(days=1)
        )
        for img in (inserted_first, inserted_second):
            _write_png(self.data_dir, img.file_path)

        self.assertEqual(self._encode(limit=1)["encoded"], 1)
        inserted_second.refresh_from_db()
        inserted_first.refresh_from_db()
        self.assertTrue(search.has_search_embedding(inserted_second))
        self.assertFalse(search.has_search_embedding(inserted_first))

    def test_the_encoder_pass_receives_the_arguments_it_was_given(self) -> None:
        """Contract: V3, V9 (the slice is encoded with the encoder, transform and batch size it was handed, under its own log label)"""
        img = _row(stamp=None)
        _write_png(self.data_dir, img.file_path)
        my_encoder, my_transform = object(), object()
        seen: dict = {}

        def _recording_encode(encoder, image_paths, transform=None, device=None, batch_size=32, progress_label=""):
            seen.update(encoder=encoder, transform=transform, batch_size=batch_size, progress_label=progress_label)
            return _fake_encode(encoder, image_paths)

        with mock.patch.object(brain, "encode", side_effect=_recording_encode):
            search.encode_stale_search_embeddings(
                self.data_dir, encoder=my_encoder, transform=my_transform, batch_size=7, progress_label="unit"
            )
        self.assertEqual(
            seen,
            {"encoder": my_encoder, "transform": my_transform, "batch_size": 7, "progress_label": "unit"},
        )

    def test_progress_line_reports_done_of_total_under_the_label(self) -> None:
        """The in-app log is how the operator follows a 10-minute slice: one line per chunk, "label: done/total images encoded with <encoder>", under the default label "index"."""
        for _ in range(3):
            _write_png(self.data_dir, _row(stamp=None).file_path)
        lines: list[str] = []
        sink_id = logger.add(lines.append, format="{message}", level="INFO")
        try:
            self._encode(chunk_size=2)
        finally:
            logger.remove(sink_id)
        progress = [line.rstrip("\n") for line in lines if "images encoded" in line]
        self.assertEqual(
            progress,
            [
                f"index: 2/3 images encoded with {CURRENT}",
                f"index: 3/3 images encoded with {CURRENT}",
            ],
        )

    def test_defaults_are_chunks_of_256_rows_in_batches_of_32(self) -> None:
        """Without explicit sizes the pass hands the encoder 256 paths at a time in batches of 32: the memory shape the 25k run was sized for."""
        for _ in range(257):
            _write_png(self.data_dir, _row(stamp=None).file_path)
        calls: list[tuple[int, int]] = []

        def _recording_encode(encoder, image_paths, transform=None, device=None, batch_size=32, progress_label=""):
            calls.append((len(image_paths), batch_size))
            return _fake_encode(encoder, image_paths)

        with mock.patch.object(brain, "encode", side_effect=_recording_encode):
            search.encode_stale_search_embeddings(self.data_dir, encoder=object(), transform=object())
        self.assertEqual(calls, [(256, 32), (1, 32)])

    def test_each_row_is_saved_as_soon_as_its_chunk_is_encoded(self) -> None:
        """Contract: V3 (risk R5)"""
        rows = [_row(stamp=None) for _ in range(3)]
        for img in rows:
            _write_png(self.data_dir, img.file_path)
        calls = {"n": 0}

        def _encode_then_crash(encoder, image_paths, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("worker killed")
            return _fake_encode(encoder, image_paths, **kwargs)

        with mock.patch.object(brain, "encode", side_effect=_encode_then_crash):
            with self.assertRaises(RuntimeError):
                search.encode_stale_search_embeddings(
                    self.data_dir, encoder=object(), transform=object(), chunk_size=1
                )
        stamped = Image.objects.filter(search_embedding_model=CURRENT).exclude(search_embedding=None)
        self.assertEqual(stamped.count(), 1)

    def test_missing_and_unreadable_files_are_counted_and_stay_stale(self) -> None:
        """Contract: V3, V4 (every file of a chunk is counted, not just the first of each kind)"""
        ok_rows = [_row(stamp=None), _row(stamp=None)]
        for img in ok_rows:
            _write_png(self.data_dir, img.file_path)
        missing = _row(stamp=None)
        unreadable_rows = [
            _row(stamp=None, file_path="images/unreadable-1.png"),
            _row(stamp=None, file_path="images/unreadable-2.png"),
        ]
        for img in unreadable_rows:
            _write_png(self.data_dir, img.file_path)

        result = self._encode()

        self.assertEqual(result, {"encoded": 2, "missing_file": 1, "unreadable": 2})
        for img in (missing, *unreadable_rows):
            img.refresh_from_db()
            self.assertIsNone(img.search_embedding)
            self.assertEqual(img.search_embedding_model, "")
        for img in ok_rows:
            img.refresh_from_db()
            expected = brain.embedding_to_bytes(_vector(_seed_for(str(self.data_dir / img.file_path))))
            self.assertEqual(bytes(img.search_embedding), expected)

    def test_taste_vector_is_never_touched(self) -> None:
        """Contract: V2"""
        img = _row(stamp=None, taste_seed=42)
        _write_png(self.data_dir, img.file_path)
        self._encode()
        img.refresh_from_db()
        self.assertEqual(bytes(img.embedding), _blob(42))
        self.assertEqual(img.embedding_model, brain.ENCODER_ID)
        self.assertTrue(search.has_search_embedding(img))

    def test_rows_are_loaded_without_the_taste_blob_and_in_one_query(self) -> None:
        """Contract: V3 (risk R15: 25k rows must not drag 75 MB of DINOv3 blobs along, nor be fetched one by one)"""
        for _ in range(3):
            _write_png(self.data_dir, _row(stamp=None, taste_seed=1).file_path)
        with CaptureQueriesContext(connection) as captured:
            self._encode()
        selects = [q["sql"] for q in captured.captured_queries if q["sql"].startswith("SELECT") and "ratings_image" in q["sql"]]
        self.assertEqual(len(selects), 1)
        self.assertNotIn('"ratings_image"."embedding"', selects[0])

    def test_missing_transform_is_loaded_from_the_checkpoint(self) -> None:
        """Contract: V9 (the processor shipped with the checkpoint, never a hand-made one)"""
        img = _row(stamp=None)
        _write_png(self.data_dir, img.file_path)
        with mock.patch.object(brain, "encode", side_effect=_fake_encode), \
             mock.patch.object(siglip, "get_image_transform", return_value=object()) as get_transform:
            search.encode_stale_search_embeddings(self.data_dir, encoder=object())
        get_transform.assert_called_once_with()

    def test_weights_that_cannot_load_raise_the_one_clear_error(self) -> None:
        """Contract: V10"""
        img = _row(stamp=None)
        _write_png(self.data_dir, img.file_path)
        with mock.patch.object(siglip, "get_image_encoder", side_effect=EncoderUnavailableError("set HF_TOKEN")):
            with self.assertRaises(EncoderUnavailableError):
                search.encode_stale_search_embeddings(self.data_dir)


ROW = st.fixed_dictionaries(
    {
        "stamp": st.sampled_from([CURRENT, FOREIGN, "", None]),
        "purged": st.booleans(),
        "file": st.sampled_from(["present", "missing", "unreadable"]),
        "taste": st.booleans(),
    }
)


class EncodeStaleProperty(HypothesisTestCase):
    @hsettings(max_examples=30, deadline=None)
    @given(rows=st.lists(ROW, max_size=6))
    def test_conservation_law_after_a_full_pass(self, rows) -> None:
        """Contract: V4 (property), V2.

        Rows reach every combination of stamp (current, foreign, empty, none),
        purged, file state (present, missing, unreadable) and taste vector.
        Unguarded for every row: the taste blob and its stamp are byte-identical
        afterwards. For every non-purged row with a readable file: vector and
        current stamp are present together; for every row out of reach
        (purged, missing, unreadable) nothing changed.
        """
        Image.objects.all().delete()
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            before = {}
            for spec in rows:
                file_path = None
                if spec["file"] == "unreadable":
                    file_path = f"images/unreadable-{uuid.uuid4().hex}.png"
                img = _row(
                    stamp=spec["stamp"], purged=spec["purged"],
                    taste_seed=7 if spec["taste"] else None, file_path=file_path,
                )
                if spec["file"] != "missing":
                    _write_png(data_dir, img.file_path)
                before[img.content_hash] = (
                    bytes(img.search_embedding) if img.search_embedding is not None else None,
                    img.search_embedding_model,
                    bytes(img.embedding) if img.embedding is not None else None,
                    img.embedding_model,
                )

            with mock.patch.object(brain, "encode", side_effect=_fake_encode):
                search.encode_stale_search_embeddings(data_dir, encoder=object(), transform=object())

            for img in Image.objects.all():
                old_blob, old_stamp, old_taste, old_taste_stamp = before[img.content_hash]
                taste_now = bytes(img.embedding) if img.embedding is not None else None
                self.assertEqual(taste_now, old_taste)
                self.assertEqual(img.embedding_model, old_taste_stamp)
                readable = (data_dir / img.file_path).exists() and "unreadable" not in img.file_path
                if img.is_purged or not readable:
                    blob_now = bytes(img.search_embedding) if img.search_embedding is not None else None
                    self.assertEqual(blob_now, old_blob)
                    self.assertEqual(img.search_embedding_model, old_stamp)
                    continue
                self.assertIsNotNone(img.search_embedding)
                self.assertEqual(img.search_embedding_model, CURRENT)


# ── query, scope, candidates ─────────────────────────────────────────────────


class QueryNormalisationTests(TestCase):
    def test_whitespace_is_collapsed_and_length_capped(self) -> None:
        """Contract: V8"""
        self.assertEqual(search.normalize_query("  katze \n auf   skateboard "), "katze auf skateboard")
        self.assertEqual(search.normalize_query("   "), "")
        self.assertEqual(search.normalize_query(None), "")
        self.assertEqual(len(search.normalize_query("x" * 500)), search.MAX_QUERY_LENGTH)

    def test_scope_is_a_whitelist(self) -> None:
        """Contract: V8"""
        self.assertEqual(search.normalize_scope("all"), "all")
        self.assertEqual(search.normalize_scope("rated"), "rated")
        self.assertEqual(search.normalize_scope("everything"), "rated")
        self.assertEqual(search.normalize_scope(None), "rated")


CANDIDATE_ROW = st.fixed_dictionaries(
    {
        "score": st.sampled_from([None, 0, 1, 2, 3, 4, 5, 6]),
        "nsfw": st.booleans(),
        "purged": st.booleans(),
        "tagged": st.booleans(),
    }
)


class CandidateImagesProperty(HypothesisTestCase):
    @hsettings(max_examples=40, deadline=None)
    @given(
        rows=st.lists(CANDIDATE_ROW, max_size=7),
        scope=st.sampled_from(["rated", "all"]),
        min_score=st.integers(1, 6),
        show_nsfw=st.booleans(),
        filter_by_tag=st.booleans(),
    )
    def test_membership_rule(self, rows, scope, min_score, show_nsfw, filter_by_tag) -> None:
        """Contract: V5 (property, risk R19).

        Rows reach unrated, trash, every score, NSFW, purged and tagged; the
        filter reaches both scopes, every min score, both NSFW switch states
        and the tag filter. The expected set is computed from the contract's
        wording, not from the code.
        """
        Image.objects.all().delete()
        tag = Tag.objects.create(name=f"t-{uuid.uuid4().hex[:8]}")
        expected = set()
        for spec in rows:
            img = _row(score=spec["score"], nsfw=spec["nsfw"], purged=spec["purged"])
            if spec["tagged"]:
                img.tags.add(tag)
            if spec["purged"]:
                continue
            if spec["nsfw"] and not show_nsfw:
                continue
            if filter_by_tag and not spec["tagged"]:
                continue
            if spec["score"] is None:
                if scope == "all":
                    expected.add(img.content_hash)
                continue
            if spec["score"] >= min_score:
                expected.add(img.content_hash)
        candidates = search.candidate_images(scope, min_score, tag.name if filter_by_tag else "", show_nsfw)
        self.assertEqual(set(candidates.values_list("content_hash", flat=True)), expected)


# ── ranking ──────────────────────────────────────────────────────────────────


class RankByTextTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        search.SEARCH_BANK.reset()

    def test_results_are_candidates_in_similarity_order(self) -> None:
        """Contract: V5, V6"""
        for seed in range(6):
            _row(seed=seed, score=4)
        candidates = search.candidate_images("rated", 1, "", False)
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text):
            ranked = search.rank_by_text("cat on a skateboard", candidates, 10)
            again = search.rank_by_text("cat on a skateboard", candidates, 10)
        self.assertEqual([h for h, _ in ranked], _expected_ranking("cat on a skateboard", candidates))
        self.assertEqual(ranked, again)
        sims = [s for _, s in ranked]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_limit_caps_the_page_and_filters_apply_before_the_cap(self) -> None:
        """Contract: V5"""
        for seed in range(5):
            _row(seed=seed, score=5)
        hidden = [_row(seed=seed, score=5, nsfw=True) for seed in range(5, 10)]
        candidates = search.candidate_images("rated", 1, "", False)
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text):
            ranked = search.rank_by_text("q", candidates, 3)
        self.assertEqual(len(ranked), 3)
        self.assertFalse({h for h, _ in ranked} & {h.content_hash for h in hidden})

    def test_rows_without_a_current_vector_never_appear(self) -> None:
        """Contract: V7"""
        current = _row(seed=1)
        _row(stamp=FOREIGN, seed=2)
        _row(stamp=None, seed=3)
        candidates = search.candidate_images("rated", 1, "", False)
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text):
            ranked = search.rank_by_text("q", candidates, 10)
        self.assertEqual([h for h, _ in ranked], [current.content_hash])
        self.assertEqual(search.unindexed_count(candidates), 2)

    def test_empty_candidate_set_never_calls_the_text_model(self) -> None:
        """Contract: V11 (risk R10)"""
        _row(seed=1, nsfw=True)
        candidates = search.candidate_images("rated", 1, "", False)
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text) as encode_text:
            self.assertEqual(search.rank_by_text("q", candidates, 10), [])
        encode_text.assert_not_called()

    def test_unindexed_count_follows_the_candidate_set(self) -> None:
        """Contract: V7 (risk R7)"""
        _row(stamp=None, score=3)
        _row(stamp=None, score=None)
        _row(stamp=None, score=3, nsfw=True)
        self.assertEqual(search.unindexed_count(search.candidate_images("rated", 1, "", False)), 1)
        self.assertEqual(search.unindexed_count(search.candidate_images("all", 1, "", False)), 2)
        self.assertEqual(search.unindexed_count(search.candidate_images("all", 1, "", True)), 3)


# ── the gallery view ─────────────────────────────────────────────────────────


def _login(testcase) -> None:
    user = get_user_model().objects.create_user(f"search-{uuid.uuid4().hex[:8]}", password="pw")
    testcase.client.force_login(user)


@override_settings(DEBUG=True)
class GallerySearchViewTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        search.SEARCH_BANK.reset()
        _login(self)

    def test_pages_without_a_query_never_touch_the_text_model(self) -> None:
        """Contract: V11, V8 (risk R8)"""
        _row(seed=1)
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text) as encode_text:
            self.client.get(reverse("gallery"))
            self.client.get(reverse("gallery"), {"q": "   "})
            html = self.client.get(reverse("gallery"), {"q": "   "}).content.decode()
        encode_text.assert_not_called()
        self.assertIn("sort=random", html)

    def test_query_is_echoed_as_text_only_and_links_carry_it_encoded(self) -> None:
        """Contract: V8 (risk R9, OWASP A03)"""
        _row(seed=1)
        hostile = '<b>&"x'
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text):
            html = self.client.get(reverse("gallery"), {"q": hostile}).content.decode()
        self.assertNotIn(hostile, html)
        self.assertIn('value="&lt;b&gt;&amp;&quot;x"', html)
        self.assertIn("q=%3Cb%3E%26%22x", html)

    def test_long_query_reaches_the_model_cut_to_the_cap(self) -> None:
        """Contract: V8"""
        _row(seed=1)
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text) as encode_text:
            self.client.get(reverse("gallery"), {"q": "a" * 500})
        encode_text.assert_called_once_with("a" * search.MAX_QUERY_LENGTH)

    def test_scope_all_shows_unrated_hits_and_the_default_does_not(self) -> None:
        """Contract: V5"""
        rated = _row(seed=1, score=4)
        unrated = _row(seed=2, score=None)
        trash = _row(seed=3, score=0)
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text):
            default = self.client.get(reverse("gallery"), {"q": "x"}).content.decode()
            everything = self.client.get(reverse("gallery"), {"q": "x", "scope": "all"}).content.decode()
        self.assertIn(f'id="item-{rated.content_hash}"', default)
        self.assertNotIn(f'id="item-{unrated.content_hash}"', default)
        self.assertIn(f'id="item-{unrated.content_hash}"', everything)
        self.assertNotIn(f'id="item-{trash.content_hash}"', everything)
        self.assertIn("scope=all", default)
        self.assertIn("gallery-sort-active", everything)

    def test_unindexed_hint_counts_the_candidate_set_only(self) -> None:
        """Contract: V7 (risk R7)"""
        _row(seed=1, score=4)
        _row(stamp=None, score=4)
        _row(stamp=None, score=4, nsfw=True)
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text):
            html = self.client.get(reverse("gallery"), {"q": "x"}).content.decode()
        self.assertIn("1 image in this view is not searchable yet", html)
        self.assertIn("start the index", html)

    def test_missing_weights_fall_back_to_the_gallery_with_a_toast(self) -> None:
        """Contract: V10"""
        img = _row(seed=1, score=4)
        with mock.patch.object(siglip, "encode_text", side_effect=EncoderUnavailableError("accept the licence")):
            response = self.client.get(reverse("gallery"), {"q": "x"})
        html = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn("accept the licence", html)
        self.assertIn(f'id="item-{img.content_hash}"', html)
        self.assertNotIn("match", html.split('class="gallery-count"')[1].split("</span>")[0])

    def test_active_tag_survives_inside_a_search(self) -> None:
        """Contract: V5, V8 (the tag filter is part of the candidate set and of every link)"""
        tag = Tag.objects.create(name=f"t-{uuid.uuid4().hex[:8]}")
        tagged = _row(seed=1, score=4)
        tagged.tags.add(tag)
        untagged = _row(seed=2, score=4)
        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text):
            html = self.client.get(reverse("gallery"), {"q": "x", "tag": tag.name}).content.decode()
        self.assertIn(f'id="item-{tagged.content_hash}"', html)
        self.assertNotIn(f'id="item-{untagged.content_hash}"', html)
        self.assertIn(f'name="tag" value="{tag.name}"', html)
        self.assertIn(f"tag={tag.name}&amp;scope=rated&amp;q=x", html)

    def test_bad_parameters_fall_back_to_the_defaults(self) -> None:
        """Contract: V8 (whitelists, never an error page)"""
        _row(seed=1, score=4)
        for params in ({"min_score": "abc", "sort": "weird"}, {"sort": "oldest"}, {"sort": "random"}):
            response = self.client.get(reverse("gallery"), params)
            self.assertEqual(response.status_code, 200, params)
        html = self.client.get(reverse("gallery"), {"min_score": "abc", "sort": "weird", "scope": "nope"}).content.decode()
        self.assertIn("gallery-score-btn--1 gallery-score-active", html)
        self.assertIn("sort=newest", html)

    def test_search_requires_login(self) -> None:
        """Contract: V8 (OWASP A01)"""
        self.client.logout()
        response = self.client.get(reverse("gallery"), {"q": "x"})
        self.assertEqual(response.status_code, 302)


VIEW_ROW = st.fixed_dictionaries(
    {
        "score": st.sampled_from([None, 0, 2, 4, 6]),
        "nsfw": st.booleans(),
        "purged": st.booleans(),
        "stamp": st.sampled_from([CURRENT, FOREIGN, None]),
    }
)


@override_settings(DEBUG=True)
class GallerySearchProperty(HypothesisTestCase):
    @hsettings(max_examples=25, deadline=None)
    @given(
        rows=st.lists(VIEW_ROW, max_size=7),
        scope=st.sampled_from(["rated", "all"]),
        min_score=st.integers(1, 6),
        show_nsfw=st.booleans(),
        query=st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
    )
    def test_results_are_a_subset_of_the_candidates_and_never_leak_nsfw(
        self, rows, scope, min_score, show_nsfw, query
    ) -> None:
        """Contract: V5, V6, V7 (property, risk R4).

        Rows reach unrated, trash, scored, NSFW, purged and every stamp state;
        the request reaches both scopes, every min score and both NSFW switch
        states; queries include unicode and markup characters. Unguarded for
        every example: no purged image, no unindexed image, and with the
        switch off no NSFW image ever renders; every rendered hit is in the
        candidate set, and the hits come in the order the fake model implies.
        """
        Image.objects.all().delete()
        search.SEARCH_BANK.reset()
        _login(self)
        session = self.client.session
        session["show_nsfw"] = show_nsfw
        session.save()
        for seed, spec in enumerate(rows):
            _row(stamp=spec["stamp"], score=spec["score"], nsfw=spec["nsfw"], purged=spec["purged"], seed=seed)

        with mock.patch.object(siglip, "encode_text", side_effect=_fake_encode_text):
            html = self.client.get(
                reverse("gallery"), {"q": query, "scope": scope, "min_score": min_score}
            ).content.decode()

        rendered = []
        for img in Image.objects.order_by("content_hash"):
            if f'id="item-{img.content_hash}"' in html:
                rendered.append(img)
        candidates = search.candidate_images(scope, min_score, "", show_nsfw)
        allowed = set(candidates.values_list("content_hash", flat=True))
        for img in rendered:
            self.assertIn(img.content_hash, allowed)
            self.assertFalse(img.is_purged)
            self.assertTrue(search.has_search_embedding(img))
            if not show_nsfw:
                self.assertFalse(img.is_nsfw)
        expected_order = _expected_ranking(search.normalize_query(query), candidates)
        positions = {h: html.index(f'id="item-{h}"') for h in expected_order}
        self.assertEqual(sorted(positions, key=positions.get), expected_order)
