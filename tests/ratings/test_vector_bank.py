"""
VectorBank: the in-process matrix behind text search and similar images.

Contract (full list in tests/ratings/test_search.py):

V6  Die Reihenfolge der Treffer hängt nur vom Suchtext und den Such-Vektoren
    ab. Dieselbe Suche liefert zweimal dieselbe Reihenfolge.
V16 Suche und Ähnlichkeit lesen nicht bei jeder Anfrage alle Vektoren aus der
    Datenbank. Jeder Webprozess hält je Encoder eine Matrix und erneuert sie,
    sobald sich die Zahl der Bilder mit aktuellem Vektor oder die Zahl der
    gelöschten Bilder geändert hat. Erhaltungssatz: Das Ergebnis mit Matrix ist
    bei jedem Datenbankzustand identisch mit dem Ergebnis ohne Matrix. Ein
    gelöschtes Bild erscheint nie, ein neu indexiertes spätestens bei der
    nächsten Anfrage.
"""

from __future__ import annotations

import os
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

import numpy as np
from django.test import TestCase
from hypothesis import given
from hypothesis import settings as hsettings
from hypothesis import strategies as st
from hypothesis.extra.django import TestCase as HypothesisTestCase

from core import brain
from ratings.models import Image
from ratings.vector_bank import VectorBank, first_allowed

STAMP = "bank_test_stamp"
FOREIGN = "some_other_encoder"


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(768).astype(np.float32)


def _row(seed: int, *, stamp: str = STAMP, purged: bool = False, blob: bool = True) -> Image:
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.png",
        source_label="t",
        is_purged=purged,
        search_embedding=brain.embedding_to_bytes(_vector(seed)) if blob else None,
        search_embedding_model=stamp,
    )


def _direct(query: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Hashes and cosines computed straight from the database, the way the bank promises to match (V16)."""
    rows = list(
        Image.objects.filter(search_embedding_model=STAMP)
        .exclude(search_embedding=None)
        .order_by("content_hash")
        .values_list("content_hash", "search_embedding")
    )
    if not rows:
        return [], np.zeros(0, dtype=np.float32)
    matrix = np.stack([brain.bytes_to_embedding(bytes(blob)) for _, blob in rows])
    sims = brain.cosine_similarity_matrix(query, matrix)
    order = np.argsort(-sims, kind="stable")
    return [rows[i][0] for i in order], sims[order]


def _direct_ranking(query: np.ndarray) -> list[str]:
    return _direct(query)[0]


def _bank() -> VectorBank:
    return VectorBank("search_embedding", "search_embedding_model", STAMP)


class VectorBankTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    def test_ranking_equals_a_fresh_computation_from_the_database(self) -> None:
        """Contract: V16"""
        for seed in range(5):
            _row(seed)
        query = _vector(99)
        hashes, sims = _bank().rank(query)
        expected_hashes, expected_sims = _direct(query)
        self.assertEqual(hashes, expected_hashes)
        np.testing.assert_allclose(sims, expected_sims, atol=1e-5)
        self.assertEqual(list(sims), sorted(sims, reverse=True))
        # float32 like the stored vectors; a float64 bank would double the 77 MB.
        self.assertEqual(sims.dtype, np.float32)

    def test_foreign_stamp_or_missing_blob_stays_out_of_the_matrix(self) -> None:
        """Contract: V16 (the matrix mirrors the stamp and nothing else)"""
        current = _row(1)
        _row(2, stamp=FOREIGN)
        _row(3, stamp="")
        _row(4, blob=False)
        bank = _bank()
        hashes, _ = bank.rank(_vector(0))
        self.assertEqual(hashes, [current.content_hash])
        self.assertEqual(bank.size(), 1)

    def test_purged_rows_stay_in_the_matrix_for_the_candidate_filter_to_drop(self) -> None:
        """Contract: V16 (purge moves the version; exclusion is the candidate set's job)"""
        _row(1)
        purged = _row(2, purged=True)
        hashes, _ = _bank().rank(_vector(0))
        self.assertIn(purged.content_hash, hashes)

    def test_unchanged_version_costs_two_counts_and_reads_no_blob(self) -> None:
        """Contract: V16 (risk R17)"""
        for seed in range(3):
            _row(seed)
        bank = _bank()
        bank.rank(_vector(0))
        with self.assertNumQueries(2):
            bank.rank(_vector(0))
        with self.assertNumQueries(2):
            self.assertFalse(bank.refresh_if_changed())

    def test_a_purge_moves_the_version_and_reloads(self) -> None:
        """Contract: V16 (risk R16/R17)"""
        rows = [_row(seed) for seed in range(3)]
        bank = _bank()
        bank.rank(_vector(0))
        rows[0].is_purged = True
        rows[0].save(update_fields=["is_purged"])
        with self.assertNumQueries(3):
            reloaded = bank.refresh_if_changed()
        self.assertTrue(reloaded)

    def test_a_newly_stamped_row_is_visible_on_the_next_call(self) -> None:
        """Contract: V16"""
        _row(1)
        bank = _bank()
        bank.rank(_vector(0))
        new = _row(2)
        hashes, _ = bank.rank(_vector(0))
        self.assertIn(new.content_hash, hashes)

    def test_empty_bank_ranks_to_nothing_without_error(self) -> None:
        """Contract: V16 (risk R10)"""
        hashes, sims = _bank().rank(_vector(0))
        self.assertEqual(hashes, [])
        self.assertEqual(sims.shape, (0,))
        self.assertEqual(sims.dtype, np.float32)

    def test_a_zero_query_ranks_everything_at_zero_without_nan(self) -> None:
        """Contract: V5 (a degenerate query, e.g. a zero text vector, has no preference; it must not poison the ranking with NaN)"""
        for seed in range(3):
            _row(seed)
        hashes, sims = _bank().rank(np.zeros(768, dtype=np.float32))
        self.assertEqual(len(hashes), 3)
        np.testing.assert_array_equal(sims, np.zeros(3, dtype=np.float32))

    def test_a_float64_query_is_ranked_in_float32(self) -> None:
        """Contract: V16 (the bank works in the storage precision whatever the caller hands in; no float64 copy of 77 MB)"""
        for seed in range(3):
            _row(seed)
        hashes, sims = _bank().rank(_vector(5).astype(np.float64))
        self.assertEqual(len(hashes), 3)
        self.assertEqual(sims.dtype, np.float32)

    def test_identical_vectors_tie_in_content_hash_order_every_time(self) -> None:
        """Contract: V6"""
        twins = [_row(7), _row(7)]
        bank = _bank()
        first, _ = bank.rank(_vector(7))
        second, _ = bank.rank(_vector(7))
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(t.content_hash for t in twins))

    def test_ties_keep_content_hash_order_even_in_large_groups(self) -> None:
        """Contract: V6 (ties are broken by content_hash beyond the small arrays where every sort happens to be stable)"""
        near = [_row(7) for _ in range(20)]
        far = [_row(8) for _ in range(20)]
        hashes, _ = _bank().rank(_vector(7))
        self.assertEqual(hashes[:20], sorted(r.content_hash for r in near))
        self.assertEqual(hashes[20:], sorted(r.content_hash for r in far))

    def test_blobs_from_the_database_round_trip_into_unit_rows(self) -> None:
        """Contract: V16 (risk R6: BinaryField hands back a memoryview)"""
        row = _row(11)
        bank = _bank()
        bank.refresh_if_changed()
        expected = _vector(11) / np.linalg.norm(_vector(11))
        index = bank._hashes.index(row.content_hash)
        np.testing.assert_allclose(bank._matrix[index], expected, atol=1e-6)
        self.assertEqual(bank._matrix.dtype, np.float32)

    def test_first_allowed_keeps_order_skips_others_and_stops_at_limit(self) -> None:
        """Contract: V5"""
        ranked = ["a", "b", "c", "d", "e"]
        sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        picked = first_allowed(ranked, sims, {"b", "d", "e"}, limit=2)
        self.assertEqual([h for h, _ in picked], ["b", "d"])
        self.assertAlmostEqual(picked[0][1], 0.8, places=5)


OPERATION = st.one_of(
    st.tuples(st.just("add"), st.integers(0, 10**6), st.booleans()),
    st.tuples(st.just("purge")),
    st.tuples(st.just("fresh_start")),
)


class VectorBankProperty(HypothesisTestCase):
    @hsettings(max_examples=25, deadline=None)
    @given(operations=st.lists(OPERATION, max_size=8), query_seed=st.integers(0, 10**6))
    def test_bank_matches_the_direct_computation_after_every_operation(
        self, operations, query_seed
    ) -> None:
        """Contract: V16 (property, risk R16).

        Operations reach: adding stamped and foreign rows, purging an existing
        row (which keeps its vector), and a fresh start that empties the
        table, in any order. After every single operation the bank's ranking
        is compared unguarded with the ranking computed straight from the
        database, so a version scheme that misses a change fails here.
        """
        Image.objects.all().delete()
        bank = _bank()
        query = _vector(query_seed)
        self.assertEqual(bank.rank(query)[0], _direct_ranking(query))
        for operation in operations:
            kind = operation[0]
            if kind == "add":
                _row(operation[1], stamp=STAMP if operation[2] else FOREIGN)
            elif kind == "purge":
                victim = Image.objects.filter(is_purged=False).order_by("content_hash").first()
                if victim is not None:
                    victim.is_purged = True
                    victim.save(update_fields=["is_purged"])
            else:
                Image.objects.all().delete()
            self.assertEqual(bank.rank(query)[0], _direct_ranking(query))
