"""
Similar images over the DINOv3 taste vectors.

Contract (full list in tests/ratings/test_search.py):

V2  Such-Vektoren und Taste-Vektoren werden nie miteinander verglichen oder
    vermischt. „Ähnliche Bilder" rechnet ausschließlich mit DINOv3-Vektoren
    des aktuellen Encoders.
V17 Jedes nicht gelöschte Bild mit aktuellem Taste-Vektor bietet „ähnliche
    Bilder": die ähnlichsten anderen Bilder nach Kosinus über DINOv3, höchste
    zuerst, höchstens 100, ohne das Bild selbst, mit derselben Kandidatenmenge
    und denselben Filtern wie die Textsuche (Umfang Standard hier „all").
    Erreichbar aus der Review-Karte und aus der Galerie-Lightbox. Ein Bild
    ohne aktuellen Taste-Vektor sagt das, statt eine leere Liste zu zeigen.
"""

from __future__ import annotations

import os
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

import numpy as np
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from core import brain, siglip
from ratings import search, similar
from ratings.models import Image

CURRENT = brain.ENCODER_ID


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(768).astype(np.float32)


def _row(
    seed: int | None,
    *,
    score: int | None = 3,
    nsfw: bool = False,
    purged: bool = False,
    stamp: str = CURRENT,
    search_seed: int | None = None,
) -> Image:
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.png",
        source_label="t",
        score=score,
        is_nsfw=nsfw,
        is_purged=purged,
        embedding=brain.embedding_to_bytes(_vector(seed)) if seed is not None else None,
        embedding_model=stamp if seed is not None else "",
        search_embedding=(
            brain.embedding_to_bytes(_vector(search_seed)) if search_seed is not None else None
        ),
        search_embedding_model=siglip.SEARCH_ENCODER_ID if search_seed is not None else "",
    )


def _all(show_nsfw: bool = True):
    return search.candidate_images("all", 1, "", show_nsfw)


class RankSimilarTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        similar.TASTE_BANK.reset()

    def test_anchor_without_a_current_vector_says_so(self) -> None:
        """Contract: V17"""
        stale = _row(1, stamp="dinov2_vitb14")
        _row(2)
        self.assertIsNone(similar.rank_similar(stale, _all(), 10))

    def test_anchor_is_excluded_but_a_true_duplicate_ranks_first(self) -> None:
        """Contract: V17 (risk R18)"""
        anchor = _row(5)
        twin = _row(5)
        others = [_row(seed) for seed in (6, 7)]
        ranked = similar.rank_similar(anchor, _all(), 10)
        hashes = [h for h, _ in ranked]
        self.assertNotIn(anchor.content_hash, hashes)
        self.assertEqual(hashes[0], twin.content_hash)
        self.assertAlmostEqual(ranked[0][1], 1.0, places=5)
        self.assertEqual(set(hashes), {twin.content_hash, *(o.content_hash for o in others)})

    def test_results_follow_the_candidate_filters_and_the_similarity_order(self) -> None:
        """Contract: V17, V5"""
        anchor = _row(1)
        rated = [_row(seed, score=4) for seed in (2, 3, 4)]
        unrated = _row(8, score=None)
        hidden_nsfw = _row(9, nsfw=True)
        purged = _row(10, purged=True)
        trash = _row(11, score=0)

        ranked = similar.rank_similar(anchor, _all(show_nsfw=False), 10)
        hashes = [h for h, _ in ranked]
        sims = [s for _, s in ranked]

        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertEqual(set(hashes), {r.content_hash for r in rated} | {unrated.content_hash})
        for excluded in (hidden_nsfw, purged, trash, anchor):
            self.assertNotIn(excluded.content_hash, hashes)
        rated_only = similar.rank_similar(anchor, search.candidate_images("rated", 1, "", False), 10)
        self.assertNotIn(unrated.content_hash, [h for h, _ in rated_only])

    def test_only_taste_vectors_take_part_never_search_vectors(self) -> None:
        """Contract: V2"""
        anchor = _row(1)
        search_only = _row(None, search_seed=1)
        taste_too = _row(2, search_seed=1)
        hashes = [h for h, _ in similar.rank_similar(anchor, _all(), 10)]
        self.assertNotIn(search_only.content_hash, hashes)
        self.assertIn(taste_too.content_hash, hashes)

    def test_an_anchor_with_no_other_candidate_gives_an_empty_list(self) -> None:
        """Contract: V17 (empty, not None: the anchor itself is fine, there is just nobody near it)"""
        anchor = _row(1)
        self.assertEqual(similar.rank_similar(anchor, _all(), 10), [])

    def test_limit_caps_the_page(self) -> None:
        """Contract: V17"""
        anchor = _row(1)
        for seed in range(2, 8):
            _row(seed)
        self.assertEqual(len(similar.rank_similar(anchor, _all(), 3)), 3)


@override_settings(DEBUG=True)
class SimilarViewTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        similar.TASTE_BANK.reset()
        user = get_user_model().objects.create_user(f"sim-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def test_similar_mode_renders_the_neighbours_without_the_anchor(self) -> None:
        """Contract: V17"""
        anchor = _row(1, score=None)
        neighbour = _row(2, score=None)
        html = self.client.get(reverse("gallery"), {"similar": anchor.content_hash, "scope": "all"}).content.decode()
        self.assertIn("similar to", html)
        self.assertIn(f'id="item-{neighbour.content_hash}"', html)
        self.assertNotIn(f'id="item-{anchor.content_hash}"', html)
        self.assertIn("1 similar image", html)
        self.assertNotIn("sort=random", html)

    def test_unknown_or_purged_anchor_is_a_404(self) -> None:
        """Contract: V8"""
        purged = _row(1, purged=True)
        self.assertEqual(self.client.get(reverse("gallery"), {"similar": "nope"}).status_code, 404)
        self.assertEqual(
            self.client.get(reverse("gallery"), {"similar": purged.content_hash}).status_code, 404
        )

    def test_anchor_without_vector_falls_back_to_the_gallery_with_a_message(self) -> None:
        """Contract: V17"""
        stale = _row(1, stamp="dinov2_vitb14", score=4)
        response = self.client.get(reverse("gallery"), {"similar": stale.content_hash})
        html = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn("no current embedding yet", html)
        self.assertNotIn("similar to", html)
        self.assertIn(f'id="item-{stale.content_hash}"', html)

    def test_lightbox_and_review_card_link_to_similar_images(self) -> None:
        """Contract: V17 (risk R20)"""
        image = _row(1, score=None)
        link = f"?similar={image.content_hash}&amp;scope=all"
        lightbox = self.client.get(reverse("lightbox", args=[image.content_hash])).content.decode()
        review = self.client.get(reverse("review_corpus_image", args=[image.content_hash])).content.decode()
        self.assertIn(link, lightbox)
        self.assertIn(link, review)
