"""
The NSFW mark moves an image between rating categories.

Taste contract (tests/core/test_taste.py):
V26 Wechselt die NSFW-Markierung eines Bildes (Taste `n` im Review, Lightbox), wird
   seine Vorhersage sofort mit dem Modell der neuen Kategorie erneuert. Fehlt Merkmal
   oder Modell, wird die alte Vorhersage verworfen und das Bild gilt als "unbewertet,
   trotzdem zeigen".

Two models that disagree on the one feature in play: the shared one calls it
good, the "(nsfw)" one calls it bad. A flip therefore has to carry the stored
score across 0.5, which no stale value can fake. The source "tg" has no model
of its own, so a cleared mark lands on the shared model.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import django
import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from sklearn.linear_model import LogisticRegression

from core import brain, siglip, taste
from ratings.models import Image
from ratings.views import common

PLUS = np.full(768, 1.0, dtype=np.float32)


def _save_disagreeing_models(path: Path) -> None:
    feature = taste.combine_features(PLUS, PLUS)
    shared = LogisticRegression().fit(np.vstack([feature, -feature]), [1, 0])
    nsfw_model = LogisticRegression().fit(np.vstack([feature, -feature]), [0, 1])
    taste.save_taste_model(
        taste.TasteModel(shared=shared, per_source={taste.NSFW_GROUP: nsfw_model}), path
    )


def _image(*, is_nsfw: bool, predicted_score: float, search: bool = True) -> Image:
    """An unrated tg row with current vectors (the +3 probe), so a prediction is possible."""
    h = uuid.uuid4().hex
    probe = brain.embedding_to_bytes(PLUS * 3)
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.png",
        source_label="tg",
        is_nsfw=is_nsfw,
        predicted_score=predicted_score,
        embedding=probe,
        embedding_model=brain.ENCODER_ID,
        search_embedding=probe if search else None,
        search_embedding_model=siglip.SEARCH_ENCODER_ID if search else "",
    )


@override_settings(DEBUG=True)
class NsfwFlipRepredictionTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        user = get_user_model().objects.create_user(f"flip-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)
        self._tmp = tempfile.TemporaryDirectory()
        self.weights = Path(self._tmp.name) / "w.pkl"
        _save_disagreeing_models(self.weights)
        self._weights_patch = mock.patch.object(common, "WEIGHTS_PATH", self.weights)
        self._weights_patch.start()

    def tearDown(self) -> None:
        self._weights_patch.stop()
        self._tmp.cleanup()

    def test_marking_nsfw_in_review_hands_the_image_to_the_nsfw_model(self) -> None:
        """Contract: V26 — the review queue branch of toggle_nsfw."""
        image = _image(is_nsfw=False, predicted_score=0.9)
        _image(is_nsfw=False, predicted_score=0.9)  # a neighbour for the card to advance to

        self.client.post(reverse("toggle_nsfw", args=[image.content_hash]), {"mode": "corpus"})

        image.refresh_from_db()
        self.assertTrue(image.is_nsfw)
        self.assertLess(image.predicted_score, 0.5)

    def test_clearing_the_mark_in_the_nsfw_queue_hands_it_back_to_its_source(self) -> None:
        """Contract: V26 — the NSFW queue branch of toggle_nsfw; tg has no model, so the shared one judges."""
        image = _image(is_nsfw=True, predicted_score=0.1)
        _image(is_nsfw=True, predicted_score=0.1)

        self.client.post(reverse("toggle_nsfw", args=[image.content_hash]), {"mode": "nsfw_corpus"})

        image.refresh_from_db()
        self.assertFalse(image.is_nsfw)
        self.assertGreater(image.predicted_score, 0.5)

    def test_the_lightbox_flip_renews_the_prediction_the_same_way(self) -> None:
        """Contract: V26 — the gallery lightbox path."""
        image = _image(is_nsfw=False, predicted_score=0.9)

        self.client.post(reverse("lightbox_nsfw", args=[image.content_hash]))

        image.refresh_from_db()
        self.assertTrue(image.is_nsfw)
        self.assertLess(image.predicted_score, 0.5)

    def test_without_a_feature_the_stale_score_is_dropped_and_nothing_replaces_it(self) -> None:
        """Contract: V26 — no SigLIP2 half: the old score goes, the row reads "show anyway"."""
        image = _image(is_nsfw=False, predicted_score=0.9, search=False)
        _image(is_nsfw=False, predicted_score=0.9)

        self.client.post(reverse("toggle_nsfw", args=[image.content_hash]), {"mode": "corpus"})

        image.refresh_from_db()
        self.assertTrue(image.is_nsfw)
        self.assertIsNone(image.predicted_score)
