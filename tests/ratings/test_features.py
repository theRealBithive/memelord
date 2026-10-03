"""ratings.features: the taste feature read from an Image row.

Taste contract (confirmed by the operator on 2026-10-03; the full list V1–V17
lives in tests/core/test_taste.py):

V12 Das Geschmacksmerkmal eines Bildes besteht aus seinem DINOv3-Vektor und seinem
   SigLIP2-Vektor, beide auf Einheitslänge gebracht, in dieser Reihenfolge
   aneinandergehängt (1536 Werte). Kein Block wiegt durch seine Skala mehr als der
   andere.
V13 Ein Bild hat nur dann ein Geschmacksmerkmal, wenn beide Vektoren vorhanden und
   aus der aktuellen Generation sind. Fehlt einer, wird das Bild weder trainiert noch
   vorhergesagt und bleibt „unbewertet, trotzdem zeigen“.
"""

import os
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from core import brain, siglip, taste
from ratings import features
from ratings.models import Image


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(768).astype(np.float32)


def _unsaved_row(*, taste_stamp: str | None, search_stamp: str | None) -> Image:
    """An Image instance (not saved: the gates read fields only). None = no vector at all."""
    return Image(
        content_hash=uuid.uuid4().hex,
        file_path="images/x.png",
        source_label="t",
        embedding=brain.embedding_to_bytes(_vector(1)) if taste_stamp is not None else None,
        embedding_model=taste_stamp or "",
        search_embedding=brain.embedding_to_bytes(_vector(2)) if search_stamp is not None else None,
        search_embedding_model=search_stamp or "",
    )


stamp_or_missing = st.one_of(st.none(), st.text(max_size=20))


# No row is saved, so these are plain functions: the gates read fields only.


@given(taste_stamp=stamp_or_missing, search_stamp=stamp_or_missing)
def test_a_row_has_a_feature_iff_both_vectors_are_current(taste_stamp, search_stamp) -> None:
    """Contract: V13"""
    row = _unsaved_row(taste_stamp=taste_stamp, search_stamp=search_stamp)
    expected = taste_stamp == brain.ENCODER_ID and search_stamp == siglip.SEARCH_ENCODER_ID
    assert features.has_taste_features(row) is expected


def test_the_four_corners() -> None:
    """Contract: V13 — present/current, present/stale, missing, for each half."""
    current, stale = brain.ENCODER_ID, "dinov2_vitb14"
    search_current, search_stale = siglip.SEARCH_ENCODER_ID, "siglip_old"
    assert features.has_taste_features(_unsaved_row(taste_stamp=current, search_stamp=search_current))
    assert not features.has_taste_features(_unsaved_row(taste_stamp=stale, search_stamp=search_current))
    assert not features.has_taste_features(_unsaved_row(taste_stamp=current, search_stamp=search_stale))
    assert not features.has_taste_features(_unsaved_row(taste_stamp=None, search_stamp=search_current))
    assert not features.has_taste_features(_unsaved_row(taste_stamp=current, search_stamp=None))
    assert not features.has_taste_features(_unsaved_row(taste_stamp=None, search_stamp=None))


def test_the_feature_is_the_combination_of_the_rows_two_blobs() -> None:
    """Contract: V12 — DINOv3 blob first, SigLIP2 blob second, through combine_features."""
    row = _unsaved_row(taste_stamp=brain.ENCODER_ID, search_stamp=siglip.SEARCH_ENCODER_ID)
    feature = features.taste_features(row)
    np.testing.assert_allclose(feature, taste.combine_features(_vector(1), _vector(2)))
    assert feature.shape == (taste.FEATURE_DIM,)
