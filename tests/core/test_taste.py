"""core.taste: the shared classifier plus one classifier per download source.

Contract (confirmed by the operator on 2026-10-03). V9 is checked in
tests/core/test_trainer.py (it needs the trainer's log lines), V10 in
tests/ratings/test_stats.py and V11 in tests/ratings/test_embedding_generation.py.

V1 Jede Quelle (`source_label`) ist eine eigene Bewertungskategorie. Die Kategorie
   eines Bildes steht beim Download fest und ändert sich nie.
V2 Eine Quelle bekommt genau dann ein eigenes Modell, wenn sie mindestens 10
   gemochte (Score ≥ 3) und 10 nicht gemochte (Score ≤ 2, Trash eingeschlossen)
   Bilder mit aktuellem Merkmal hat. 10/10 reicht, 9/10 nicht.
V3 Das eigene Modell einer Quelle wird nur aus den Bewertungen dieser Quelle
   gelernt. Bewertungen anderer Quellen ändern es nicht, weder beim Hinzufügen noch
   beim Entfernen.
V4 Das gemeinsame Modell wird wie bisher aus allen Bewertungen gelernt. Es urteilt
   über jedes Bild, dessen Quelle kein eigenes Modell hat, auch bei unbekanntem oder
   leerem Quellnamen.
V5 Die Vorhersage für ein Bild kommt vom Modell seiner eigenen Quelle, wenn es
   eins gibt, sonst vom gemeinsamen. Nie vom Modell einer anderen Quelle.
V6 In jedem Modell wiegen die gemochte und die nicht gemochte Seite insgesamt
   gleich viel, egal wie viele Bilder jede Seite hat. Innerhalb einer Seite gelten die
   Score-Gewichte weiter (0 und 5–6 dreifach, 1–4 einfach).
V7 Eine Gewichtsdatei von einem anderen Encoder oder im alten Format ohne Stempel
   wird nicht angewendet. Eine Datei mit beiden Encoder-Stempeln, aber ohne
   Quellenmodelle gilt weiter; alle Quellen nutzen dann das gemeinsame Modell bis
   zum nächsten Training. (Wording sharpened with V14: the original said "eine
   gestempelte Datei", and a file from before the SigLIP2 block carries one stamp
   of two; its 768-d hyperplane cannot judge a 1536-d feature.)
V8 Ein Training schreibt genau eine Geschmacks-Gewichtsdatei. Fresh Start löscht
   sie wie bisher.
V9 Das Training meldet pro Quelle in einer eigenen Logzeile, ob sie ein eigenes
   Modell bekommen hat, mit der Zahl gemochter und nicht gemochter Bilder. Eine Quelle
   ohne eigenes Modell nennt die Mindestzahl.
V10 Die Statistikseite zeigt pro Quelle die Zahl gemochter und nicht gemochter
   Bilder und ob die Quelle ein eigenes Modell hat.
V11 Nach dem Training wird jedes unbewertete Bild mit dem Modell seiner Quelle
   neu vorhergesagt.
V12 Das Geschmacksmerkmal eines Bildes besteht aus seinem DINOv3-Vektor und seinem
   SigLIP2-Vektor, beide auf Einheitslänge gebracht, in dieser Reihenfolge
   aneinandergehängt (1536 Werte). Kein Block wiegt durch seine Skala mehr als der
   andere.
V13 Ein Bild hat nur dann ein Geschmacksmerkmal, wenn beide Vektoren vorhanden und
   aus der aktuellen Generation sind. Fehlt einer, wird das Bild weder trainiert noch
   vorhergesagt und bleibt „unbewertet, trotzdem zeigen“.
   (tests/ratings/test_features.py, test_trainer.py, test_embedding_generation.py)
V14 Die Gewichtsdatei trägt beide Encoder-Stempel. Weicht einer ab, wird sie nicht
   angewendet.
V15 Das Training besorgt fehlende SigLIP2-Vektoren nur für bewertete Bilder
   selbst. Unbewertete bekommen sie allein von der Index-Kette. Ein Scrape kodiert nie
   SigLIP2 inline. (tests/core/test_trainer.py, tests/ratings/test_search.py)
V16 Die Klassifikation meldet, wie viele unbewertete Bilder wegen eines fehlenden
   Suchvektors noch keine Vorhersage bekommen haben.
   (tests/ratings/test_embedding_generation.py)
V17 Der NSFW-Kopf bleibt beim DINOv3-Vektor allein und ist unverändert.
   (tests/core/test_trainer.py)

Phase 3: 448 px und Neu-Kodierungs-Kette (tests in tests/core/test_brain_encoder.py,
tests/core/test_brain.py, tests/ratings/test_embedding_generation.py and
tests/ratings/test_taste_reencode.py)
V18 Der Encoder sieht jedes Bild in 448×448 Pixeln. Die Vektoren tragen einen
   neuen Stempel; jeder Vektor mit altem Stempel gilt ohne Migration als veraltet.
V19 Die Neu-Kodierung läuft als Kette von Scheiben zu 500 Bildern, bewertete
   zuerst, dann nach Downloadzeit. Sie endet, wenn nichts mehr veraltet ist oder eine
   Scheibe nichts kodieren konnte, und sagt im zweiten Fall, was zu tun ist. Zwischen
   zwei Scheiben ist die Warteschlange nie leer.
V20 Ein Scrape kodiert höchstens 200 veraltete Bestandsbilder inline und überlässt
   den Rest der Kette. Neue Downloads werden wie bisher beim Scrape kodiert. Die Kette
   startet nach jedem Scrape, wenn etwas veraltet ist, und per Knopf auf der
   Config-Seite.
V21 Die Config-Seite zeigt, wie viele Bilder einen aktuellen Geschmacksvektor
   haben. Der Job-Indikator zeigt die laufende Kette wie die Suchindex-Kette.
V22 Ein Training nach dem Encoder-Wechsel kodiert die bewerteten Bilder selbst neu
   und schreibt Geschmacks- und NSFW-Modell mit dem neuen Stempel.
"""

import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from loguru import logger
from sklearn.linear_model import LogisticRegression

from core import brain, siglip, taste

DIM = 8
GOOD_SCORES = (3, 4, 5, 6)
BAD_SCORES = (0, 1, 2)
# Mirror of the trainer's weight tables, by score.
WEIGHT_BY_SCORE = {0: 3.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 3.0, 6: 3.0}

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Corpus:
    """A synthetic training set: one row per rated image, grouped by source."""

    labels: list[str]
    X: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray
    good_by_source: dict[str, int]
    bad_by_source: dict[str, int]

    def without(self, label: str) -> Corpus:
        keep = [i for i, row_label in enumerate(self.labels) if row_label != label]
        return Corpus(
            labels=[self.labels[i] for i in keep],
            X=self.X[keep],
            y=self.y[keep],
            sample_weight=self.sample_weight[keep],
            good_by_source={k: v for k, v in self.good_by_source.items() if k != label},
            bad_by_source={k: v for k, v in self.bad_by_source.items() if k != label},
        )


labels_strategy = st.lists(
    st.text(min_size=0, max_size=6), unique=True, min_size=1, max_size=4
)


@st.composite
def corpora(draw) -> Corpus:
    """
    Build a corpus whose sources sit at different centres, each with its own
    liked/disliked direction, so every per-source fit is well posed. Reaches:
    sources below, exactly on and above the 10/10 threshold; sources whose
    disliked side is all trash; a single source; an empty label.
    """
    labels = draw(labels_strategy)
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    rows_labels: list[str] = []
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    w_rows: list[float] = []
    good_by_source: dict[str, int] = {}
    bad_by_source: dict[str, int] = {}
    for label in labels:
        good_count = draw(st.integers(0, 30))
        bad_count = draw(st.integers(0, 30))
        only_trash = draw(st.booleans())
        centre = rng.standard_normal(DIM) * 5
        direction = rng.standard_normal(DIM)
        for _ in range(good_count):
            score = draw(st.sampled_from(GOOD_SCORES))
            rows_labels.append(label)
            X_rows.append(centre + direction + rng.standard_normal(DIM) * 0.3)
            y_rows.append(1)
            w_rows.append(WEIGHT_BY_SCORE[score])
        for _ in range(bad_count):
            score = 0 if only_trash else draw(st.sampled_from(BAD_SCORES))
            rows_labels.append(label)
            X_rows.append(centre - direction + rng.standard_normal(DIM) * 0.3)
            y_rows.append(0)
            w_rows.append(WEIGHT_BY_SCORE[score])
        good_by_source[label] = good_count
        bad_by_source[label] = bad_count
    if not X_rows:
        X = np.zeros((0, DIM))
    else:
        X = np.vstack(X_rows)
    return Corpus(
        labels=rows_labels,
        X=X,
        y=np.array(y_rows, dtype=np.intp),
        sample_weight=np.array(w_rows, dtype=np.float64),
        good_by_source=good_by_source,
        bad_by_source=bad_by_source,
    )


# --- V2: the threshold -----------------------------------------------------------


@given(corpora())
def test_a_source_gets_its_own_model_iff_it_has_ten_of_each(corpus: Corpus) -> None:
    """Contract: V2"""
    groups = taste.group_by_source(corpus.labels, corpus.X, corpus.y, corpus.sample_weight)
    assert set(groups) == {label for label in corpus.labels}
    for label, group in groups.items():
        assert group.good_count == corpus.good_by_source[label]
        assert group.bad_count == corpus.bad_by_source[label]
        expected = group.good_count >= 10 and group.bad_count >= 10
        assert taste.has_enough_examples(group) is expected


@pytest.mark.parametrize(
    ("good", "bad", "expected"),
    [(10, 10, True), (9, 10, False), (10, 9, False), (30, 0, False), (0, 30, False)],
)
def test_threshold_is_inclusive_on_both_sides(good: int, bad: int, expected: bool) -> None:
    """Contract: V2 (10/10 reicht, 9/10 nicht)"""
    group = taste.SourceGroup(
        X=np.zeros((good + bad, DIM)),
        y=np.array([1] * good + [0] * bad),
        sample_weight=np.ones(good + bad),
        good_count=good,
        bad_count=bad,
    )
    assert taste.has_enough_examples(group) is expected


def test_trash_counts_as_disliked() -> None:
    """Contract: V2 (Trash eingeschlossen): ten trash rows are ten disliked rows."""
    labels = ["tg"] * 20
    y = np.array([1] * 10 + [0] * 10)
    weights = np.array([1.0] * 10 + [WEIGHT_BY_SCORE[0]] * 10)
    group = taste.group_by_source(labels, np.zeros((20, DIM)), y, weights)["tg"]
    assert group.bad_count == 10
    assert taste.has_enough_examples(group)


# --- V3: a source's model depends on that source only -------------------------------


@given(corpora())
@settings(deadline=None)
def test_other_sources_never_change_a_sources_own_model(corpus: Corpus) -> None:
    """Contract: V3 — fitting a source's model with and without another source's rows is identical."""
    groups = taste.group_by_source(corpus.labels, corpus.X, corpus.y, corpus.sample_weight)
    eligible = [label for label, group in groups.items() if taste.has_enough_examples(group)]
    others = [label for label in groups if label not in eligible] + eligible[1:]
    if not eligible or not others:
        # Fewer than two sources, or no source past the threshold: nothing to
        # compare. The corpus strategy reaches the other branch often enough.
        return
    target = eligible[0]
    removed = others[0]
    with_all = taste.fit_classifier(groups[target].X, groups[target].y, groups[target].sample_weight)
    smaller = corpus.without(removed)
    smaller_groups = taste.group_by_source(smaller.labels, smaller.X, smaller.y, smaller.sample_weight)
    without_other = taste.fit_classifier(
        smaller_groups[target].X, smaller_groups[target].y, smaller_groups[target].sample_weight
    )
    np.testing.assert_array_equal(with_all.coef_, without_other.coef_)
    np.testing.assert_array_equal(with_all.intercept_, without_other.intercept_)


def test_group_rows_follow_the_label_list_row_for_row() -> None:
    """Contract: V3 — row i of X, y and weights belongs to labels[i], never to a neighbour."""
    labels = ["a", "b", "a", "", "b"]
    X = np.arange(5 * DIM, dtype=float).reshape(5, DIM)
    y = np.array([1, 0, 0, 1, 1])
    weights = np.array([1.0, 3.0, 1.0, 3.0, 1.0])
    groups = taste.group_by_source(labels, X, y, weights)
    np.testing.assert_array_equal(groups["a"].X, X[[0, 2]])
    np.testing.assert_array_equal(groups["a"].y, [1, 0])
    np.testing.assert_array_equal(groups["a"].sample_weight, [1.0, 1.0])
    np.testing.assert_array_equal(groups["b"].X, X[[1, 4]])
    np.testing.assert_array_equal(groups[""].X, X[[3]])
    assert list(groups) == ["a", "b", ""]


# --- V4 / V5: which classifier judges which image ---------------------------------


@given(
    own=st.lists(st.text(max_size=6), unique=True, max_size=4),
    queries=st.lists(st.text(max_size=6), max_size=6),
)
def test_each_image_is_judged_by_its_own_source_or_the_shared_model(own, queries) -> None:
    """Contract: V4, V5"""
    shared = object()
    per_source = {label: object() for label in own}
    model = taste.TasteModel(shared=shared, per_source=per_source)
    for label in [*queries, "", *own]:
        chosen = model.classifier_for(label)
        if label in per_source:
            assert chosen is per_source[label]
        else:
            assert chosen is shared
        # Never another source's model.
        for other_label, other_clf in per_source.items():
            if other_label != label:
                assert chosen is not other_clf


# --- V6: balanced sides, weights kept inside a side ---------------------------------


weights_strategy = st.lists(st.sampled_from([1.0, 3.0]), min_size=2, max_size=60)


@given(weights=weights_strategy, data=st.data())
def test_both_sides_weigh_the_same_and_ratios_survive_inside_a_side(weights, data) -> None:
    """Contract: V6"""
    n = len(weights)
    y = np.array(data.draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)))
    if y.min() == y.max():
        # One class only is outside the function's domain (the trainer and the
        # threshold rule guarantee both classes); nothing to balance.
        return
    w = np.array(weights)
    balanced = taste.balance_class_weights(y, w)
    good_total = balanced[y == 1].sum()
    bad_total = balanced[y == 0].sum()
    assert good_total == pytest.approx(bad_total)
    # Conservation: balancing moves weight between the sides, never creates or destroys it.
    assert balanced.sum() == pytest.approx(w.sum())
    # Inside a side the score weights keep their ratios: w_i / w_j == b_i / b_j.
    for class_value in (0, 1):
        rows = np.flatnonzero(y == class_value)
        first = rows[0]
        for other in rows[1:]:
            assert balanced[other] * w[first] == pytest.approx(balanced[first] * w[other])
    # Symmetry: the two sides are treated alike, so swapping the labels changes nothing.
    np.testing.assert_allclose(taste.balance_class_weights(1 - y, w), balanced)


@given(bad_copies=st.integers(1, 25), good_weight=st.sampled_from([1.0, 3.0]))
@settings(deadline=None)
def test_a_lone_liked_image_is_not_outvoted_by_many_disliked_ones(bad_copies, good_weight) -> None:
    """Contract: V6 — with one liked row at +1 and many disliked rows at -1 the boundary stays at 0."""
    X = np.array([[1.0]] + [[-1.0]] * bad_copies)
    y = np.array([1] + [0] * bad_copies)
    weights = np.array([good_weight] + [1.0] * bad_copies)
    classifier = taste.fit_classifier(X, y, weights)
    midpoint = float(brain.predict_proba(classifier, np.array([0.0])))
    assert midpoint == pytest.approx(0.5, abs=0.02)


# --- V7: which files apply ----------------------------------------------------------


def _fitted() -> LogisticRegression:
    X = np.vstack([np.full(DIM, 1.0), np.full(DIM, -1.0)])
    return LogisticRegression().fit(X, [1, 0])


def _capture_warnings() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    return messages, sink_id


def test_bare_estimator_file_is_not_applied(tmp_path: Path) -> None:
    """Contract: V7 (altes Format ohne Stempel)"""
    path = tmp_path / "w.pkl"
    with path.open("wb") as f:
        pickle.dump(_fitted(), f)
    messages, sink_id = _capture_warnings()
    try:
        assert taste.load_taste_model(path) is None
    finally:
        logger.remove(sink_id)
    assert messages == [f"{path}: classifier predates encoder stamping; ignored until the next train run"]


def _write_stamped(path: Path, encoder: str, search_encoder: str, per_source: dict | None) -> None:
    payload = {"encoder": encoder, "search_encoder": search_encoder, "classifier": _fitted()}
    if per_source is not None:
        payload["per_source"] = per_source
    with path.open("wb") as f:
        pickle.dump(payload, f)


@given(stamp=st.text(max_size=20).filter(lambda s: s != brain.ENCODER_ID))
def test_file_from_another_encoder_is_not_applied(stamp: str) -> None:
    """Contract: V7 (anderer Encoder), V14"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.pkl"
        _write_stamped(path, stamp, siglip.SEARCH_ENCODER_ID, {"tg": _fitted()})
        messages, sink_id = _capture_warnings()
        try:
            assert taste.load_taste_model(path) is None
        finally:
            logger.remove(sink_id)
    assert messages == [
        f"{path}: classifier was trained on {stamp} but the encoder is {brain.ENCODER_ID}; "
        "ignored until the next train run"
    ]


@given(stamp=st.text(max_size=20).filter(lambda s: s != siglip.SEARCH_ENCODER_ID))
def test_file_from_another_search_encoder_is_not_applied(stamp: str) -> None:
    """Contract: V14 (the second block has its own stamp)"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.pkl"
        _write_stamped(path, brain.ENCODER_ID, stamp, {})
        messages, sink_id = _capture_warnings()
        try:
            assert taste.load_taste_model(path) is None
        finally:
            logger.remove(sink_id)
    assert messages == [
        f"{path}: classifier was trained on search encoder {stamp} but the search encoder is "
        f"{siglip.SEARCH_ENCODER_ID}; ignored until the next train run"
    ]


@given(encoder=st.text(max_size=20), search_encoder=st.text(max_size=20))
def test_a_file_applies_iff_both_stamps_are_current(encoder: str, search_encoder: str) -> None:
    """Contract: V14"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.pkl"
        _write_stamped(path, encoder, search_encoder, {})
        sink_id = logger.add(lambda message: None, level="WARNING")
        try:
            model = taste.load_taste_model(path)
        finally:
            logger.remove(sink_id)
    both_current = encoder == brain.ENCODER_ID and search_encoder == siglip.SEARCH_ENCODER_ID
    assert (model is not None) is both_current


def test_file_from_before_the_second_feature_block_is_not_applied(tmp_path: Path) -> None:
    """Contract: V14 — brain.save_classifier writes one stamp of two; such a file is foreign."""
    path = tmp_path / "w.pkl"
    brain.save_classifier(_fitted(), path)
    messages, sink_id = _capture_warnings()
    try:
        assert taste.load_taste_model(path) is None
    finally:
        logger.remove(sink_id)
    assert messages == [
        f"{path}: classifier was trained on search encoder None but the search encoder is "
        f"{siglip.SEARCH_ENCODER_ID}; ignored until the next train run"
    ]


def test_stamped_file_without_source_models_still_applies_as_shared_only(tmp_path: Path) -> None:
    """Contract: V7 (beide Stempel, keine Quellenmodelle): the shared model judges every source."""
    path = tmp_path / "w.pkl"
    _write_stamped(path, brain.ENCODER_ID, siglip.SEARCH_ENCODER_ID, per_source=None)
    model = taste.load_taste_model(path)
    assert model is not None
    assert model.per_source == {}
    assert model.classifier_for("tg") is model.shared
    assert model.classifier_for("") is model.shared


# --- V12: the feature -----------------------------------------------------------------


def _random_vector(seed: int, dim: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


@given(
    seed=st.integers(0, 2**31 - 1),
    taste_scale=st.floats(0.01, 100.0),
    search_scale=st.floats(0.01, 100.0),
)
def test_feature_is_unit_taste_block_then_unit_search_block(seed, taste_scale, search_scale) -> None:
    """Contract: V12"""
    taste_vector = _random_vector(seed, brain.EMBEDDING_DIM) * taste_scale
    search_vector = _random_vector(seed + 1, siglip.SEARCH_DIM) * search_scale
    feature = taste.combine_features(taste_vector, search_vector)

    assert feature.shape == (taste.FEATURE_DIM,) == (1536,)
    taste_block = feature[: brain.EMBEDDING_DIM]
    search_block = feature[brain.EMBEDDING_DIM :]
    assert np.linalg.norm(taste_block) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(search_block) == pytest.approx(1.0, abs=1e-5)
    # Order: the first block points where the taste vector points, the second where the search vector points.
    np.testing.assert_allclose(taste_block, taste_vector / np.linalg.norm(taste_vector), atol=1e-5)
    np.testing.assert_allclose(search_block, search_vector / np.linalg.norm(search_vector), atol=1e-5)
    # Scale never matters: the raw norm of either block is gone.
    np.testing.assert_allclose(
        taste.combine_features(taste_vector * 7, search_vector / 7), feature, atol=1e-5
    )
    # Swapping the blocks is a different feature (so the order is observable).
    swapped = taste.combine_features(search_vector, taste_vector)
    assert not np.allclose(swapped, feature, atol=1e-3)


@pytest.mark.parametrize("zero_side", ["taste", "search"])
def test_a_zero_vector_has_no_feature(zero_side: str) -> None:
    """Contract: V12 (R8) — a zero block cannot be brought to unit length."""
    taste_vector = np.zeros(brain.EMBEDDING_DIM) if zero_side == "taste" else _random_vector(1, brain.EMBEDDING_DIM)
    search_vector = np.zeros(siglip.SEARCH_DIM) if zero_side == "search" else _random_vector(2, siglip.SEARCH_DIM)
    with pytest.raises(ValueError, match=f"{zero_side} vector is all zeros"):
        taste.combine_features(taste_vector, search_vector)


@pytest.mark.parametrize(("taste_dim", "search_dim"), [(767, 768), (768, 512), (1536, 768)])
def test_a_block_of_the_wrong_size_is_refused(taste_dim: int, search_dim: int) -> None:
    """Contract: V12 — the feature is exactly 768 + 768 values, never a silently reshaped blob."""
    with pytest.raises(ValueError, match="must have"):
        taste.combine_features(_random_vector(1, taste_dim), _random_vector(2, search_dim))


def test_round_trip_keeps_every_model_and_the_shared_one_stays_readable_by_brain(tmp_path: Path) -> None:
    """Contract: V7, V8 — one file, both halves, old reader still sees the shared model."""
    path = tmp_path / "nested" / "w.pkl"
    shared = _fitted()
    tg = LogisticRegression().fit(np.vstack([np.full(DIM, -2.0), np.full(DIM, 2.0)]), [1, 0])
    taste.save_taste_model(taste.TasteModel(shared=shared, per_source={"tg": tg}), path)

    assert [p.name for p in path.parent.iterdir()] == ["w.pkl"]
    with path.open("rb") as f:
        stamps = {key: value for key, value in pickle.load(f).items() if key.endswith("encoder")}
    assert stamps == {"encoder": brain.ENCODER_ID, "search_encoder": siglip.SEARCH_ENCODER_ID}
    loaded = taste.load_taste_model(path)
    assert loaded is not None
    assert set(loaded.per_source) == {"tg"}
    probe = np.full((3, DIM), 1.5)
    np.testing.assert_allclose(loaded.shared.predict_proba(probe), shared.predict_proba(probe))
    np.testing.assert_allclose(loaded.per_source["tg"].predict_proba(probe), tg.predict_proba(probe))
    # The two halves really are different models, or the test proves nothing.
    assert not np.allclose(shared.predict_proba(probe), tg.predict_proba(probe))
    legacy_reader = brain.load_classifier(path)
    np.testing.assert_allclose(legacy_reader.predict_proba(probe), shared.predict_proba(probe))


# --- V6, V8, V12: shapes the mutation run showed were unpinned ------------------------


def test_feature_is_float32_even_from_float64_inputs() -> None:
    """Contract: V12 (the feature keeps the float32 storage format of both blocks; a float64 feature would double the trainer's matrix)"""
    rng = np.random.default_rng(3)
    feature = taste.combine_features(
        rng.standard_normal(brain.EMBEDDING_DIM), rng.standard_normal(siglip.SEARCH_DIM)
    )
    assert feature.dtype == np.float32
    assert feature.shape == (taste.FEATURE_DIM,)


def test_integer_weights_are_balanced_exactly_and_the_total_is_kept() -> None:
    """Contract: V6 (the weight table may arrive as integers; balancing must not truncate them)"""
    balanced = taste.balance_class_weights(np.array([0, 0, 1]), np.array([1, 1, 3]))
    np.testing.assert_allclose(balanced, [1.25, 1.25, 2.5])
    assert balanced.sum() == 5.0


def test_fit_gets_a_thousand_iterations() -> None:
    """Contract: V6 (same estimator everywhere: 1536-d features on a few thousand rows need more than sklearn's 100 lbfgs steps)"""
    X = np.vstack([np.full(4, 0.1), np.full(4, 0.9)])
    classifier = taste.fit_classifier(X, np.array([0, 1]), np.array([1.0, 1.0]))
    assert classifier.get_params()["max_iter"] == 1000


def test_save_taste_model_creates_missing_parent_directories(tmp_path: Path) -> None:
    """Contract: V8 (a fresh DATA_DIR has no weights folder yet; the first train run must not fail on it)"""
    X = np.vstack([np.full(taste.FEATURE_DIM, 0.1), np.full(taste.FEATURE_DIM, 0.9)])
    model = taste.TasteModel(shared=LogisticRegression().fit(X, [0, 1]))
    path = tmp_path / "a" / "b" / "weights.pkl"
    taste.save_taste_model(model, path)
    loaded = taste.load_taste_model(path)
    assert loaded is not None
    assert loaded.per_source == {}


# --- V1: the category is written once, at download -----------------------------------


def test_source_label_is_written_only_when_the_row_is_created() -> None:
    """Contract: V1 — no code path assigns or updates source_label after Image.objects.create."""
    offenders = []
    for path in list((REPO_ROOT / "ratings").glob("*.py")) + list((REPO_ROOT / "core").glob("*.py")):
        if path.name == "models.py":
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            assigns = re.search(r"\.source_label\s*=[^=]", line)
            updates = re.search(r"update\([^)]*source_label\s*=", line)
            if assigns or updates:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")
    assert offenders == []
    admin_source = (REPO_ROOT / "ratings" / "admin.py").read_text()
    readonly_lists = re.findall(r"readonly_fields\s*=\s*\((.*?)\)", admin_source, re.S)
    assert any('"source_label"' in fields for fields in readonly_lists), "the admin must not edit the category"
