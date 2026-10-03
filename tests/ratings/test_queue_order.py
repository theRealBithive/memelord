"""
Review queue order (Config → Review queue → Order).

Contract: V1 Es gibt genau eine Einstellung für die Review-Reihenfolge mit vier
             Werten: älteste zuerst, neueste zuerst, gemischt, unsicherste
             zuerst. Sie gilt für die SFW- und die NSFW-Review-Queue
             gleichermaßen. (Reworded on 2026-10-03 from three values.)
Contract: V2 Frische Installationen und bestehende Datenbanken verhalten sich
             wie bisher (älteste zuerst), bis der Operator die Einstellung
             ändert.
Contract: V3 In jedem Modus kommt ein Bild, das noch nie in der Review angezeigt
             wurde, vor jedem Bild, das schon angezeigt und übersprungen wurde.
Contract: V4 Übersprungene Bilder behalten in jedem Modus ihre Reihenfolge nach
             dem Zeitpunkt, an dem sie zuerst gesehen wurden.
Contract: V5 Solange nichts bewertet, gelöscht oder gescrapt wird, ist die Queue
             zwischen zwei Anfragen dieselbe feste Folge. Next und Prev sind die
             Nachbarn in dieser Folge, und Prev von Next ist wieder das
             Ausgangsbild.
Contract: V6 "n / total" ist der Rang des Bilds in dieser Folge. Anzahl davor
             + 1 + Anzahl danach ist für jedes Bild die Queue-Größe.
Contract: V7 Bewerten oder Löschen springt in jedem Modus zum Nachfolger, am
             Ende zum Vorgänger, und bei leerer Queue auf die Leer-Ansicht.
Contract: V8 Die gemischte Reihenfolge hängt nur vom Bildinhalt ab: ändern sich
             die Downloadzeiten, ändert sich die gemischte Folge nicht. Sie ist
             über Anfragen, Sessions und Neustarts hinweg stabil.
Contract: V9 Die Einstellung wird auf der Config-Seite gesetzt und in der
             Datenbank gespeichert. Ein unbekannter oder fehlender Wert wird als
             "älteste zuerst" gespeichert und erreicht die Abfrageschicht nie.
Contract: V10 Die Einstellung ändert nur die Reihenfolge, nie die Menge: welche
             Bilder in der Queue sind, bestimmen weiterhin Scores, Schwellen und
             der NSFW-Schalter.
Contract: V11 „Unsicherste zuerst" ordnet nach dem Abstand der Vorhersage von
             50 %: je näher an 50 %, desto früher; je sicherer das Modell (nahe
             0 % oder 100 %), desto später. Bilder ohne Vorhersage kommen nach
             allen Bildern mit Vorhersage. Ungesehene bleiben vor Gesehenen
             (V3), die Menge bleibt dieselbe (V10). Die Folge ändert sich nur,
             wenn sich Vorhersagen ändern (Scrape, Training, NSFW-Wechsel).
Contract: V12 Gleiche Sortierwerte brechen die Folge nicht: Gleichstände werden
             in jeder Reihenfolge nach dem Bildinhalt (Hash) aufgelöst. Die
             Folge ist dadurch eindeutig und stabil, und Prev/Next, Position
             und Bewerten-und-Weiter gelten auch innerhalb solcher Blöcke
             (V5–V7).
(V11 and V12 confirmed by the operator on 2026-10-03.)

Generator: `queue_specs` builds 1–7 images with distinct content hashes,
download times drawn from six minutes so ties are common (V12), a distinct
first-seen time for a random subset, an NSFW flag, an optional score and an
optional predicted score drawn from None, the exact values 0.25 / 0.5 / 0.75
(equal uncertainties for different predictions, V12) and any float in [0, 1].
Hash order, download order, seen order and prediction order are drawn
independently of each other, so the examples reach the states in which the
four orders actually differ, plus all-unseen, all-seen, single-image,
partly-scored, all-unpredicted and tied queues. The tests that need a full
queue pin the thresholds to 1 and open NSFW; the membership test draws
thresholds and the NSFW switch too.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.django import TestCase as HypothesisTestCase

from ratings.models import Image, ReviewThresholds
from ratings.queue_rules import (
    QUEUE_ORDERS,
    get_queue_order,
    normalize_queue_order,
    review_settings_row,
)
from ratings.views.review import (
    _browse_ctx,
    _queue_neighbor_hash,
    _review_nsfw_qs,
    _review_qs,
)

OLDEST = ReviewThresholds.ORDER_OLDEST
NEWEST = ReviewThresholds.ORDER_NEWEST
SHUFFLE = ReviewThresholds.ORDER_SHUFFLE
UNCERTAIN = ReviewThresholds.ORDER_UNCERTAIN
ALL_ORDERS = [OLDEST, NEWEST, SHUFFLE, UNCERTAIN]

BASE_TIME = timezone.make_aware(datetime(2026, 1, 1, 12, 0, 0))

HEX64 = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)


@dataclass(frozen=True)
class ImageSpec:
    content_hash: str
    download_minutes: int
    seen_minutes: int | None
    is_nsfw: bool
    score: int | None
    predicted: float | None


@st.composite
def queue_specs(draw) -> list[ImageSpec]:
    n = draw(st.integers(min_value=1, max_value=7))
    hashes = draw(st.lists(HEX64, min_size=n, max_size=n, unique=True))
    download_minutes = draw(st.lists(st.integers(0, 5), min_size=n, max_size=n))
    seen_minutes = draw(
        st.lists(st.integers(0, 100_000), min_size=n, max_size=n, unique=True)
    )
    seen_flags = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    nsfw_flags = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    scores = draw(
        st.lists(st.one_of(st.none(), st.integers(0, 6)), min_size=n, max_size=n)
    )
    predictions = draw(
        st.lists(
            st.one_of(st.none(), st.sampled_from([0.25, 0.5, 0.75]), st.floats(0.0, 1.0)),
            min_size=n,
            max_size=n,
        )
    )
    specs = []
    for i in range(n):
        seen = seen_minutes[i] if seen_flags[i] else None
        specs.append(
            ImageSpec(
                hashes[i], download_minutes[i], seen, nsfw_flags[i], scores[i],
                predictions[i],
            )
        )
    return specs


def _wipe() -> None:
    Image.objects.all().delete()
    ReviewThresholds.objects.all().delete()


def _set_order(order_name: str, sfw: int = 1, nsfw: int = 1) -> None:
    ReviewThresholds.objects.update_or_create(
        pk=1,
        defaults={
            "sfw_threshold": sfw,
            "nsfw_threshold": nsfw,
            "queue_order": order_name,
        },
    )


def _create(specs: list[ImageSpec]) -> None:
    for spec in specs:
        Image.objects.create(
            content_hash=spec.content_hash,
            file_path=f"images/{spec.content_hash}.jpg",
            source_label="test",
            is_nsfw=spec.is_nsfw,
            score=spec.score,
            predicted_score=spec.predicted,
        )
        seen_at = None
        if spec.seen_minutes is not None:
            seen_at = BASE_TIME + timedelta(minutes=spec.seen_minutes)
        # downloaded_at is auto_now_add, so it can only be set after the insert.
        Image.objects.filter(pk=spec.content_hash).update(
            downloaded_at=BASE_TIME + timedelta(minutes=spec.download_minutes),
            queue_seen_at=seen_at,
        )


def _sequence(qs) -> list[str]:
    return list(qs.values_list("content_hash", flat=True))


def _download_order(specs: list[ImageSpec], newest_first: bool) -> list[str]:
    """Oracle for V1/V3/V4/V12 in the two download orders, written from the contract."""
    unseen = [s for s in specs if s.seen_minutes is None]
    seen = [s for s in specs if s.seen_minutes is not None]
    if newest_first:
        unseen_sorted = sorted(unseen, key=lambda s: (-s.download_minutes, s.content_hash))
    else:
        unseen_sorted = sorted(unseen, key=lambda s: (s.download_minutes, s.content_hash))
    seen_sorted = sorted(seen, key=lambda s: s.seen_minutes)
    return [s.content_hash for s in unseen_sorted + seen_sorted]


def _uncertainty(predicted: float | None) -> float:
    """Oracle for V11: distance from the coin flip; an image without a prediction sorts after every predicted one."""
    if predicted is None:
        return 1.0
    return abs(predicted - 0.5)


def _uncertain_order(specs: list[ImageSpec]) -> list[str]:
    """Oracle for V11/V12: least certain first, hash on ties, seen block as in every order."""
    unseen = [s for s in specs if s.seen_minutes is None]
    seen = [s for s in specs if s.seen_minutes is not None]
    unseen_sorted = sorted(unseen, key=lambda s: (_uncertainty(s.predicted), s.content_hash))
    seen_sorted = sorted(seen, key=lambda s: s.seen_minutes)
    return [s.content_hash for s in unseen_sorted + seen_sorted]


class QueueOrderProperties(HypothesisTestCase):
    """Contract V1, V3, V4, V5, V6, V8, V10 over generated queues."""

    @settings(max_examples=40, deadline=None)
    @given(specs=queue_specs())
    def test_navigation_follows_one_fixed_sequence_in_every_order(self, specs):
        """Contract: V1, V3, V4, V5, V6, V11, V12.

        For every order the queue is one sequence; prev/next of every image
        are its neighbours in that sequence, the position is its rank, and
        rank-before + 1 + rank-after is the queue size for every image (the
        unguarded conservation law). Unseen images precede seen ones, seen
        ones keep their first-seen order, and the download orders and the
        uncertain order match the contract's oracles, ties included.
        """
        _wipe()
        _create(specs)
        unscored = sorted(s.content_hash for s in specs if s.score is None)
        by_hash = {s.content_hash: s for s in specs}

        for order_name in ALL_ORDERS:
            _set_order(order_name)
            order = get_queue_order()
            qs = _review_qs(show_nsfw=True, order=order)
            sequence = _sequence(qs)

            self.assertEqual(sorted(sequence), unscored)
            self.assertEqual(sequence, _sequence(qs), "stable between two requests")

            seen_positions = [
                i for i, h in enumerate(sequence) if by_hash[h].seen_minutes is not None
            ]
            unseen_positions = [
                i for i, h in enumerate(sequence) if by_hash[h].seen_minutes is None
            ]
            if seen_positions and unseen_positions:
                self.assertLess(max(unseen_positions), min(seen_positions))
            seen_times = [by_hash[sequence[i]].seen_minutes for i in seen_positions]
            self.assertEqual(seen_times, sorted(seen_times))

            if order_name == OLDEST:
                expected = [h for h in _download_order(specs, False) if h in unscored]
                self.assertEqual(sequence, expected)
            if order_name == NEWEST:
                expected = [h for h in _download_order(specs, True) if h in unscored]
                self.assertEqual(sequence, expected)
            if order_name == UNCERTAIN:
                expected = [h for h in _uncertain_order(specs) if h in unscored]
                self.assertEqual(sequence, expected)
                predicted = [h for h in sequence if by_hash[h].seen_minutes is None and by_hash[h].predicted is not None]
                unpredicted = [h for h in sequence if by_hash[h].seen_minutes is None and by_hash[h].predicted is None]
                if predicted and unpredicted:
                    self.assertLess(sequence.index(predicted[-1]), sequence.index(unpredicted[0]))

            total = len(sequence)
            for i, content_hash in enumerate(sequence):
                ctx = _browse_ctx(qs, content_hash, "corpus", True, None, order)
                expected_prev = sequence[i - 1] if i > 0 else None
                expected_next = sequence[i + 1] if i + 1 < total else None
                self.assertEqual(ctx["prev_hash"], expected_prev)
                self.assertEqual(ctx["next_hash"], expected_next)
                self.assertEqual(ctx["position"], i + 1)
                self.assertEqual(ctx["total"], total)

                image = Image.objects.get(pk=content_hash)
                advance_target = _queue_neighbor_hash(qs, image, order)
                if expected_next is not None:
                    self.assertEqual(advance_target, expected_next)
                else:
                    self.assertEqual(advance_target, expected_prev)

    @settings(max_examples=40, deadline=None)
    @given(specs=queue_specs())
    def test_nsfw_queue_uses_the_same_order(self, specs):
        """Contract: V1.

        The NSFW queue is the SFW+NSFW queue restricted to NSFW images, in the
        same sequence, for every order.
        """
        _wipe()
        _create(specs)
        nsfw_hashes = {s.content_hash for s in specs if s.is_nsfw}
        for order_name in ALL_ORDERS:
            _set_order(order_name)
            order = get_queue_order()
            full = _sequence(_review_qs(show_nsfw=True, order=order))
            nsfw_only = _sequence(_review_nsfw_qs(order=order))
            self.assertEqual(nsfw_only, [h for h in full if h in nsfw_hashes])

    @settings(max_examples=40, deadline=None)
    @given(specs=queue_specs())
    def test_shuffle_ignores_download_times(self, specs):
        """Contract: V8.

        Rotating the download times among the images leaves the shuffled
        sequence unchanged, so it can only depend on the content and on the
        first-seen stamps.
        """
        _wipe()
        _create(specs)
        _set_order(SHUFFLE)
        before = _sequence(_review_qs(show_nsfw=True))

        rotated = specs[1:] + specs[:1]
        for spec, donor in zip(specs, rotated, strict=True):
            Image.objects.filter(pk=spec.content_hash).update(
                downloaded_at=BASE_TIME + timedelta(minutes=donor.download_minutes)
            )
        after = _sequence(_review_qs(show_nsfw=True))
        self.assertEqual(after, before)

    @settings(max_examples=40, deadline=None)
    @given(
        specs=queue_specs(),
        show_nsfw=st.booleans(),
        sfw=st.integers(1, 6),
        nsfw=st.integers(1, 6),
    )
    def test_order_changes_the_sequence_but_never_the_membership(
        self, specs, show_nsfw, sfw, nsfw
    ):
        """Contract: V10.

        Under any thresholds and NSFW switch, the set of queued images is the
        same for every order, in both review queues.
        """
        _wipe()
        _create(specs)
        memberships = set()
        nsfw_memberships = set()
        for order_name in ALL_ORDERS:
            _set_order(order_name, sfw=sfw, nsfw=nsfw)
            memberships.add(frozenset(_sequence(_review_qs(show_nsfw=show_nsfw))))
            nsfw_memberships.add(frozenset(_sequence(_review_nsfw_qs())))
        self.assertEqual(len(memberships), 1)
        self.assertEqual(len(nsfw_memberships), 1)


@given(
    value=st.one_of(
        st.none(), st.text(), st.sampled_from(ALL_ORDERS), st.integers(), st.lists(st.text())
    )
)
def test_normalize_queue_order_only_ever_returns_a_known_order(value):
    """Contract: V9.

    Whatever arrives (POST field, hand-edited DB cell, wrong type) maps to a
    table entry; known names map to themselves, everything else to oldest.
    """
    result = normalize_queue_order(value)
    assert result in QUEUE_ORDERS
    if value in ALL_ORDERS:
        assert result == value
    else:
        assert result == OLDEST


def _login(client: Client) -> None:
    username = f"qo_{uuid.uuid4().hex[:8]}"
    User.objects.create_user(username, password="secret")
    client.login(username=username, password="secret")


# Three images whose hash order, oldest-first order, newest-first order and
# uncertain-first order are four different sequences, so each order is told
# apart from the others: a 0.7 (0.2 from the flip), b 0.5 (0.0), c 0.9 (0.4).
MIXED_SET = [
    ImageSpec("a" * 64, 0, None, False, None, 0.7),
    ImageSpec("c" * 64, 10, None, False, None, 0.9),
    ImageSpec("b" * 64, 20, None, False, None, 0.5),
]


@override_settings(DEBUG=True)
class QueueOrderViewTests(TestCase):
    """Contract V2, V7, V9 through the config form and the review views."""

    def setUp(self) -> None:
        _wipe()
        self.client = Client()
        _login(self.client)

    def test_fresh_database_serves_oldest_first(self) -> None:
        """Contract: V2."""
        self.assertIs(get_queue_order(), QUEUE_ORDERS[OLDEST])
        self.assertEqual(review_settings_row().queue_order, OLDEST)
        row = ReviewThresholds(sfw_threshold=3, nsfw_threshold=3)
        self.assertEqual(row.queue_order, OLDEST)

    def test_the_four_orders_are_four_different_sequences(self) -> None:
        """Contract: V1, V11."""
        _create(MIXED_SET)
        sequences = {}
        for order_name in ALL_ORDERS:
            _set_order(order_name)
            sequences[order_name] = tuple(_sequence(_review_qs()))
        self.assertEqual(len(set(sequences.values())), 4)
        self.assertEqual(sequences[OLDEST], ("a" * 64, "c" * 64, "b" * 64))
        self.assertEqual(sequences[NEWEST], ("b" * 64, "c" * 64, "a" * 64))
        self.assertEqual(sequences[SHUFFLE], ("a" * 64, "b" * 64, "c" * 64))
        self.assertEqual(sequences[UNCERTAIN], ("b" * 64, "a" * 64, "c" * 64))

    def test_ties_are_broken_by_hash_in_every_order(self) -> None:
        """Contract: V12 — same download minute, no prediction: the hash decides, and navigation still works."""
        _create([
            ImageSpec("b" * 64, 5, None, False, None, None),
            ImageSpec("a" * 64, 5, None, False, None, None),
        ])
        for order_name in ALL_ORDERS:
            with self.subTest(order=order_name):
                _set_order(order_name)
                order = get_queue_order()
                qs = _review_qs(show_nsfw=True, order=order)
                self.assertEqual(_sequence(qs), ["a" * 64, "b" * 64])
                first = _browse_ctx(qs, "a" * 64, "corpus", True, None, order)
                second = _browse_ctx(qs, "b" * 64, "corpus", True, None, order)
                self.assertEqual((first["position"], first["total"], first["next_hash"]), (1, 2, "b" * 64))
                self.assertEqual((second["position"], second["total"], second["prev_hash"]), (2, 2, "a" * 64))
                self.assertEqual(_queue_neighbor_hash(qs, Image.objects.get(pk="a" * 64), order), "b" * 64)

    def test_config_page_offers_the_order_select(self) -> None:
        """Contract: V9."""
        _set_order(SHUFFLE)
        response = self.client.get(reverse("config"))
        content = response.content.decode()
        self.assertIn('name="queue_order"', content)
        for order_name in ALL_ORDERS:
            self.assertIn(f'value="{order_name}"', content)
        self.assertIn('value="shuffle" selected', content)

    def test_saving_each_order_persists_it(self) -> None:
        """Contract: V9."""
        for order_name in ALL_ORDERS:
            with self.subTest(order=order_name):
                response = self.client.post(
                    reverse("set_vision_thresholds"),
                    {"sfw_threshold": 1, "nsfw_threshold": 1, "queue_order": order_name},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(ReviewThresholds.objects.get(pk=1).queue_order, order_name)
                self.assertIn(f'value="{order_name}" selected', response.content.decode())

    def test_unknown_or_missing_order_is_stored_as_oldest(self) -> None:
        """Contract: V9."""
        _set_order(SHUFFLE)
        self.client.post(
            reverse("set_vision_thresholds"),
            {"sfw_threshold": 1, "nsfw_threshold": 1, "queue_order": "../../etc"},
        )
        self.assertEqual(ReviewThresholds.objects.get(pk=1).queue_order, OLDEST)

        _set_order(SHUFFLE)
        self.client.post(
            reverse("set_vision_thresholds"), {"sfw_threshold": 1, "nsfw_threshold": 1}
        )
        self.assertEqual(ReviewThresholds.objects.get(pk=1).queue_order, OLDEST)

    def test_hand_edited_db_value_never_reaches_the_query(self) -> None:
        """Contract: V9."""
        _set_order(SHUFFLE)
        ReviewThresholds.objects.filter(pk=1).update(queue_order="DROP TABLE")
        self.assertIs(get_queue_order(), QUEUE_ORDERS[OLDEST])
        _create(MIXED_SET)
        self.assertEqual(_sequence(_review_qs()), ["a" * 64, "c" * 64, "b" * 64])

    def test_rate_and_purge_advance_along_the_configured_order(self) -> None:
        """Contract: V7.

        Scoring the first image shows its successor, scoring the new tail
        shows its predecessor, purging the last leaves the empty state — in
        each order, on the mixed set where the orders differ.
        """
        for order_name in ALL_ORDERS:
            with self.subTest(order=order_name):
                Image.objects.all().delete()
                _create(MIXED_SET)
                _set_order(order_name)
                first, second, third = _sequence(_review_qs())

                page = self.client.get(reverse("review_corpus")).content.decode()
                self.assertIn(f"images/{first}.jpg", page)

                page = self._score(first, 4)
                self.assertIn(f"images/{second}.jpg", page)

                page = self._score(third, 4)
                self.assertIn(f"images/{second}.jpg", page)

                page = self.client.post(
                    reverse("purge_corpus", args=[second])
                ).content.decode()
                self.assertIn("Queue empty", page)

    def _score(self, content_hash: str, score: int) -> str:
        response = self.client.post(
            reverse("score_corpus", args=[content_hash]), {"score": str(score)}
        )
        self.assertEqual(response.status_code, 200)
        return response.content.decode()


@override_settings(DEBUG=True)
class SkipStampTests(TestCase):
    """Contract V3: "shown and skipped" means moved on from, not merely rendered."""

    def setUp(self) -> None:
        _wipe()
        _set_order(OLDEST)
        _create(MIXED_SET)
        self.client = Client()
        _login(self.client)
        self.first, self.second, self.third = _sequence(_review_qs())

    def test_rendering_a_card_does_not_skip_it(self) -> None:
        """Contract: V3, V5."""
        self.client.get(reverse("review_corpus"))
        self.client.get(reverse("review_corpus"))
        self.assertIsNone(Image.objects.get(pk=self.first).queue_seen_at)
        self.assertEqual(_sequence(_review_qs()), [self.first, self.second, self.third])

    def test_moving_on_sends_the_left_image_to_the_back(self) -> None:
        """Contract: V3, V4."""
        self.client.get(reverse("review_corpus"))
        url = reverse("review_corpus_image", args=[self.second]) + f"?left={self.first}"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(Image.objects.get(pk=self.first).queue_seen_at)
        self.assertEqual(_sequence(_review_qs()), [self.second, self.third, self.first])

        url = reverse("review_corpus_image", args=[self.third]) + f"?left={self.second}"
        self.client.get(url)
        self.assertEqual(_sequence(_review_qs()), [self.third, self.first, self.second])

    def test_the_card_links_name_the_image_being_left(self) -> None:
        """Contract: V3."""
        page = self.client.get(reverse("review_corpus")).content.decode()
        next_url = reverse("review_corpus_image", args=[self.second])
        self.assertIn(f'hx-get="{next_url}?left={self.first}"', page)

    def test_unknown_or_rated_left_values_are_ignored(self) -> None:
        """Contract: V9-style input handling for the navigation parameter."""
        Image.objects.filter(pk=self.third).update(score=5)
        for left in ("", "nope", "../../etc", self.third):
            with self.subTest(left=left):
                url = reverse("review_corpus_image", args=[self.second]) + f"?left={left}"
                self.assertEqual(self.client.get(url).status_code, 200)
        self.assertIsNone(Image.objects.get(pk=self.third).queue_seen_at)
        self.assertEqual(_sequence(_review_qs()), [self.first, self.second])

    def test_nsfw_queue_stamps_on_leave_too(self) -> None:
        """Contract: V1, V3."""
        Image.objects.all().update(is_nsfw=True)
        url = reverse("review_nsfw_corpus_image", args=[self.second]) + f"?left={self.first}"
        self.assertEqual(self.client.get(url).status_code, 200)
        self.assertEqual(
            _sequence(_review_nsfw_qs()), [self.second, self.third, self.first]
        )
