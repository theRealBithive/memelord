"""
Review latency: the next picture after a rating is painted from what the card
already fetched, and the queue queries never scan the wide image table.

Measured on 2026-10-03 against a 25k-row copy with vectors on every row: the
score POST went from ~266 ms to ~60 ms, and the next picture is painted in the
same instant the card swaps because it was preloaded.

Contract:
R1 The review queue's head, neighbour and position queries and the nav badge
   counts are answered from the review-queue indexes, never by reading image
   rows, under every queue order: where an order needs a sort, the sort runs
   over index entries and only the chosen image is then fetched by key.
R2 A review card lists, as image preloads, the next PRELOAD_AHEAD images of
   its queue in the order the queue is served; the first of them is the image
   rate-and-advance and the next arrow show. A card outside the queue, and the
   empty state, preload nothing.
R3 Media and thumbnail responses tell the browser to keep the bytes (private,
   a year, immutable), so a preloaded picture is painted from the cache. A
   purged path still answers 404.
R4 The media view's purged-path check is an index lookup, not a table scan.
R5 The two chain counts shown in the nav while a chain is queued (taste
   vectors current, search vectors indexed) are answered from their own
   partial indexes, never by reading the vector blobs of every row.
"""

from __future__ import annotations

import io
import os
import re
import uuid
from datetime import timedelta
from pathlib import Path

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.db import connection
from django.test import TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.django import TestCase as HypothesisTestCase

from ratings import embeddings, search
from ratings.models import Image, ReviewThresholds
from ratings.queue_rules import QUEUE_ORDERS, get_queue_order
from ratings.views.common import nav_counts
from ratings.views.review import (
    PRELOAD_AHEAD,
    _review_ctx,
    _review_nsfw_ctx,
    _review_nsfw_qs,
    _review_qs,
)

HEX64 = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)
BASE_TIME = timezone.make_aware(timezone.datetime(2026, 1, 1, 12, 0, 0))
PRELOAD_RE = re.compile(r'<link rel="preload" as="image" href="([^"]+)"')
IMG_RE = re.compile(r'<div class="image-wrap">\s*<img src="([^"]+)"')


def _wipe() -> None:
    Image.objects.all().delete()
    ReviewThresholds.objects.all().delete()


def _set_order(order_name: str) -> None:
    ReviewThresholds.objects.update_or_create(
        pk=1,
        defaults={"sfw_threshold": 1, "nsfw_threshold": 1, "queue_order": order_name},
    )


def _create(content_hash: str, download_minutes: int, *, seen_minutes: int | None = None,
            predicted: float | None = None, score: int | None = None, is_nsfw: bool = False) -> None:
    # The whole hash in the path: every image has its own file, as the scraper
    # guarantees, and the fixture hashes differ only in their last digits.
    Image.objects.create(
        content_hash=content_hash,
        file_path=f"images/{content_hash}.jpg",
        source_label="test",
        predicted_score=predicted,
        score=score,
        is_nsfw=is_nsfw,
    )
    seen_at = None if seen_minutes is None else BASE_TIME + timedelta(minutes=seen_minutes)
    Image.objects.filter(pk=content_hash).update(
        downloaded_at=BASE_TIME + timedelta(minutes=download_minutes),
        queue_seen_at=seen_at,
    )


def _plan(queryset_or_sql) -> str:
    if isinstance(queryset_or_sql, str):
        with connection.cursor() as cursor:
            cursor.execute("EXPLAIN QUERY PLAN " + queryset_or_sql)
            return "\n".join(str(row[-1]) for row in cursor.fetchall())
    return queryset_or_sql.explain()


def _image_table_queries(captured) -> list[str]:
    """The SELECTs on the image table itself (the tag prefetch is not part of the queue)."""
    return [
        q["sql"]
        for q in captured
        if 'FROM "ratings_image"' in q["sql"] and "ratings_image_tags" not in q["sql"]
    ]


class QueueQueriesUseTheIndexTests(TestCase):
    def setUp(self) -> None:
        _wipe()
        _set_order(ReviewThresholds.ORDER_OLDEST)
        for i in range(3):
            _create(f"{i:064x}", download_minutes=i, predicted=0.8)
        _create(f"{7:064x}", download_minutes=7, predicted=0.8, is_nsfw=True)

    def test_every_card_query_reads_index_entries_or_one_row_by_key(self) -> None:
        """Contract: R1 (every order, the head card and a later card, both queues)"""
        for order_name in sorted(QUEUE_ORDERS):
            _set_order(order_name)
            for build_ctx, content_hash in (
                (_review_ctx, None),
                (_review_ctx, f"{1:064x}"),
                (_review_nsfw_ctx, None),
            ):
                with CaptureQueriesContext(connection) as ctx:
                    build_ctx(content_hash, False)
                for sql in _image_table_queries(ctx.captured_queries):
                    plan = _plan(sql)
                    reads_index_entries = "COVERING INDEX" in plan
                    one_row_by_key = "(content_hash=?)" in plan
                    self.assertTrue(
                        reads_index_entries or one_row_by_key,
                        f"{order_name} {build_ctx.__name__} {content_hash}:\n{plan}\n{sql}",
                    )

    def test_oldest_and_shuffle_heads_are_seeks_without_a_sort(self) -> None:
        """Contract: R1 (each of the two has an index in its own sort order)"""
        _set_order(ReviewThresholds.ORDER_OLDEST)
        plan = _plan(_review_qs(False).values_list("content_hash", flat=True)[:1])
        self.assertIn("COVERING INDEX image_review_queue_idx", plan)
        self.assertNotIn("TEMP B-TREE", plan)

        _set_order("shuffle")
        plan = _plan(_review_qs(False).values_list("content_hash", flat=True)[:1])
        self.assertIn("COVERING INDEX image_review_shuffle_idx", plan)
        self.assertNotIn("TEMP B-TREE", plan)
        plan = _plan(_review_nsfw_qs(False).values_list("content_hash", flat=True)[:1])
        self.assertIn("COVERING INDEX image_review_shuffle_idx", plan)

    def test_neighbour_and_position_queries_use_the_index(self) -> None:
        """Contract: R1"""
        order = get_queue_order()
        qs = _review_qs(False, order)
        middle = Image.objects.get(pk=f"{1:064x}")
        prev_filter, next_filter = order.position_filters(middle)
        self.assertIn("image_review_queue_idx", _plan(qs.filter(prev_filter)))
        self.assertIn("image_review_queue_idx", _plan(qs.filter(next_filter)))
        self.assertIn(
            "COVERING INDEX image_review_queue_idx",
            _plan(qs.filter(next_filter).values_list("content_hash", flat=True)[:PRELOAD_AHEAD]),
        )

    def test_nav_badge_counts_walk_the_index_not_the_rows(self) -> None:
        """Contract: R1"""
        for show_nsfw in (False, True):
            with CaptureQueriesContext(connection) as ctx:
                nav_counts(show_nsfw)
            plan = _plan(ctx.captured_queries[-1]["sql"])
            # Either review index covers the aggregate; SQLite picks one.
            self.assertIn("COVERING INDEX image_review_", plan, show_nsfw)

    def test_media_purged_check_is_an_index_lookup(self) -> None:
        """Contract: R4"""
        plan = _plan(Image.objects.filter(file_path="images/x.jpg", is_purged=True))
        self.assertIn("SEARCH ratings_image USING", plan)
        self.assertNotIn("SCAN ratings_image", plan)

    def test_chain_counts_walk_their_partial_indexes(self) -> None:
        """Contract: R5"""
        with CaptureQueriesContext(connection) as ctx:
            embeddings.taste_vector_counts()
        self.assertIn("image_taste_vector_idx", _plan(ctx.captured_queries[-1]["sql"]))

        with CaptureQueriesContext(connection) as ctx:
            search.index_counts()
        self.assertIn("image_search_vector_idx", _plan(ctx.captured_queries[-1]["sql"]))


@override_settings(DEBUG=True)
class PreloadProperties(HypothesisTestCase):
    """Contract: R2 for every order and every position in queues of up to six images."""

    @st.composite
    def unrated_queue(draw):
        n = draw(st.integers(min_value=1, max_value=6))
        hashes = draw(st.lists(HEX64, min_size=n, max_size=n, unique=True))
        download_minutes = draw(st.lists(st.integers(0, 4), min_size=n, max_size=n))
        seen = draw(st.lists(st.one_of(st.none(), st.integers(0, 50)), min_size=n, max_size=n))
        predicted = draw(
            st.lists(st.one_of(st.none(), st.floats(0.0, 1.0)), min_size=n, max_size=n)
        )
        return list(zip(hashes, download_minutes, seen, predicted, strict=True))

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(order_name=st.sampled_from(sorted(QUEUE_ORDERS)), specs=unrated_queue())
    def test_card_preloads_the_following_queue_entries(self, order_name, specs) -> None:
        """Contract: R2 (the queue as served is the oracle, not the sort rule)"""
        _wipe()
        _set_order(order_name)
        for content_hash, minutes, seen_minutes, predicted in specs:
            _create(content_hash, minutes, seen_minutes=seen_minutes, predicted=predicted)
        order = get_queue_order()
        served = list(_review_qs(False, order).values_list("content_hash", "file_path"))
        self.assertEqual(len(served), len(specs))
        for position, (content_hash, _path) in enumerate(served):
            ctx = _review_ctx(content_hash, False)
            following = served[position + 1 : position + 1 + PRELOAD_AHEAD]
            self.assertEqual(ctx["preload_paths"], [path for _h, path in following])
            expected_next = following[0][0] if following else None
            self.assertEqual(ctx["next_hash"], expected_next)
            self.assertEqual(len(ctx["preload_paths"]), min(PRELOAD_AHEAD, len(served) - position - 1))


@override_settings(DEBUG=True)
class PreloadInTheRenderedCardTests(TestCase):
    def setUp(self) -> None:
        _wipe()
        _set_order(ReviewThresholds.ORDER_OLDEST)
        self.hashes = [f"{i:064x}" for i in range(4)]
        for i, content_hash in enumerate(self.hashes):
            _create(content_hash, download_minutes=i, predicted=0.9)
        user = get_user_model().objects.create_user(f"lat-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def test_rate_and_advance_shows_the_first_preloaded_picture(self) -> None:
        """Contract: R2"""
        first_card = self.client.get(reverse("review_corpus")).content.decode()
        preloads = PRELOAD_RE.findall(first_card)
        self.assertEqual(len(preloads), PRELOAD_AHEAD)
        self.assertNotIn(IMG_RE.search(first_card).group(1), preloads)

        second_card = self.client.post(
            reverse("score_corpus", args=[self.hashes[0]]), {"score": "4"}
        ).content.decode()
        self.assertEqual(IMG_RE.search(second_card).group(1), preloads[0])
        self.assertEqual(PRELOAD_RE.findall(second_card)[0], preloads[1])

    def test_last_image_and_empty_queue_preload_nothing(self) -> None:
        """Contract: R2"""
        last_card = self.client.get(reverse("review_corpus_image", args=[self.hashes[-1]])).content.decode()
        self.assertEqual(PRELOAD_RE.findall(last_card), [])
        Image.objects.update(score=5)
        empty = self.client.get(reverse("review_corpus")).content.decode()
        self.assertEqual(PRELOAD_RE.findall(empty), [])
        self.assertIn("Queue empty", empty)

    def test_out_of_queue_card_preloads_nothing(self) -> None:
        """Contract: R2 (a rated image opened for re-review stands alone)"""
        Image.objects.filter(pk=self.hashes[1]).update(score=6)
        card = self.client.get(reverse("review_corpus_image", args=[self.hashes[1]])).content.decode()
        self.assertEqual(PRELOAD_RE.findall(card), [])
        self.assertIn(self.hashes[1], IMG_RE.search(card).group(1))


def _png_bytes() -> bytes:
    from PIL import Image as PilImage

    buffer = io.BytesIO()
    PilImage.new("RGB", (8, 8), (200, 30, 30)).save(buffer, "PNG")
    return buffer.getvalue()


@override_settings(DEBUG=False)
class MediaCacheHeaderTests(TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "images").mkdir()
        (root / "images" / "pic.png").write_bytes(_png_bytes())
        self.root = root
        user = get_user_model().objects.create_user(f"lat-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_media_and_thumbnail_are_cacheable_for_the_browser_only(self) -> None:
        """Contract: R3"""
        with override_settings(MEDIA_ROOT=self.root, DATA_DIR=self.root):
            media = self.client.get("/media/images/pic.png")
            thumb = self.client.get("/thumb/images/pic.png")
        for response in (media, thumb):
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response["Cache-Control"], "private, max-age=31536000, immutable")

    def test_purged_path_is_still_refused(self) -> None:
        """Contract: R3 (the cache header never resurrects a purged file)"""
        Image.objects.create(
            content_hash="f" * 64, file_path="images/pic.png", source_label="t", is_purged=True
        )
        with override_settings(MEDIA_ROOT=self.root, DATA_DIR=self.root):
            self.assertEqual(self.client.get("/media/images/pic.png").status_code, 404)
