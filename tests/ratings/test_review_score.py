"""Tests for score submission + queue navigation in the corpus review flow.

Covers POST /review/<hash>/score/ (score_corpus) — setting the score, advancing
to the next queue item, end-of-queue behaviour — and re-scoring a below-cutoff
image from the gallery lightbox (gallery_action).
"""

import os
import uuid
from datetime import timedelta

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from ratings.models import Image


@override_settings(DEBUG=True)
class ScoreCorpusTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self.client = Client()
        username = f"score_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")
        self.base = timezone.now()

    def _unscored(self, *, age_hours: float) -> str:
        """Create an unscored image; older age_hours sorts earlier in the queue."""
        h = uuid.uuid4().hex
        Image.objects.create(
            content_hash=h, file_path=f"images/{h}.jpg", source_label="test"
        )
        # downloaded_at is auto_now_add — override it so queue order is deterministic.
        Image.objects.filter(content_hash=h).update(
            downloaded_at=self.base - timedelta(hours=age_hours)
        )
        return h

    def test_score_sets_value_and_rated_at(self) -> None:
        h = self._unscored(age_hours=1)
        response = self.client.post(reverse("score_corpus", args=[h]), {"score": "5"})
        self.assertEqual(response.status_code, 200)
        img = Image.objects.get(content_hash=h)
        self.assertEqual(img.score, 5)
        self.assertIsNotNone(img.rated_at)

    def test_score_advances_to_next_image(self) -> None:
        a = self._unscored(age_hours=3)
        b = self._unscored(age_hours=2)
        self._unscored(age_hours=1)  # c — stays further back in the queue

        response = self.client.post(reverse("score_corpus", args=[a]), {"score": "4"})

        content = response.content.decode()
        # The rendered card embeds the shown image's hash in its action URLs.
        self.assertIn(b, content)
        self.assertNotIn(a, content)

    def test_scored_image_leaves_queue(self) -> None:
        a = self._unscored(age_hours=2)
        self._unscored(age_hours=1)

        self.client.post(reverse("score_corpus", args=[a]), {"score": "3"})

        # A fresh review GET must no longer surface the scored image.
        review = self.client.get(reverse("review_corpus")).content.decode()
        self.assertNotIn(a, review)

    def test_score_last_image_navigates_to_previous(self) -> None:
        a = self._unscored(age_hours=2)
        b = self._unscored(age_hours=1)  # last in queue

        response = self.client.post(reverse("score_corpus", args=[b]), {"score": "6"})

        content = response.content.decode()
        self.assertIn(a, content)
        self.assertNotIn(b, content)

    def test_score_after_initial_review_advances(self) -> None:
        """
        After a GET stamps queue_seen_at, scoring the seen image must still
        advance to its neighbour — this exercises the non-NULL branch of the
        windowed prev/next lookup, which the other tests bypass by POSTing
        directly without a prior GET.
        """
        a = self._unscored(age_hours=3)
        b = self._unscored(age_hours=2)
        # GET sets queue_seen_at on the head image (a).
        self.client.get(reverse("review_corpus"))
        response = self.client.post(reverse("score_corpus", args=[a]), {"score": "4"})
        content = response.content.decode()
        self.assertIn(b, content)
        self.assertNotIn(a, content)

    def test_score_only_image_shows_empty_queue(self) -> None:
        h = self._unscored(age_hours=1)
        response = self.client.post(reverse("score_corpus", args=[h]), {"score": "2"})
        self.assertIn("Queue empty", response.content.decode())

    def test_invalid_score_is_ignored(self) -> None:
        h = self._unscored(age_hours=1)
        self.client.post(reverse("score_corpus", args=[h]), {"score": "99"})
        self.assertIsNone(Image.objects.get(content_hash=h).score)

    def test_missing_score_is_noop(self) -> None:
        """A POST without a score must not be treated as score 0 (trash)."""
        h = self._unscored(age_hours=1)
        self.client.post(reverse("score_corpus", args=[h]), {})
        self.assertIsNone(Image.objects.get(content_hash=h).score)

    def test_trash_sets_score_zero_and_leaves_queue(self) -> None:
        a = self._unscored(age_hours=2)
        self._unscored(age_hours=1)

        response = self.client.post(reverse("score_corpus", args=[a]), {"score": "0"})

        self.assertEqual(response.status_code, 200)
        img = Image.objects.get(content_hash=a)
        self.assertEqual(img.score, 0)
        self.assertIsNotNone(img.rated_at)
        review = self.client.get(reverse("review_corpus")).content.decode()
        self.assertNotIn(a, review)

    def test_trashed_image_kept_and_shows_in_below_cutoff(self) -> None:
        h = self._unscored(age_hours=1)
        self.client.post(reverse("score_corpus", args=[h]), {"score": "0"})
        img = Image.objects.get(content_hash=h)
        self.assertFalse(img.is_purged)  # file kept for training
        below = self.client.get(reverse("below_cutoff")).content.decode()
        self.assertIn(h, below)

    def test_review_by_hash_shows_that_scored_image(self) -> None:
        """GET /review/<hash>/ for an already-scored image shows THAT image,
        not the first unscored one (the gallery 'review' link)."""
        scored = self._unscored(age_hours=2)
        Image.objects.filter(content_hash=scored).update(score=5)
        unscored = self._unscored(age_hours=1)

        response = self.client.get(reverse("review_corpus_image", args=[scored]))

        content = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn(scored, content)
        self.assertNotIn(unscored, content)

    def test_rescore_scored_image_updates_and_stays(self) -> None:
        """Re-scoring an already-scored (out-of-queue) image works and keeps the
        review on that image rather than jumping into the rate queue."""
        scored = self._unscored(age_hours=2)
        Image.objects.filter(content_hash=scored).update(score=4)
        self._unscored(age_hours=1)  # an unscored image also exists

        response = self.client.post(
            reverse("score_corpus", args=[scored]), {"score": "6"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Image.objects.get(content_hash=scored).score, 6)
        self.assertIn(scored, response.content.decode())

    def test_score_requires_post(self) -> None:
        h = self._unscored(age_hours=1)
        response = self.client.get(reverse("score_corpus", args=[h]))
        self.assertEqual(response.status_code, 405)

    def test_score_requires_login(self) -> None:
        h = self._unscored(age_hours=1)
        response = Client().post(reverse("score_corpus", args=[h]), {"score": "5"})
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)


@override_settings(DEBUG=True)
class BelowCutoffRescoreTests(TestCase):
    """Re-scoring a low-scored image from the gallery lightbox (gallery_action)."""

    def setUp(self) -> None:
        Image.objects.all().delete()
        self.client = Client()
        username = f"rescore_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    def _scored(self, score: int) -> str:
        h = uuid.uuid4().hex
        Image.objects.create(
            content_hash=h,
            file_path=f"images/{h}.jpg",
            source_label="test",
            score=score,
            rated_at=timezone.now(),
        )
        return h

    def test_rescore_updates_score_and_returns_json(self) -> None:
        h = self._scored(1)
        response = self.client.post(
            reverse("gallery_action", args=[h]), {"action": "score", "score": "5"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 5)
        self.assertEqual(Image.objects.get(content_hash=h).score, 5)

    def test_rescored_image_leaves_below_cutoff(self) -> None:
        h = self._scored(1)
        # Present in Below Cutoff while score <= 2.
        before = self.client.get(reverse("below_cutoff")).content.decode()
        self.assertIn(h, before)

        self.client.post(
            reverse("gallery_action", args=[h]), {"action": "score", "score": "5"}
        )

        after = self.client.get(reverse("below_cutoff")).content.decode()
        self.assertNotIn(h, after)

    def test_purge_hard_deletes_and_marks_purged(self) -> None:
        h = self._scored(1)
        response = self.client.post(
            reverse("gallery_action", args=[h]), {"action": "purge"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json().get("deleted"))
        self.assertTrue(Image.objects.get(content_hash=h).is_purged)

    def test_legacy_trash_action_does_not_hard_delete(self) -> None:
        """The hard-delete action is 'purge' now; a stray 'trash' must not delete."""
        h = self._scored(1)
        response = self.client.post(
            reverse("gallery_action", args=[h]), {"action": "trash"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("deleted", response.json())
        self.assertFalse(Image.objects.get(content_hash=h).is_purged)

    def test_rescore_requires_post(self) -> None:
        h = self._scored(1)
        response = self.client.get(reverse("gallery_action", args=[h]))
        self.assertEqual(response.status_code, 405)

    def test_rescore_requires_login(self) -> None:
        h = self._scored(1)
        response = Client().post(
            reverse("gallery_action", args=[h]), {"action": "score", "score": "5"}
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)
