"""[vision] hide-threshold behaviour: helper, queueset filter, count badges."""

import os
import uuid

import django
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from ratings.models import Image, ReviewThresholds
from ratings.queue_rules import below_cutoff_q, bucket_to_cutoff, get_review_thresholds
from ratings.views import _counts


def test_bucket_to_cutoff_known_buckets():
    """Bucket 1 lets everything through; bucket 6 is the strictest in-range value."""
    assert bucket_to_cutoff(1) == 0.0
    assert bucket_to_cutoff(2) == pytest.approx(1 / 6)
    assert bucket_to_cutoff(4) == 0.5
    assert bucket_to_cutoff(6) == pytest.approx(5 / 6)


def test_bucket_to_cutoff_clamps_out_of_range():
    """Out-of-range inputs collapse to the nearest valid bucket — the value
    flows from user-edited TOML / POST data so we can't trust the type or range."""
    assert bucket_to_cutoff(0) == 0.0
    assert bucket_to_cutoff(-5) == 0.0
    assert bucket_to_cutoff(7) == pytest.approx(5 / 6)
    assert bucket_to_cutoff(99) == pytest.approx(5 / 6)


class ReviewThresholdsSeedingTests(TestCase):
    """get_review_thresholds seeds the singleton from Django settings on first call."""

    def setUp(self) -> None:
        ReviewThresholds.objects.all().delete()

    @override_settings(SFW_THRESHOLD_BUCKET=4, NSFW_THRESHOLD_BUCKET=5)
    def test_seeds_from_settings_when_missing(self) -> None:
        sfw, nsfw = get_review_thresholds()
        self.assertEqual((sfw, nsfw), (4, 5))
        # And persists — subsequent edits via DB win over settings.
        row = ReviewThresholds.objects.get(pk=1)
        self.assertEqual((row.sfw_threshold, row.nsfw_threshold), (4, 5))

    @override_settings(SFW_THRESHOLD_BUCKET=2, NSFW_THRESHOLD_BUCKET=2)
    def test_db_wins_over_settings_after_first_seed(self) -> None:
        ReviewThresholds.objects.create(sfw_threshold=6, nsfw_threshold=3)
        sfw, nsfw = get_review_thresholds()
        self.assertEqual((sfw, nsfw), (6, 3))


def _make_image(*, predicted: float | None, is_nsfw: bool = False) -> Image:
    content_hash = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=content_hash,
        file_path=f"images/{content_hash}.jpg",
        source_label="test",
        is_nsfw=is_nsfw,
        predicted_score=predicted,
    )


@override_settings(SFW_THRESHOLD_BUCKET=1, NSFW_THRESHOLD_BUCKET=1)
class ReviewQueueThresholdFilterTests(TestCase):
    """_review_qs / _review_nsfw_qs respect the DB singleton's thresholds."""

    def setUp(self) -> None:
        ReviewThresholds.objects.all().delete()
        Image.objects.all().delete()

    def _set_thresholds(self, sfw: int, nsfw: int) -> None:
        ReviewThresholds.objects.update_or_create(
            pk=1, defaults={"sfw_threshold": sfw, "nsfw_threshold": nsfw}
        )

    def test_sfw_threshold_hides_low_score_images(self) -> None:
        from ratings.views import _review_qs

        keep = _make_image(predicted=0.7)
        hide = _make_image(predicted=0.2)
        self._set_thresholds(sfw=4, nsfw=1)  # cutoff 0.5

        hashes = set(_review_qs(show_nsfw=False).values_list("content_hash", flat=True))
        self.assertIn(keep.content_hash, hashes)
        self.assertNotIn(hide.content_hash, hashes)

    def test_null_predicted_score_always_visible(self) -> None:
        """Never-classified images shouldn't be hidden — that would silently
        empty the queue after migration or after fresh scrapes pre-classify."""
        from ratings.views import _review_qs

        unscored = _make_image(predicted=None)
        self._set_thresholds(sfw=6, nsfw=6)  # strictest

        hashes = set(_review_qs(show_nsfw=False).values_list("content_hash", flat=True))
        self.assertIn(unscored.content_hash, hashes)

    def test_nsfw_threshold_independent_of_sfw(self) -> None:
        from ratings.views import _review_nsfw_qs

        keep_nsfw = _make_image(predicted=0.9, is_nsfw=True)
        hide_nsfw = _make_image(predicted=0.4, is_nsfw=True)
        # SFW threshold is strict but NSFW is permissive enough to keep 0.9 only.
        self._set_thresholds(sfw=6, nsfw=5)  # NSFW cutoff ≈ 0.667

        hashes = set(_review_nsfw_qs().values_list("content_hash", flat=True))
        self.assertIn(keep_nsfw.content_hash, hashes)
        self.assertNotIn(hide_nsfw.content_hash, hashes)

    def test_default_bucket_1_shows_everything(self) -> None:
        from ratings.views import _review_qs

        a = _make_image(predicted=0.0)
        b = _make_image(predicted=0.5)
        c = _make_image(predicted=1.0)
        # No singleton row -> seeds from default settings (1, 1).

        hashes = set(_review_qs(show_nsfw=False).values_list("content_hash", flat=True))
        self.assertEqual(hashes, {a.content_hash, b.content_hash, c.content_hash})

    def test_dial_hides_low_confidence_at_max(self) -> None:
        """Images with low predicted_score are hidden at the strictest dial setting."""
        from ratings.views import _review_qs

        hide = _make_image(predicted=0.1)
        self._set_thresholds(sfw=6, nsfw=1)

        hashes = set(_review_qs(show_nsfw=False).values_list("content_hash", flat=True))
        self.assertNotIn(hide.content_hash, hashes)

    def test_hidden_from_review_appears_in_below_cutoff_q(self) -> None:
        hide = _make_image(predicted=0.2)
        self._set_thresholds(sfw=4, nsfw=1)

        below = below_cutoff_q(4, 1, show_nsfw=False)
        hashes = set(Image.objects.filter(below).values_list("content_hash", flat=True))
        self.assertIn(hide.content_hash, hashes)


@override_settings(DEBUG=True)
class SetVisionThresholdsNavTests(TestCase):
    """Saving thresholds on /config/ refreshes nav badge counts via HTMX OOB."""

    def setUp(self) -> None:
        ReviewThresholds.objects.all().delete()
        Image.objects.all().delete()
        ReviewThresholds.objects.create(sfw_threshold=1, nsfw_threshold=1)
        self.client = Client()
        username = f"vis_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")
        content_hash = uuid.uuid4().hex
        Image.objects.create(
            content_hash=content_hash,
            file_path=f"images/{content_hash}.jpg",
            source_label="test",
            predicted_score=0.2,
        )

    def test_save_returns_oob_nav_with_updated_counts(self) -> None:
        before = _counts(show_nsfw=False)
        response = self.client.post(
            reverse("set_vision_thresholds"),
            {"sfw_threshold": 4, "nsfw_threshold": 1},
        )
        after = _counts(show_nsfw=False)

        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('hx-swap-oob="true"', content)
        self.assertIn(f'id="badge-queue">{after["queue_count"]}</span>', content)
        self.assertIn(f'id="badge-below">{after["below_cutoff_count"]}</span>', content)
        self.assertGreater(after["below_cutoff_count"], before["below_cutoff_count"])
        self.assertLess(after["queue_count"], before["queue_count"])
