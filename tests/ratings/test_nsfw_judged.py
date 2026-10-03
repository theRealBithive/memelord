"""
NSFW decisions: only a person's judgement trains the NSFW head, and the head
only touches images nobody has decided on.

Before this, is_nsfw was one boolean with no memory of who set it: the head's
own flags from scrape time fed the next training as positives, and the 25k
unreviewed images counted as safe examples, so every unflagged NSFW picture
among them taught the head that its kind is safe. nsfw_judged is the split
the taste side always had (score is the operator's, predicted_score the
model's).

Contract (confirmed 2026-10-03):
N1 Only a person's decision trains the NSFW head. An image is an NSFW example
   when a person flagged it, and a safe example when a person rated it without
   flagging it or removed a flag. Images nobody looked at count for neither
   side, whatever flag the model gave them.
N2 Flagging, un-flagging and rating record that a person decided the flag. A
   flag the model set does not.
N3 The model may set or clear the flag only on images no person has decided. A
   decided flag is never changed by the model.
N4 A model flag routes the image into the NSFW queue exactly like a hand-set
   one. Queues, taste category and hide dial do not distinguish them.
N5 Classify now runs in the background, shows in the nav like the other jobs,
   refuses to start twice, and recomputes the taste prediction for every image
   whose category changed.
N6 The head is trained only when at least 10 flagged and 10 safe decided
   examples exist. Otherwise Train says so and keeps the previous head. Same
   rule as the taste models.
N7 Upgrade: every rated image and every image flagged so far counts as decided.
   Today's training set stays exactly as it is, and only rule N2 adds decisions
   from here on.
N8 Stats shows how many images trained the head on each side, and how many
   model flags nobody has decided yet.
N9 The taste models are not affected. Only the NSFW head's training set changes.
"""

from __future__ import annotations

import importlib
import os
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

import numpy as np
from django.apps import apps
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from django_q.models import OrmQ, Task
from django_q.tasks import async_task
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.django import TestCase as HypothesisTestCase
from loguru import logger
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression

from core import brain, dedup, nsfw, siglip, taste, trainer
from ratings import classify, scraper, tasks
from ratings.context_processors import _classify_job_ctx
from ratings.models import Image, LogEntry, ReviewThresholds
from ratings.views.common import nav_counts
from ratings.views.review import _review_nsfw_qs, _review_qs


def _vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(768).astype(np.float32)


def _row(
    *,
    flagged: bool = False,
    judged: bool = False,
    score: int | None = None,
    purged: bool = False,
    seed: int = 1,
    label: str = "tg",
    predicted: float | None = 0.5,
) -> Image:
    """A row with current DINOv3 and SigLIP2 vectors, so no encoder is ever needed."""
    content_hash = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=content_hash,
        file_path=f"images/{content_hash}.png",
        source_label=label,
        is_nsfw=flagged,
        nsfw_judged=judged,
        score=score,
        is_purged=purged,
        predicted_score=predicted,
        phash="0" * 16,
        embedding=brain.embedding_to_bytes(_vector(seed)),
        embedding_model=brain.ENCODER_ID,
        search_embedding=brain.embedding_to_bytes(_vector(1000 + seed)),
        search_embedding_model=siglip.SEARCH_ENCODER_ID,
    )


def _write_png(data_dir: Path, rel: str) -> Path:
    path = data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (4, 4), color=(200, 20, 20)).save(path)
    return path


def _save_taste_model(path: Path) -> None:
    X = np.vstack([
        taste.combine_features(_vector(1), _vector(2)),
        taste.combine_features(_vector(3), _vector(4)),
    ])
    taste.save_taste_model(taste.TasteModel(shared=LogisticRegression().fit(X, [1, 0])), path)


def _clear_queue() -> None:
    OrmQ.objects.all().delete()
    Task.objects.filter(func=classify.CLASSIFY_TASK).delete()


def _finished_task(*, success: bool, result) -> Task:
    now = timezone.now()
    return Task.objects.create(
        id=uuid.uuid4().hex, name=uuid.uuid4().hex[:8], func=classify.CLASSIFY_TASK,
        started=now, stopped=now, success=success, result=result,
    )


# ── N1: the training set ─────────────────────────────────────────────────────


class TrainingSetProperties(HypothesisTestCase):
    """Contract: N1 over every mix of rated/unrated, flagged/unflagged, decided/undecided, purged."""

    row_spec = st.tuples(
        st.one_of(st.none(), st.integers(0, 6)),  # score
        st.booleans(),  # flagged
        st.booleans(),  # judged
        st.booleans(),  # purged
    )

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(specs=st.lists(row_spec, min_size=1, max_size=8))
    def test_only_decided_rows_train_and_each_on_the_side_of_its_flag(self, specs) -> None:
        """Contract: N1 (the generator reaches every quadrant, including model flags on unseen rows)"""
        Image.objects.all().delete()
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            expected_nsfw: set[Path] = set()
            expected_safe: set[Path] = set()
            decided_live: set[Path] = set()
            for score, flagged, judged, purged in specs:
                row = _row(flagged=flagged, judged=judged, score=score, purged=purged)
                path = _write_png(data_dir, row.file_path)
                if judged and not purged:
                    decided_live.add(path)
                    (expected_nsfw if flagged else expected_safe).add(path)

            nsfw_paths, safe_paths = trainer.collect_nsfw_paths(data_dir)

            self.assertEqual(set(nsfw_paths), expected_nsfw)
            self.assertEqual(set(safe_paths), expected_safe)
            # Conservation across both sides: every decided live row on exactly
            # one side, and nothing else anywhere.
            self.assertEqual(set(nsfw_paths) | set(safe_paths), decided_live)
            self.assertEqual(set(nsfw_paths) & set(safe_paths), set())


class TrainingSetEdgeTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_an_unrated_hand_flag_is_an_example_and_an_unrated_model_flag_is_not(self) -> None:
        """Contract: N1 (the operator flags far more than they rate)"""
        hand = _row(flagged=True, judged=True)
        model = _row(flagged=True, judged=False)
        removed = _row(flagged=False, judged=True)
        unseen = _row(flagged=False, judged=False)
        for row in (hand, model, removed, unseen):
            _write_png(self.data_dir, row.file_path)

        nsfw_paths, safe_paths = trainer.collect_nsfw_paths(self.data_dir)

        self.assertEqual(nsfw_paths, [self.data_dir / hand.file_path])
        self.assertEqual(safe_paths, [self.data_dir / removed.file_path])

    def test_a_decided_row_without_its_file_is_left_out(self) -> None:
        """Contract: N1 (as before: nothing to encode means nothing to train on)"""
        _row(flagged=True, judged=True)
        self.assertEqual(trainer.collect_nsfw_paths(self.data_dir), ([], []))


# ── N2: who records a decision ───────────────────────────────────────────────


@override_settings(DEBUG=True)
class DecisionRecordingTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        user = get_user_model().objects.create_user(f"nsfw_{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def test_rating_records_the_decision_with_the_flag_as_it_stands(self) -> None:
        """Contract: N2 (a flag left on is confirmed, a flag left off is a safe example)"""
        flagged = _row(flagged=True, seed=1)
        plain = _row(flagged=False, seed=2)
        _row(seed=3)  # a neighbour for rate-and-advance

        self.client.post(reverse("score_corpus", args=[flagged.content_hash]), {"score": 4})
        self.client.post(reverse("score_corpus", args=[plain.content_hash]), {"score": 2})

        flagged.refresh_from_db()
        plain.refresh_from_db()
        self.assertEqual((flagged.is_nsfw, flagged.nsfw_judged, flagged.score), (True, True, 4))
        self.assertEqual((plain.is_nsfw, plain.nsfw_judged, plain.score), (False, True, 2))

    def test_a_hand_toggle_records_the_decision_in_both_directions(self) -> None:
        """Contract: N2"""
        image = _row(flagged=False)
        _row(seed=2)

        self.client.post(reverse("toggle_nsfw", args=[image.content_hash]))
        image.refresh_from_db()
        self.assertEqual((image.is_nsfw, image.nsfw_judged), (True, True))

        Image.objects.filter(pk=image.pk).update(nsfw_judged=False)
        self.client.post(reverse("lightbox_nsfw", args=[image.content_hash]))
        image.refresh_from_db()
        self.assertEqual((image.is_nsfw, image.nsfw_judged), (False, True))

    def test_a_flag_the_model_sets_at_download_time_is_no_decision(self) -> None:
        """Contract: N2 (the scrape's own flag stays undecided)"""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            path = _write_png(data_dir, "images/new.png")
            candidates = [(path, "https://x/1", "tg", uuid.uuid4().hex, "0" * 16)]
            vision = scraper.VisionConfig()
            with mock.patch.object(brain, "encode", return_value=(np.stack([_vector(9)]), [path])), \
                 mock.patch.object(nsfw, "predict_nsfw", return_value=True):
                inserted = scraper._process_candidates(
                    candidates, data_dir, dedup.DedupIndex(), object(), object(), vision, object()
                )
        self.assertEqual(inserted, 1)
        row = Image.objects.get(content_hash=candidates[0][3])
        self.assertEqual((row.is_nsfw, row.nsfw_judged), (True, False))

    def test_a_flag_the_model_sets_in_the_classify_pass_is_no_decision(self) -> None:
        """Contract: N2"""
        row = _row(flagged=False)
        with mock.patch.object(nsfw, "predict_nsfw", return_value=True):
            scraper.classify_images(
                Path("/nonexistent"), scraper.VisionConfig(), encoder=object(), transform=object(), nsfw_clf=object()
            )
        row.refresh_from_db()
        self.assertEqual((row.is_nsfw, row.nsfw_judged), (True, False))


# ── N3, N5 (recompute): the model touches undecided rows only ────────────────


class ModelTouchesUndecidedOnlyTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _classify(self, verdict: bool, *, taste_weights: Path | None = None) -> dict:
        vision = scraper.VisionConfig(weights_path=taste_weights or self.data_dir / "missing.pkl")
        with mock.patch.object(nsfw, "predict_nsfw", return_value=verdict):
            return scraper.classify_images(
                self.data_dir, vision, encoder=object(), transform=object(), nsfw_clf=object()
            )

    def test_a_head_that_says_nsfw_flags_only_undecided_rows(self) -> None:
        """Contract: N3"""
        decided_safe = _row(flagged=False, judged=True, seed=1)
        decided_nsfw = _row(flagged=True, judged=True, seed=2)
        undecided_plain = _row(flagged=False, judged=False, seed=3)
        undecided_flag = _row(flagged=True, judged=False, seed=4)

        counts = self._classify(True)

        for row in (decided_safe, decided_nsfw, undecided_plain, undecided_flag):
            row.refresh_from_db()
        self.assertFalse(decided_safe.is_nsfw)
        self.assertTrue(decided_nsfw.is_nsfw)
        self.assertTrue(undecided_plain.is_nsfw)
        self.assertTrue(undecided_flag.is_nsfw)
        self.assertEqual([r.nsfw_judged for r in (undecided_plain, undecided_flag)], [False, False])
        self.assertEqual((counts["nsfw_tagged"], counts["nsfw_untagged"], counts["processed"]), (1, 0, 4))

    def test_a_head_that_says_safe_clears_only_undecided_flags(self) -> None:
        """Contract: N3 (both directions, so a newer head may revise its predecessor's guess)"""
        decided_nsfw = _row(flagged=True, judged=True, seed=1)
        undecided_flag = _row(flagged=True, judged=False, seed=2)
        undecided_plain = _row(flagged=False, judged=False, seed=3)

        counts = self._classify(False)

        for row in (decided_nsfw, undecided_flag, undecided_plain):
            row.refresh_from_db()
        self.assertTrue(decided_nsfw.is_nsfw)
        self.assertFalse(undecided_flag.is_nsfw)
        self.assertFalse(undecided_plain.is_nsfw)
        self.assertEqual((counts["nsfw_tagged"], counts["nsfw_untagged"]), (0, 1))

    def test_a_flipped_flag_drops_the_old_categorys_prediction(self) -> None:
        """Contract: N5 (no taste model: the stale prediction goes, nothing replaces it)"""
        flipped = _row(flagged=False, judged=False, predicted=0.9, seed=1)
        kept = _row(flagged=True, judged=True, predicted=0.9, seed=2)

        self._classify(True)

        flipped.refresh_from_db()
        kept.refresh_from_db()
        self.assertIsNone(flipped.predicted_score)
        self.assertEqual(kept.predicted_score, 0.9)

    def test_a_flipped_flag_is_repredicted_with_the_new_categorys_model(self) -> None:
        """Contract: N5"""
        weights = self.data_dir / "taste.pkl"
        _save_taste_model(weights)
        flipped = _row(flagged=False, judged=False, predicted=None, seed=1)

        self._classify(True, taste_weights=weights)

        flipped.refresh_from_db()
        self.assertTrue(flipped.is_nsfw)
        self.assertIsNotNone(flipped.predicted_score)

    def test_without_any_model_the_pass_reports_zero_and_touches_nothing(self) -> None:
        """Contract: N5 (the report says what happened, also when nothing could)"""
        row = _row(flagged=False, judged=False)
        counts = scraper.classify_images(
            self.data_dir, scraper.VisionConfig(), encoder=object(), transform=object()
        )
        row.refresh_from_db()
        self.assertEqual(counts, {"processed": 0, "nsfw_tagged": 0, "nsfw_untagged": 0})
        self.assertFalse(row.is_nsfw)

    def test_a_rated_row_is_never_touched(self) -> None:
        """Contract: N3 (the pass walks unrated rows only, as before)"""
        rated = _row(flagged=False, judged=False, score=5)
        self._classify(True)
        rated.refresh_from_db()
        self.assertFalse(rated.is_nsfw)


# ── N4: a model flag and a hand flag route the same way ──────────────────────


class ModelFlagRoutesLikeAHandFlagTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        # Dial fully open, so the fixture's 0.5 predictions are visible in both queues.
        ReviewThresholds.objects.update_or_create(
            pk=1, defaults={"sfw_threshold": 1, "nsfw_threshold": 1, "queue_order": "oldest"}
        )

    def test_both_flags_land_in_the_nsfw_queue_and_leave_the_main_one(self) -> None:
        """Contract: N4"""
        by_model = _row(flagged=True, judged=False, seed=1)
        by_hand = _row(flagged=True, judged=True, seed=2)
        plain = _row(flagged=False, judged=False, seed=3)

        nsfw_queue = set(_review_nsfw_qs(False).values_list("content_hash", flat=True))
        main_queue = set(_review_qs(False).values_list("content_hash", flat=True))

        self.assertEqual(nsfw_queue, {by_model.content_hash, by_hand.content_hash})
        self.assertEqual(main_queue, {plain.content_hash})
        self.assertEqual(nav_counts(False)["nsfw_queue_count"], 2)
        self.assertEqual(
            taste.taste_group(by_model.source_label, by_model.is_nsfw),
            taste.taste_group(by_hand.source_label, by_hand.is_nsfw),
        )


# ── N5: the Classify now job ─────────────────────────────────────────────────


@override_settings(DEBUG=True)
class ClassifyJobTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        _clear_queue()
        user = get_user_model().objects.create_user(f"cls_{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def test_the_button_queues_one_run_and_refuses_a_second(self) -> None:
        """Contract: N5"""
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertTrue(classify.enqueue_classify_job())
        enqueue.assert_called_once_with(classify.CLASSIFY_TASK)

        async_task(classify.CLASSIFY_TASK)
        self.assertTrue(classify.classify_job_queued())
        with mock.patch("django_q.tasks.async_task") as enqueue:
            self.assertFalse(classify.enqueue_classify_job())
        enqueue.assert_not_called()

    def test_the_chains_are_not_mistaken_for_a_classify_run(self) -> None:
        """Contract: N5 (the queue check reads this job's rows only)"""
        from ratings import embeddings, search

        async_task(embeddings.REENCODE_TASK)
        async_task(search.INDEX_TASK)
        self.assertFalse(classify.classify_job_queued())

    def test_trigger_view_renders_pending_and_the_nav_shows_the_job(self) -> None:
        """Contract: N5"""
        response = self.client.post(reverse("trigger_classify"))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(classify.classify_job_queued())
        body = response.content.decode()
        self.assertIn("Classifying…", body)
        self.assertIn(reverse("classify_status"), body)
        indicator = self.client.get(reverse("job_indicator")).content.decode()
        self.assertIn("Classifying", indicator)
        self.assertTrue(_classify_job_ctx()["active_classify"])

    def test_status_view_shows_the_last_report_when_nothing_is_queued(self) -> None:
        """Contract: N5, N8"""
        _row(flagged=True, judged=False)
        _finished_task(success=True, result={"ok": True, "processed": 7, "nsfw_tagged": 2, "nsfw_untagged": 1})

        body = self.client.get(reverse("classify_status")).content.decode()

        self.assertIn("Classified 7 unrated images: 2 flagged, 1 unflagged by the NSFW head.", body)
        self.assertIn("1 model flag awaiting your decision.", body)
        self.assertIn("train-ok", body)

    def test_a_failed_run_is_reported_with_its_error(self) -> None:
        """Contract: N5"""
        _finished_task(success=True, result={"ok": False, "error": "no weights"})
        self.assertEqual(
            classify.last_classify_report(), {"ok": False, "text": "Last classify run failed: no weights"}
        )
        _finished_task(success=False, result=None)
        self.assertEqual(
            classify.last_classify_report(),
            {"ok": False, "text": "Last classify run failed: Task exited without a result."},
        )

    def test_no_run_yet_means_no_report(self) -> None:
        """Contract: N5"""
        self.assertIsNone(classify.last_classify_report())

    def test_the_task_runs_the_pass_and_returns_its_counts(self) -> None:
        """Contract: N5"""
        counts = {"processed": 3, "nsfw_tagged": 1, "nsfw_untagged": 0}
        with mock.patch.object(scraper, "classify_images", return_value=counts) as run, \
             mock.patch.object(scraper, "vision_config_from_settings", return_value="vision"):
            result = tasks.run_classify()
        self.assertEqual(result, {"ok": True, **counts})
        self.assertEqual(run.call_args.kwargs["vision"], "vision")

    def test_a_crashing_pass_is_logged_under_its_own_source(self) -> None:
        """Contract: N5"""
        LogEntry.objects.filter(source="classify").delete()
        with mock.patch.object(scraper, "classify_images", side_effect=RuntimeError("boom")), \
             mock.patch.object(scraper, "vision_config_from_settings"):
            result = tasks.run_classify()
        self.assertEqual(result, {"ok": False, "error": "boom"})
        self.assertEqual(
            list(LogEntry.objects.filter(source="classify").values_list("message", flat=True)),
            ["Classify failed: boom"],
        )

    def test_config_page_shows_the_counts_and_the_button(self) -> None:
        """Contract: N5, N8"""
        _row(flagged=True, judged=True, seed=1)
        _row(flagged=False, judged=True, seed=2)
        _row(flagged=True, judged=False, seed=3)

        response = self.client.get(reverse("config"))

        self.assertContains(response, "1 flagged and 1 safe")
        self.assertContains(response, "1 model flag awaiting your decision.")
        self.assertContains(response, reverse("trigger_classify"))


# ── N6: the 10/10 rule ───────────────────────────────────────────────────────


class TrainingThresholdTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.lines: list[str] = []
        self.sink = logger.add(lambda m: self.lines.append(m.record["message"]), level="INFO")

    def tearDown(self) -> None:
        logger.remove(self.sink)
        self._tmp.cleanup()

    def test_the_rule_itself(self) -> None:
        """Contract: N6"""
        self.assertTrue(nsfw.has_enough_examples(10, 10))
        self.assertFalse(nsfw.has_enough_examples(9, 10))
        self.assertFalse(nsfw.has_enough_examples(10, 9))
        self.assertEqual((nsfw.MIN_NSFW_EXAMPLES, nsfw.MIN_SAFE_EXAMPLES), (10, 10))

    def _rated_pair(self) -> None:
        for score, seed in ((5, 101), (1, 102)):
            row = _row(score=score, judged=True, seed=seed)
            _write_png(self.data_dir, row.file_path)

    def _decided(self, flagged: int, safe: int) -> None:
        for i in range(flagged):
            _write_png(self.data_dir, _row(flagged=True, judged=True, seed=200 + i).file_path)
        for i in range(safe):
            _write_png(self.data_dir, _row(flagged=False, judged=True, seed=300 + i).file_path)

    def test_too_few_examples_keep_the_previous_head_and_say_so(self) -> None:
        """Contract: N6 (the rated pair is decided and unflagged, so safe is 10 + 2)"""
        self._rated_pair()
        self._decided(flagged=9, safe=10)
        nsfw_path = self.data_dir / "nsfw.pkl"
        nsfw_path.write_bytes(b"previous head")

        trainer.run(
            data_dir=self.data_dir,
            weights_path=self.data_dir / "taste.pkl",
            nsfw_weights_path=nsfw_path,
        )

        self.assertEqual(nsfw_path.read_bytes(), b"previous head")
        self.assertIn(
            "NSFW head not trained: 9 flagged, 12 safe decided image(s), needs 10/10; "
            "keeping the previous head.",
            self.lines,
        )

    def test_enough_examples_train_the_head(self) -> None:
        """Contract: N6 (the rated safe pair counts toward the safe side: 10 + 2)"""
        self._rated_pair()
        self._decided(flagged=10, safe=8)
        nsfw_path = self.data_dir / "nsfw.pkl"

        trainer.run(
            data_dir=self.data_dir,
            weights_path=self.data_dir / "taste.pkl",
            nsfw_weights_path=nsfw_path,
        )

        self.assertIsNotNone(brain.load_classifier(nsfw_path))
        self.assertIn("Training NSFW classifier on 10 NSFW + 10 safe samples", self.lines)


# ── N7: the upgrade ──────────────────────────────────────────────────────────


class UpgradeBackfillTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()

    def test_rated_and_flagged_rows_become_decided_the_rest_stays_open(self) -> None:
        """Contract: N7"""
        rated_plain = _row(score=3, seed=1)
        rated_flag = _row(score=0, flagged=True, seed=2)
        unrated_flag = _row(flagged=True, seed=3)
        unseen = _row(seed=4)
        migration = importlib.import_module("ratings.migrations.0028_image_nsfw_judged")

        migration.mark_existing_decisions(apps, None)

        decided = {
            row.content_hash
            for row in Image.objects.filter(nsfw_judged=True)
        }
        self.assertEqual(decided, {rated_plain.content_hash, rated_flag.content_hash, unrated_flag.content_hash})
        unseen.refresh_from_db()
        self.assertFalse(unseen.nsfw_judged)


# ── N8: stats ────────────────────────────────────────────────────────────────


@override_settings(DEBUG=True)
class StatsLineTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        user = get_user_model().objects.create_user(f"st_{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def test_stats_shows_both_sides_and_the_open_model_flags(self) -> None:
        """Contract: N8"""
        _row(flagged=True, judged=True, seed=1)
        _row(flagged=True, judged=True, seed=2)
        _row(flagged=False, judged=True, score=4, seed=3)
        _row(flagged=True, judged=False, seed=4)
        _row(flagged=True, judged=False, seed=5)
        _row(flagged=True, judged=False, seed=6)
        _row(flagged=True, judged=False, purged=True, seed=7)

        response = self.client.get(reverse("stats"))

        self.assertContains(response, "2 flagged + 1 safe decided · 3 model flags awaiting your decision")


# ── N9: the taste side is untouched ──────────────────────────────────────────


class TasteSideUntouchedTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_taste_training_set_ignores_the_decision_flag(self) -> None:
        """Contract: N9"""
        decided = _row(score=5, judged=True, seed=1)
        open_ = _row(score=5, judged=False, seed=2)
        bad = _row(score=1, judged=False, seed=3)
        for row in (decided, open_, bad):
            _write_png(self.data_dir, row.file_path)

        good_paths, bad_paths = trainer.collect_image_paths(self.data_dir)

        self.assertEqual(set(good_paths), {self.data_dir / decided.file_path, self.data_dir / open_.file_path})
        self.assertEqual(bad_paths, [self.data_dir / bad.file_path])
