"""
Flickr wiring in ratings.scraper: cursor, error isolation, labels, sources.

Contract V1–V11 lives in tests/retina/test_flickr.py; this module pins the
parts that need the scrape run and the DB: V3 (the feed leaves the cursor
alone), V5 (the cursor moves only after the images are in the DB), V8 (one
failing source does not stop the others), V9 (the key is not stored), V10
(the source name is the label) and the config.toml path of V1/V2.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.test import TestCase, override_settings

from ratings import scraper
from ratings.models import Source

_EMPTY_SOURCES = {
    "boards": [],
    "topics": [],
    "blogs": [],
    "pixelfed_accounts": [],
    "mastodon_accounts": [],
    "flickr_sources": [],
}
_GROUP = "group/419512@N22"
_DOWNLOAD = [(Path("/tmp/flickr_1.jpg"), "https://live.staticflickr.com/1_s_b.jpg", _GROUP)]


@override_settings(DEBUG=True, FLICKR_API_KEY="SECRETKEY123")
class FlickrScrapeRunTests(TestCase):
    def setUp(self) -> None:
        Source.objects.all().delete()

    def _run(self, names, iter_side_effect, process=None):
        sources = {**_EMPTY_SOURCES, "flickr_sources": names}
        with (
            patch.object(scraper, "_load_sources", return_value=sources),
            patch.object(scraper.dedup.DedupIndex, "from_db", return_value=MagicMock()),
            patch.object(scraper.brain, "get_encoder", return_value=MagicMock()),
            patch.object(scraper.brain, "get_transform", return_value=MagicMock()),
            patch.object(scraper, "_process_downloads", side_effect=process or (lambda *a: 1)) as mock_process,
            patch.object(scraper.flickr, "download_images", return_value=_DOWNLOAD) as mock_download,
            patch.object(scraper.flickr, "iter_image_items", side_effect=iter_side_effect) as mock_iter,
        ):
            counts = scraper.run(
                config_path=Path("/nonexistent/config.toml"),
                data_dir=Path("/tmp/memelord-test-data"),
            )
        return counts, mock_iter, mock_download, mock_process

    def test_stored_cursor_and_key_go_in_and_the_new_cursor_is_saved(self) -> None:
        """Contract: V4, V5"""
        Source.objects.create(type=Source.FLICKR, name=_GROUP, cursor="100")
        counts, mock_iter, _, _ = self._run([_GROUP], lambda *a, **k: ([("1", "u", None)], "250"))

        _, kwargs = mock_iter.call_args
        self.assertEqual(kwargs["since"], "100")
        self.assertEqual(kwargs["api_key"], "SECRETKEY123")
        self.assertEqual(Source.objects.get(name=_GROUP).cursor, "250")
        self.assertEqual(counts[f"flickr/{_GROUP}"], 1)

    def test_cursor_stays_when_the_images_did_not_reach_the_db(self) -> None:
        """Contract: V5, V8"""
        Source.objects.create(type=Source.FLICKR, name=_GROUP, cursor="100")

        def process(*args):
            raise OSError("disk full")

        counts, _, _, _ = self._run([_GROUP], lambda *a, **k: ([("1", "u", None)], "250"), process)
        self.assertEqual(Source.objects.get(name=_GROUP).cursor, "100")
        self.assertNotIn(f"flickr/{_GROUP}", counts)

    def test_feed_run_leaves_the_cursor_alone(self) -> None:
        """Contract: V3"""
        Source.objects.create(type=Source.FLICKR, name=_GROUP, cursor="100")
        self._run([_GROUP], lambda *a, **k: ([("1", "u", "s")], None))
        self.assertEqual(Source.objects.get(name=_GROUP).cursor, "100")

    @override_settings(FLICKR_API_KEY="")
    def test_no_key_selects_the_feed(self) -> None:
        """Contract: V3"""
        _, mock_iter, _, _ = self._run([_GROUP], lambda *a, **k: ([], None))
        _, kwargs = mock_iter.call_args
        self.assertIsNone(kwargs["api_key"])

    def test_one_failing_source_does_not_stop_the_next(self) -> None:
        """Contract: V8"""
        Source.objects.create(type=Source.FLICKR, name="group/1@N01", cursor="5")
        Source.objects.create(type=Source.FLICKR, name="group/2@N02", cursor="5")

        def iter_side_effect(name, **kwargs):
            if name == "group/1@N01":
                raise TimeoutError("read timed out")
            return ([("1", "u", None)], "9")

        counts, _, _, _ = self._run(["group/1@N01", "group/2@N02"], iter_side_effect)
        self.assertNotIn("flickr/group/1@N01", counts)
        self.assertEqual(counts["flickr/group/2@N02"], 1)
        self.assertEqual(Source.objects.get(name="group/1@N01").cursor, "5")
        self.assertEqual(Source.objects.get(name="group/2@N02").cursor, "9")

    def test_source_name_is_the_download_label_and_the_key_is_not_stored(self) -> None:
        """Contract: V9, V10"""
        Source.objects.create(type=Source.FLICKR, name=_GROUP)
        _, _, mock_download, mock_process = self._run([_GROUP], lambda *a, **k: ([("1", "u", None)], "7"))
        download_args, _ = mock_download.call_args
        self.assertEqual(download_args[2], _GROUP)
        process_args = mock_process.call_args[0]
        self.assertEqual(process_args[0], _DOWNLOAD)
        row = Source.objects.get(name=_GROUP)
        self.assertNotIn("SECRETKEY123", f"{row.name}{row.cursor}")


class FlickrSourceLoadingTests(TestCase):
    def setUp(self) -> None:
        Source.objects.all().delete()

    def _config(self, text: bytes) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as fh:
            fh.write(text)
        self.addCleanup(Path(fh.name).unlink, missing_ok=True)
        return Path(fh.name)

    def test_enabled_flickr_rows_are_loaded(self) -> None:
        Source.objects.create(type=Source.FLICKR, name=_GROUP)
        Source.objects.create(type=Source.FLICKR, name="user/off", enabled=False)
        sources = scraper._load_sources(self._config(b""))
        self.assertEqual(sources["flickr_sources"], [_GROUP])

    def test_config_toml_entries_are_normalized_and_invalid_ones_dropped(self) -> None:
        """Contract: V1, V2"""
        config = self._config(
            b'[flickr]\nsources = ["https://www.flickr.com/groups/419512@N22/pool/", '
            b'"https://example.com/x", "user/alexgee"]\n'
        )
        self.assertEqual(scraper._load_sources(config)["flickr_sources"], [_GROUP, "user/alexgee"])

    def test_import_from_config_creates_canonical_rows_once(self) -> None:
        """Contract: V1"""
        config = self._config(
            b'[flickr]\nsources = ["https://www.flickr.com/groups/419512@N22/", "group/419512@N22"]\n'
        )
        self.assertEqual(scraper.import_from_config(config), 1)
        self.assertEqual(scraper.import_from_config(config), 0)
        self.assertEqual(
            list(Source.objects.filter(type=Source.FLICKR).values_list("name", flat=True)), [_GROUP]
        )


@override_settings(DEBUG=True, FLICKR_API_KEY="SECRETKEY123")
class FlickrKeyNeverLoggedTests(TestCase):
    """
    Contract: V9 through the whole scrape run.

    run() logs an escaping exception with logger.exception, and loguru's default
    diagnose mode prints the local variables of every frame, among them the
    query dict that holds the key. So no exception from the API path may leave
    retina.flickr; this test drives a read error that is not an OSError
    (http.client.IncompleteRead) through the real iter_image_items.
    """

    def test_key_is_absent_from_messages_and_tracebacks(self) -> None:
        from http.client import IncompleteRead

        from loguru import logger

        Source.objects.create(type=Source.FLICKR, name=_GROUP)
        lines: list[str] = []
        sink_id = logger.add(
            lines.append, format="{message}\n{exception}", diagnose=True, backtrace=True, level="DEBUG"
        )
        sources = {**_EMPTY_SOURCES, "flickr_sources": [_GROUP]}
        try:
            with (
                patch.object(scraper, "_load_sources", return_value=sources),
                patch.object(scraper.dedup.DedupIndex, "from_db", return_value=MagicMock()),
                patch.object(scraper.brain, "get_encoder", return_value=MagicMock()),
                patch.object(scraper.brain, "get_transform", return_value=MagicMock()),
                patch.object(scraper.flickr.time, "sleep"),
                patch.object(scraper.flickr, "_get_json", side_effect=IncompleteRead(b"partial")),
            ):
                counts = scraper.run(
                    config_path=Path("/nonexistent/config.toml"),
                    data_dir=Path("/tmp/memelord-test-data"),
                )
        finally:
            logger.remove(sink_id)

        self.assertEqual(counts[f"flickr/{_GROUP}"], 0)
        output = "".join(lines)
        self.assertIn("IncompleteRead", output)
        self.assertNotIn("SECRETKEY123", output)
