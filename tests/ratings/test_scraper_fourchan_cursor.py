"""4chan incremental cursor wiring in scraper.run + _cursor_int parsing."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.test import TestCase, override_settings

from ratings import scraper
from ratings.models import Source


def test_cursor_int_parses_or_none() -> None:
    assert scraper._cursor_int("250") == 250
    assert scraper._cursor_int(None) is None
    assert scraper._cursor_int("") is None
    assert scraper._cursor_int("not-a-number") is None


_EMPTY_SOURCES = {
    "boards": [],
    "topics": [],
    "blogs": [],
    "pixelfed_accounts": [],
    "mastodon_accounts": [],
}


@override_settings(DEBUG=True)
class FourchanCursorWiringTests(TestCase):
    """run() passes the stored cursor into the scraper and persists the new one."""

    def _run_with_board(self, board: str, iter_return):
        sources = {**_EMPTY_SOURCES, "boards": [board]}
        with (
            patch.object(scraper, "_load_sources", return_value=sources),
            patch.object(scraper.dedup.DedupIndex, "from_db", return_value=MagicMock()),
            patch.object(scraper.brain, "get_encoder", return_value=MagicMock()),
            patch.object(scraper.brain, "get_transform", return_value=MagicMock()),
            patch.object(scraper, "_process_downloads", return_value=0),
            patch.object(scraper.fourchan, "download_images", return_value=[]),
            patch.object(
                scraper.fourchan, "iter_image_urls", return_value=iter_return
            ) as mock_iter,
        ):
            scraper.run(
                config_path=Path("/nonexistent/config.toml"),
                data_dir=Path("/tmp/memelord-test-data"),
            )
        return mock_iter

    def test_reads_stored_cursor_and_persists_new_one(self) -> None:
        Source.objects.create(type=Source.FOURCHAN, name="wg", cursor="100")

        mock_iter = self._run_with_board("wg", (["u"], 250))

        # The stored cursor was parsed and passed as since_modified.
        _, kwargs = mock_iter.call_args
        self.assertEqual(kwargs["since_modified"], 100)
        # The returned high-water mark was written back as text.
        self.assertEqual(Source.objects.get(type=Source.FOURCHAN, name="wg").cursor, "250")

    def test_first_scrape_passes_none_cursor(self) -> None:
        Source.objects.create(type=Source.FOURCHAN, name="wg", cursor=None)

        mock_iter = self._run_with_board("wg", (["u"], 42))

        _, kwargs = mock_iter.call_args
        self.assertIsNone(kwargs["since_modified"])
        self.assertEqual(Source.objects.get(type=Source.FOURCHAN, name="wg").cursor, "42")

    def test_no_source_row_does_not_crash(self) -> None:
        # Config-fallback mode (board name with no Source row): nothing to persist.
        mock_iter = self._run_with_board("wg", (["u"], 99))

        _, kwargs = mock_iter.call_args
        self.assertIsNone(kwargs["since_modified"])
        self.assertFalse(Source.objects.filter(name="wg").exists())


@override_settings(DEBUG=True)
class PerSourceErrorIsolationTests(TestCase):
    """A failure (e.g. an escaped socket timeout) in one source must not abort
    the whole batch — run() guards each per-source block and continues."""

    def test_one_source_raising_does_not_abort_later_sources(self) -> None:
        # Two boards: the first raises a socket-style read timeout that escapes
        # the scraper's own handlers; the second must still be scraped.
        sources = {**_EMPTY_SOURCES, "boards": ["bad", "good"]}

        def iter_side_effect(board, **kwargs):
            if board == "bad":
                raise TimeoutError("read timed out")
            return (["u"], 7)

        with (
            patch.object(scraper, "_load_sources", return_value=sources),
            patch.object(scraper.dedup.DedupIndex, "from_db", return_value=MagicMock()),
            patch.object(scraper.brain, "get_encoder", return_value=MagicMock()),
            patch.object(scraper.brain, "get_transform", return_value=MagicMock()),
            patch.object(scraper, "_process_downloads", return_value=3),
            patch.object(scraper.fourchan, "download_images", return_value=[]),
            patch.object(
                scraper.fourchan, "iter_image_urls", side_effect=iter_side_effect
            ),
        ):
            counts = scraper.run(
                config_path=Path("/nonexistent/config.toml"),
                data_dir=Path("/tmp/memelord-test-data"),
            )

        # The bad board produced no count entry; the good board was still reached.
        self.assertNotIn("4chan/bad", counts)
        self.assertEqual(counts["4chan/good"], 3)
