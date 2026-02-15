"""Tests for main CLI."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from main import main


def test_main_4chan_downloads_to_output_folder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """main --source 4chan --board wg --output_folder X calls scraper and download."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pics"
        with patch(
            "main.fourchan.iter_image_urls",
            return_value=["https://i.4cdn.org/wg/1.jpg"],
        ):
            with patch(
                "main.fourchan.download_images", return_value=[out / "1.jpg"]
            ) as dl:
                with patch(
                    "sys.argv",
                    [
                        "main.py",
                        "--source",
                        "4chan",
                        "--board",
                        "wg",
                        "--output_folder",
                        str(out),
                        "--index_pages",
                        "1",
                    ],
                ):
                    main()
                dl.assert_called_once()
                call_args = dl.call_args
                assert call_args[0][0] == ["https://i.4cdn.org/wg/1.jpg"]
                assert call_args[0][1] == out
                assert call_args[0][2] == "wg"
                assert "skip_dirs" in call_args[1]
                assert len(call_args[1]["skip_dirs"]) == 2  # data/corpus, data/void
    assert "Downloaded 1 images" in caplog.text
    assert str(out) in caplog.text


def test_main_4chan_accepts_defaults() -> None:
    """main --source 4chan uses default board and output_folder."""
    with patch("main.fourchan.iter_image_urls", return_value=[]):
        with patch("sys.argv", ["main.py", "--source", "4chan", "--index_pages", "1"]):
            main()
        from main import fourchan

        fourchan.iter_image_urls.assert_called_once_with(board="wg", index_pages=1)


def test_main_reddit_exits_with_message(caplog: pytest.LogCaptureFixture) -> None:
    """main --source reddit logs not implemented and exits 1."""
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["main.py", "--source", "reddit"]):
            main()
    assert exc_info.value.code == 1
    assert "not implemented" in caplog.text.lower()
