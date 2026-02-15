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


def test_main_tumblr_requires_blog(caplog: pytest.LogCaptureFixture) -> None:
    """main --source tumblr without --blog exits 1 and logs error."""
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["main.py", "--source", "tumblr"]):
            main()
    assert exc_info.value.code == 1
    assert "blog" in caplog.text.lower()


def test_main_tumblr_downloads_to_output_folder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """main --source tumblr --blog X --output_folder Y calls scraper and download."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tumblr_pics"
        with patch(
            "main.tumblr.iter_image_urls",
            return_value=["https://64.media.tumblr.com/abc/photo.jpg"],
        ):
            with patch(
                "main.tumblr.download_images", return_value=[out / "blog_abc.jpg"]
            ) as dl:
                with patch(
                    "sys.argv",
                    [
                        "main.py",
                        "--source",
                        "tumblr",
                        "--blog",
                        "staff",
                        "--output_folder",
                        str(out),
                        "--num_posts",
                        "20",
                    ],
                ):
                    main()
                dl.assert_called_once()
                call_args = dl.call_args
                assert call_args[0][0] == ["https://64.media.tumblr.com/abc/photo.jpg"]
                assert call_args[0][1] == out
                assert call_args[0][2] == "staff"
    assert "Downloaded 1 images" in caplog.text


def test_main_reddit_exits_with_message(caplog: pytest.LogCaptureFixture) -> None:
    """main --source reddit logs not implemented and exits 1."""
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["main.py", "--source", "reddit"]):
            main()
    assert exc_info.value.code == 1
    assert "not implemented" in caplog.text.lower()
