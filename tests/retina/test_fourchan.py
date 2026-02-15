"""Tests for retina.fourchan."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from retina import fourchan
from tests.conftest import minimal_png_bytes


def test_fourchan_module_imports() -> None:
    """Fourchan scraper module can be imported."""
    assert fourchan is not None


def test_image_url_from_post_returns_url_when_valid() -> None:
    """Valid post with tim and image ext yields full image URL."""
    post = {"tim": 1234567890123, "ext": ".jpg"}
    assert fourchan.image_url_from_post(post, "wg") == (
        "https://i.4cdn.org/wg/1234567890123.jpg"
    )


def test_image_url_from_post_returns_none_when_filedeleted() -> None:
    """Post with filedeleted does not yield URL."""
    post = {"tim": 1, "ext": ".png", "filedeleted": 1}
    assert fourchan.image_url_from_post(post, "wg") is None


def test_image_url_from_post_returns_none_when_no_tim() -> None:
    """Post without tim does not yield URL."""
    post = {"ext": ".jpg"}
    assert fourchan.image_url_from_post(post, "wg") is None


def test_image_url_from_post_returns_none_when_non_image_ext() -> None:
    """Post with .webm does not yield URL (only static images)."""
    post = {"tim": 1, "ext": ".webm"}
    assert fourchan.image_url_from_post(post, "wg") is None


def test_image_url_from_post_accepts_png_and_gif() -> None:
    """PNG and GIF extensions are accepted."""
    assert fourchan.image_url_from_post({"tim": 1, "ext": ".png"}, "wg") == (
        "https://i.4cdn.org/wg/1.png"
    )
    assert fourchan.image_url_from_post({"tim": 2, "ext": ".gif"}, "wg") == (
        "https://i.4cdn.org/wg/2.gif"
    )


def test_iter_image_urls_uses_mocked_index() -> None:
    """iter_image_urls collects URLs from index-like structure."""
    fake_index = {
        "threads": [
            {
                "posts": [
                    {"tim": 100, "ext": ".jpg"},
                    {"tim": 101, "ext": ".png"},
                ],
            },
            {"posts": [{"tim": 102, "ext": ".jpg", "filedeleted": 1}]},
        ],
    }

    with patch.object(fourchan, "get_index", return_value=fake_index):
        urls = fourchan.iter_image_urls(
            board="wg",
            index_pages=1,
            rate_limit_sec=0,
        )
    assert urls == [
        "https://i.4cdn.org/wg/100.jpg",
        "https://i.4cdn.org/wg/101.png",
    ]


def test_iter_image_urls_deduplicates() -> None:
    """Same image in multiple previews appears only once."""
    fake_index = {
        "threads": [
            {"posts": [{"tim": 42, "ext": ".jpg"}]},
            {"posts": [{"tim": 42, "ext": ".jpg"}]},
        ],
    }
    with patch.object(fourchan, "get_index", return_value=fake_index):
        urls = fourchan.iter_image_urls(board="wg", index_pages=1, rate_limit_sec=0)
    assert urls == ["https://i.4cdn.org/wg/42.jpg"]


def test_download_images_writes_files_with_board_prefix() -> None:
    """download_images writes files named {board}_{tim}{ext} with correct content."""
    content = minimal_png_bytes()
    url = "https://i.4cdn.org/wg/999.png"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = content
            paths = fourchan.download_images([url], out, "wg", rate_limit_sec=0)
        assert len(paths) == 1
        assert paths[0] == out / "wg_999.png"
        assert paths[0].read_bytes() == content


def test_download_images_creates_directory() -> None:
    """download_images creates output_dir if it does not exist."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "dir"
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = minimal_png_bytes()
            fourchan.download_images(
                ["https://i.4cdn.org/wg/1.jpg"], out, "wg", rate_limit_sec=0
            )
        assert (out / "wg_1.jpg").exists()


def test_download_images_skips_if_already_in_output() -> None:
    """download_images skips URL when file already exists in output_dir."""
    url = "https://i.4cdn.org/wg/42.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "wg_42.jpg").write_bytes(b"already here")
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            paths = fourchan.download_images([url], out, "wg", rate_limit_sec=0)
        assert len(paths) == 0
        mock_urlopen.assert_not_called()
        assert (out / "wg_42.jpg").read_bytes() == b"already here"


def test_download_images_skips_if_already_in_skip_dirs() -> None:
    """download_images skips URL when file exists in one of skip_dirs (e.g. corpus)."""
    url = "https://i.4cdn.org/wg/99.png"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        (corpus / "wg_99.png").write_bytes(b"in corpus")
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            paths = fourchan.download_images(
                [url], out, "wg", skip_dirs=[corpus], rate_limit_sec=0
            )
        assert len(paths) == 0
        mock_urlopen.assert_not_called()
        assert not (out / "wg_99.png").exists()


def test_download_images_skips_if_path_in_skip_paths() -> None:
    """download_images skips URL when destination path is in skip_paths (e.g. from DB)."""
    url = "https://i.4cdn.org/wg/77.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        dest = (out / "wg_77.jpg").resolve()
        skip_paths = {str(dest)}
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            paths = fourchan.download_images(
                [url], out, "wg", skip_paths=skip_paths, rate_limit_sec=0
            )
        assert len(paths) == 0
        mock_urlopen.assert_not_called()
        assert not (out / "wg_77.jpg").exists()


def test_download_images_removes_corrupted_download() -> None:
    """download_images removes file and does not return it when content is not a valid image."""
    url = "https://i.4cdn.org/wg/123.png"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = b"corrupted or not an image"
            paths = fourchan.download_images([url], out, "wg", rate_limit_sec=0)
        assert len(paths) == 0
        assert not (out / "wg_123.png").exists()
