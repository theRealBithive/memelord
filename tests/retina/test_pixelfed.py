"""Tests for retina.pixelfed."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from retina import pixelfed
from tests.conftest import minimal_png_bytes


def test_pixelfed_module_imports() -> None:
    """Pixelfed module can be imported."""
    assert pixelfed is not None


def test_normalize_base_strips_trailing_slash() -> None:
    """_normalize_base returns URL without trailing slash."""
    assert (
        pixelfed._normalize_base("https://pixelfed.social/")
        == "https://pixelfed.social"
    )
    assert (
        pixelfed._normalize_base("https://pixelfed.social") == "https://pixelfed.social"
    )


def test_fetch_public_timeline_returns_list_from_json() -> None:
    """fetch_public_timeline returns parsed list of statuses."""
    fake_timeline = [
        {"id": "1", "url": "https://pf.example/p/u/1", "media_attachments": []},
    ]
    with patch.object(pixelfed, "_get_json", return_value=fake_timeline):
        result = pixelfed.fetch_public_timeline(
            "https://pixelfed.social",
            limit=40,
            rate_limit_sec=0,
        )
    assert result == fake_timeline


def test_fetch_public_timeline_returns_empty_on_http_error() -> None:
    """fetch_public_timeline returns empty list on request failure."""
    from urllib.error import HTTPError

    with patch.object(
        pixelfed, "_get_json", side_effect=HTTPError("u", 500, "Err", None, None)
    ):
        result = pixelfed.fetch_public_timeline(
            "https://pixelfed.social",
            limit=10,
            rate_limit_sec=0,
        )
    assert result == []


def test_iter_image_items_yields_image_url_and_post_url() -> None:
    """iter_image_items returns (image_url, post_url) for each status with image media."""
    fake_timeline = [
        {
            "id": "1",
            "url": "https://pf.example/p/user/1",
            "uri": "https://pf.example/p/user/1",
            "media_attachments": [
                {"type": "image", "url": "https://pf.example/storage/abc.jpg"},
            ],
        },
        {
            "id": "2",
            "url": "https://pf.example/p/user/2",
            "media_attachments": [
                {"type": "image", "url": "https://pf.example/storage/def.png"},
            ],
        },
    ]
    with patch.object(
        pixelfed,
        "fetch_public_timeline",
        return_value=fake_timeline,
    ):
        items = pixelfed.iter_image_items(
            "https://pixelfed.social",
            limit=40,
            rate_limit_sec=0,
        )
    assert items == [
        ("https://pf.example/storage/abc.jpg", "https://pf.example/p/user/1"),
        ("https://pf.example/storage/def.png", "https://pf.example/p/user/2"),
    ]


def test_iter_image_items_skips_status_without_image() -> None:
    """iter_image_items skips statuses with no image attachment."""
    fake_timeline = [
        {"id": "1", "url": "https://pf.example/p/1", "media_attachments": []},
        {
            "id": "2",
            "url": "https://pf.example/p/2",
            "media_attachments": [{"type": "video", "url": "https://x/v.mp4"}],
        },
    ]
    with patch.object(pixelfed, "fetch_public_timeline", return_value=fake_timeline):
        items = pixelfed.iter_image_items(
            "https://pf.example", limit=10, rate_limit_sec=0
        )
    assert items == []


def test_iter_image_items_uses_uri_when_url_missing() -> None:
    """iter_image_items uses uri when url is not set."""
    fake_timeline = [
        {
            "id": "1",
            "uri": "https://pf.example/users/u/statuses/1",
            "media_attachments": [
                {"type": "image", "url": "https://pf.example/img.jpg"}
            ],
        },
    ]
    with patch.object(pixelfed, "fetch_public_timeline", return_value=fake_timeline):
        items = pixelfed.iter_image_items(
            "https://pf.example", limit=10, rate_limit_sec=0
        )
    assert items == [
        ("https://pf.example/img.jpg", "https://pf.example/users/u/statuses/1")
    ]


def test_iter_image_items_uses_browser_fallback_when_api_returns_empty() -> None:
    """When API returns no statuses, iter_image_items tries Playwright and uses result."""
    browser_statuses = [
        {
            "url": "https://pf.example/p/user/99",
            "media_attachments": [
                {"type": "image", "url": "https://pf.example/storage/x.jpg"}
            ],
        },
    ]
    with patch.object(pixelfed, "fetch_public_timeline", return_value=[]):
        with patch.object(
            pixelfed,
            "_fetch_timeline_with_browser",
            return_value=browser_statuses,
        ) as browser_mock:
            items = pixelfed.iter_image_items(
                "https://pf.example", limit=10, rate_limit_sec=0
            )
    browser_mock.assert_called_once()
    assert items == [
        ("https://pf.example/storage/x.jpg", "https://pf.example/p/user/99")
    ]


def test_download_images_writes_file_and_returns_path_with_post_url() -> None:
    """download_images downloads image and returns (path, post_url, 'pixelfed')."""
    items = [
        ("https://example.com/image.jpg", "https://pixelfed.social/p/user/1"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina.pixelfed.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = minimal_png_bytes()
            paths = pixelfed.download_images(items, out, rate_limit_sec=0)
        assert len(paths) == 1
        path, source_url, source_label = paths[0]
        assert path.exists()
        assert source_url == "https://pixelfed.social/p/user/1"
        assert source_label == "pixelfed"
        assert path.read_bytes() == minimal_png_bytes()


def test_download_images_skips_if_path_in_skip_paths() -> None:
    """download_images skips when destination path is in skip_paths."""
    items = [("https://example.com/photo.png", "https://pf.example/p/1")]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # Filename will be pixelfed_0_photo.png
        dest = (out / "pixelfed_0_photo.png").resolve()
        with patch("retina.pixelfed.urlopen") as mock_urlopen:
            paths = pixelfed.download_images(
                items,
                out,
                skip_paths={str(dest)},
                rate_limit_sec=0,
            )
        assert len(paths) == 0
        mock_urlopen.assert_not_called()


def test_download_images_removes_corrupted_download() -> None:
    """download_images removes file when content is not a valid image."""
    items = [("https://example.com/bad.jpg", "https://pf.example/p/1")]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina.pixelfed.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = b"not an image"
            paths = pixelfed.download_images(items, out, rate_limit_sec=0)
        assert len(paths) == 0
        assert not any(out.iterdir())
