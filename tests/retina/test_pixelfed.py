"""Tests for retina.pixelfed (account-handle targeting via Mastodon-compatible API)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from retina import _mastoapi, pixelfed
from tests.conftest import minimal_png_bytes


def test_pixelfed_module_imports() -> None:
    """pixelfed module can be imported."""
    assert pixelfed is not None


def test_filename_for_item_uses_pixelfed_prefix() -> None:
    """_filename_for_item produces a stable pixelfed_<slug>_<hash><ext> filename."""
    name = pixelfed._filename_for_item("@art@pixelfed.social", "https://pf.example/img.jpg")
    assert name.startswith("pixelfed_")
    # Same inputs → same output (deterministic)
    assert name == pixelfed._filename_for_item("@art@pixelfed.social", "https://pf.example/img.jpg")


def test_filename_for_item_differs_from_mastodon_prefix() -> None:
    """Pixelfed and Mastodon filenames for the same URL are distinct."""
    from retina import mastodon

    url = "https://example.com/photo.jpg"
    handle = "@user@pixelfed.social"
    pf_name = pixelfed._filename_for_item(handle, url)
    masto_name = mastodon._filename_for_item(handle, url)
    assert pf_name != masto_name
    assert pf_name.startswith("pixelfed_")
    assert masto_name.startswith("mastodon_")


def test_iter_image_items_returns_items_and_cursor() -> None:
    """iter_image_items returns (items, new_cursor) parsed from account statuses."""
    statuses = [
        {
            "id": "42",
            "url": "https://pf.example/p/1",
            "media_attachments": [{"type": "image", "url": "https://pf.example/img.jpg"}],
        }
    ]
    with patch.object(_mastoapi, "_lookup_account_id", return_value="99"):
        with patch.object(_mastoapi, "_fetch_statuses_page", side_effect=[statuses, []]):
            with patch.object(_mastoapi, "time") as mock_time:
                mock_time.sleep = lambda _: None
                items, cursor = pixelfed.iter_image_items("@art@pixelfed.social", since_id="10")
    assert items == [("https://pf.example/img.jpg", "https://pf.example/p/1")]
    assert cursor == "42"


def test_resume_pages_forward_through_multiple_pages() -> None:
    """Resume must walk forward with min_id and collect every new post, not just
    the newest page, when more than one page accumulated since the last cursor."""
    page_size = _mastoapi._PAGE_SIZE

    def _statuses(id_range):
        return [
            {
                "id": str(i),
                "url": f"https://pf.example/p/{i}",
                "media_attachments": [
                    {"type": "image", "url": f"https://pf.example/img{i}.jpg"}
                ],
            }
            for i in id_range
        ]

    full_page = _statuses(range(101, 101 + page_size))  # full → keep paging
    tail_page = _statuses(range(101 + page_size, 101 + page_size + 3))  # short → stop

    with patch.object(_mastoapi, "_lookup_account_id", return_value="99"):
        with patch.object(
            _mastoapi, "_fetch_statuses_page", side_effect=[full_page, tail_page]
        ) as fetch:
            with patch.object(_mastoapi, "time") as mock_time:
                mock_time.sleep = lambda _: None
                items, cursor = pixelfed.iter_image_items(
                    "@art@pixelfed.social", since_id="100"
                )

    # Every post from both pages collected; cursor = highest id seen.
    assert len(items) == page_size + 3
    assert cursor == str(101 + page_size + 2)
    # Page 0 starts at the stored cursor; page 1 advances forward via min_id.
    assert fetch.call_args_list[0].kwargs["min_id"] == "100"
    assert fetch.call_args_list[1].kwargs["min_id"] == str(101 + page_size - 1)


def test_iter_image_items_returns_empty_on_bad_handle() -> None:
    """iter_image_items returns ([], None) for a malformed account handle."""
    items, cursor = pixelfed.iter_image_items("not-a-valid-handle")
    assert items == []
    assert cursor is None


def test_iter_image_items_skips_boosts() -> None:
    """iter_image_items skips statuses that are boosts (reblog is set)."""
    statuses = [
        {
            "id": "1",
            "reblog": {"id": "original"},
            "url": "https://pf.example/p/1",
            "media_attachments": [{"type": "image", "url": "https://pf.example/img.jpg"}],
        },
        {
            "id": "2",
            "url": "https://pf.example/p/2",
            "media_attachments": [{"type": "image", "url": "https://pf.example/img2.jpg"}],
        },
    ]
    with patch.object(_mastoapi, "_lookup_account_id", return_value="99"):
        with patch.object(_mastoapi, "_fetch_statuses_page", side_effect=[statuses, []]):
            with patch.object(_mastoapi, "time") as mock_time:
                mock_time.sleep = lambda _: None
                items, _ = pixelfed.iter_image_items("@art@pixelfed.social")
    image_urls = [url for url, _ in items]
    assert "https://pf.example/img.jpg" not in image_urls
    assert "https://pf.example/img2.jpg" in image_urls


def test_download_images_writes_file_and_returns_path() -> None:
    """download_images downloads image and returns (path, post_url, account_handle)."""
    items = [("https://pf.example/img.jpg", "https://pf.example/p/user/1")]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina._mastoapi.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = minimal_png_bytes()
            paths = pixelfed.download_images(
                items, out, "@art@pixelfed.social", rate_limit_sec=0
            )
        assert len(paths) == 1
        path, source_url, source_label = paths[0]
        assert path.exists()
        assert source_url == "https://pf.example/p/user/1"
        assert source_label == "@art@pixelfed.social"
        assert path.name.startswith("pixelfed_")


def test_download_images_skips_if_already_exists() -> None:
    """download_images skips when destination file already exists."""
    items = [("https://pf.example/img.jpg", "https://pf.example/p/1")]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        filename = pixelfed._filename_for_item("@art@pixelfed.social", "https://pf.example/img.jpg")
        (out / filename).write_bytes(b"existing")
        with patch("retina._mastoapi.urlopen") as mock_urlopen:
            paths = pixelfed.download_images(
                items, out, "@art@pixelfed.social", rate_limit_sec=0
            )
        mock_urlopen.assert_not_called()
        assert paths == []


def test_download_images_skips_corrupted_download() -> None:
    """download_images removes file when content is not a valid image."""
    items = [("https://pf.example/bad.jpg", "https://pf.example/p/1")]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina._mastoapi.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = b"not an image"
            paths = pixelfed.download_images(
                items, out, "@art@pixelfed.social", rate_limit_sec=0
            )
        assert paths == []
        assert not any(out.iterdir())
