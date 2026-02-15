"""Tests for retina.imgur."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from retina import imgur


def test_imgur_module_imports() -> None:
    """Imgur scraper module can be imported."""
    assert imgur is not None


def test_image_urls_from_item_single_image() -> None:
    """Single image item returns its link."""
    item = {"id": "x", "link": "https://i.imgur.com/abc.jpg"}
    assert imgur.image_urls_from_item(item) == ["https://i.imgur.com/abc.jpg"]


def test_image_urls_from_item_album() -> None:
    """Album item with images array returns all links."""
    item = {
        "id": "y",
        "images": [
            {"link": "https://i.imgur.com/1.jpg"},
            {"link": "https://i.imgur.com/2.png"},
        ],
    }
    assert imgur.image_urls_from_item(item) == [
        "https://i.imgur.com/1.jpg",
        "https://i.imgur.com/2.png",
    ]


def test_image_urls_from_item_empty_album() -> None:
    """Album with empty images falls back to item link if present."""
    item = {"id": "z", "images": [], "link": "https://i.imgur.com/one.jpg"}
    assert imgur.image_urls_from_item(item) == ["https://i.imgur.com/one.jpg"]


def test_image_urls_from_item_returns_empty_for_no_link() -> None:
    """Item with no link and no images returns empty list."""
    assert imgur.image_urls_from_item({}) == []
    assert imgur.image_urls_from_item({"id": "x"}) == []


def test_image_urls_from_item_returns_empty_for_non_dict() -> None:
    """Non-dict item yields empty list."""
    assert imgur.image_urls_from_item([]) == []
    assert imgur.image_urls_from_item("x") == []


def test_iter_image_urls_uses_mocked_page() -> None:
    """iter_image_urls collects URLs from get_topic_page response."""
    fake_response = {
        "success": True,
        "data": {
            "items": [
                {"link": "https://i.imgur.com/a.jpg"},
                {
                    "images": [{"link": "https://i.imgur.com/b.png"}],
                },
            ],
        },
    }

    with patch.object(imgur, "get_topic_page", return_value=fake_response):
        urls = imgur.iter_image_urls(
            topic="funny",
            client_id="test_client_id",
            max_items=10,
            rate_limit_sec=0,
        )
    assert "https://i.imgur.com/a.jpg" in urls
    assert "https://i.imgur.com/b.png" in urls
    assert len(urls) == 2


def test_iter_image_urls_deduplicates() -> None:
    """Same image in multiple items appears only once."""
    fake_response = {
        "success": True,
        "data": {
            "items": [
                {"link": "https://i.imgur.com/same.jpg"},
                {"link": "https://i.imgur.com/same.jpg"},
            ],
        },
    }
    with patch.object(imgur, "get_topic_page", return_value=fake_response):
        urls = imgur.iter_image_urls(
            topic="funny",
            client_id="test_id",
            max_items=10,
            rate_limit_sec=0,
        )
    assert urls == ["https://i.imgur.com/same.jpg"]


def test_download_images_writes_files_with_topic_prefix() -> None:
    """download_images writes files with stable topic_hash.ext naming."""
    fake_content = b"fake image bytes"
    url = "https://i.imgur.com/abc.jpg"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina.imgur.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = fake_content
            paths = imgur.download_images([url], out, "funny", rate_limit_sec=0)
        assert len(paths) == 1
        assert paths[0].read_bytes() == fake_content
        assert paths[0].name.startswith("funny_")
        assert paths[0].suffix == ".jpg"


def test_download_images_skips_if_already_in_output() -> None:
    """download_images skips URL when file already exists in output_dir."""
    url = "https://i.imgur.com/xyz.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        name = imgur._filename_for_url("funny", url)
        (out / name).write_bytes(b"already here")
        with patch("retina.imgur.urlopen") as mock_urlopen:
            paths = imgur.download_images([url], out, "funny", rate_limit_sec=0)
        assert len(paths) == 0
        mock_urlopen.assert_not_called()


def test_get_topic_page_normalizes_topic_slug() -> None:
    """get_topic_page strips t/ prefix and uses slug in API path."""
    with patch.object(
        imgur, "_get_json", return_value={"success": True, "data": {"items": []}}
    ) as mock_get:
        imgur.get_topic_page("t/funny", 0, "cid")
        call_url = mock_get.call_args[0][0]
        assert "funny" in call_url
        assert "gallery/t/" in call_url


def test_is_gallery_url_filters_banner_and_logo() -> None:
    """_is_gallery_url returns False for logo/banner-style URLs."""
    assert imgur._is_gallery_url("https://i.imgur.com/abc.jpg") is True
    assert imgur._is_gallery_url("https://i.imgur.com/logo-1200.png") is False
    assert imgur._is_gallery_url("https://i.imgur.com/banner.jpg") is False


def test_extract_image_urls_from_html_finds_direct_links() -> None:
    """_extract_image_urls_from_html finds https://i.imgur.com/... URLs."""
    html = (
        "foo https://i.imgur.com/abc123.jpg bar https://i.imgur.com/xyz.png?size=medium"
    )
    urls = imgur._extract_image_urls_from_html(html)
    assert "https://i.imgur.com/abc123.jpg" in urls
    assert "https://i.imgur.com/xyz.png" in urls
    assert len(urls) == 2


def test_extract_image_urls_from_html_deduplicates() -> None:
    """Repeated URLs appear only once."""
    html = "https://i.imgur.com/same.jpg https://i.imgur.com/same.jpg"
    urls = imgur._extract_image_urls_from_html(html)
    assert urls == ["https://i.imgur.com/same.jpg"]


def test_iter_image_urls_without_client_id_uses_scrape() -> None:
    """iter_image_urls with no client_id calls get_topic_page_scrape."""
    with patch.object(
        imgur,
        "get_topic_page_scrape",
        return_value=["https://i.imgur.com/scraped.jpg"],
    ) as mock_scrape:
        urls = imgur.iter_image_urls(
            topic="funny",
            client_id="",
            max_items=10,
        )
    mock_scrape.assert_called()
    assert urls == ["https://i.imgur.com/scraped.jpg"]


def test_get_topic_page_scrape_normalizes_topic() -> None:
    """get_topic_page_scrape builds URL with normalized slug."""
    with patch.object(imgur, "_fetch_html", return_value="no images") as mock_fetch:
        imgur.get_topic_page_scrape("t/funny", 0)
        call_url = mock_fetch.call_args[0][0]
        assert call_url == "https://imgur.com/t/funny"
