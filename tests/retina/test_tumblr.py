"""Tests for retina.tumblr."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from retina import tumblr
from tests.conftest import minimal_png_bytes


def test_tumblr_module_imports() -> None:
    """Tumblr scraper module can be imported."""
    assert tumblr is not None


def test_image_urls_from_post_returns_url_for_photo_url_key() -> None:
    """Photo post with photo-url-500 yields that URL."""
    post = {"type": "photo", "photo-url-500": "https://example.com/photo.jpg"}
    assert tumblr.image_urls_from_post(post) == ["https://example.com/photo.jpg"]


def test_image_urls_from_post_collects_all_sizes() -> None:
    """When multiple photo-url-* keys exist, returns all (order: 1280, 500, 400, 250)."""
    post = {
        "type": "photo",
        "photo-url-250": "https://a.com/small.jpg",
        "photo-url-500": "https://a.com/med.jpg",
        "photo-url-1280": "https://a.com/large.jpg",
    }
    urls = tumblr.image_urls_from_post(post)
    assert "https://a.com/large.jpg" in urls
    assert "https://a.com/med.jpg" in urls
    assert "https://a.com/small.jpg" in urls
    assert len(urls) == 3


def test_image_urls_from_post_returns_empty_for_non_photo() -> None:
    """Text post yields no URLs."""
    post = {"type": "text", "regular-body": "<p>Hi</p>"}
    assert tumblr.image_urls_from_post(post) == []


def test_image_urls_from_post_returns_empty_for_empty_dict() -> None:
    """Empty or non-dict yields empty list."""
    assert tumblr.image_urls_from_post({}) == []
    assert tumblr.image_urls_from_post({"type": "photo"}) == []


def test_image_urls_from_post_uses_photos_array_when_no_photo_url_keys() -> None:
    """Photo post with photos[].original_size yields URL."""
    post = {
        "type": "photo",
        "photos": [
            {"original_size": {"url": "https://media.tumblr.com/abc.png"}},
        ],
    }
    assert tumblr.image_urls_from_post(post) == ["https://media.tumblr.com/abc.png"]


def test_image_urls_from_post_returns_none_for_non_dict() -> None:
    """Non-dict post yields empty list."""
    assert tumblr.image_urls_from_post([]) == []
    assert tumblr.image_urls_from_post("x") == []


def test_iter_image_urls_uses_mocked_posts() -> None:
    """iter_image_urls collects URLs from get_posts response."""
    fake_response = {
        "posts": [
            {"type": "photo", "photo-url-500": "https://x.com/1.jpg"},
            {
                "type": "photo",
                "photos": [{"original_size": {"url": "https://x.com/2.png"}}],
            },
        ],
    }

    with patch.object(tumblr, "get_posts", return_value=fake_response):
        urls = tumblr.iter_image_urls(
            blog="testblog",
            num_posts=20,
            rate_limit_sec=0,
        )
    assert "https://x.com/1.jpg" in urls
    assert "https://x.com/2.png" in urls
    assert len(urls) == 2


def test_iter_image_urls_deduplicates() -> None:
    """Same image in multiple posts appears only once."""
    fake_response = {
        "posts": [
            {"type": "photo", "photo-url-500": "https://same.com/img.jpg"},
            {"type": "photo", "photo-url-500": "https://same.com/img.jpg"},
        ],
    }
    with patch.object(tumblr, "get_posts", return_value=fake_response):
        urls = tumblr.iter_image_urls(
            blog="testblog",
            num_posts=20,
            rate_limit_sec=0,
        )
    assert urls == ["https://same.com/img.jpg"]


def test_download_images_writes_files_with_blog_prefix() -> None:
    """download_images writes files with stable blog_hash.ext naming."""
    content = minimal_png_bytes()
    url = "https://64.media.tumblr.com/abc/photo.jpg"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina.tumblr.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = content
            paths = tumblr.download_images([url], out, "myblog", rate_limit_sec=0)
        assert len(paths) == 1
        assert paths[0].read_bytes() == content
        assert paths[0].parent == out
        assert paths[0].name.startswith("myblog_")
        assert paths[0].suffix == ".jpg"


def test_download_images_creates_directory() -> None:
    """download_images creates output_dir if it does not exist."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "dir"
        with patch("retina.tumblr.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = minimal_png_bytes()
            tumblr.download_images(
                ["https://x.com/photo.jpg"], out, "blog", rate_limit_sec=0
            )
        assert out.exists()
        assert next(out.iterdir()).name.startswith("blog_")


def test_download_images_skips_if_already_in_output() -> None:
    """download_images skips URL when file already exists in output_dir."""
    url = "https://64.media.tumblr.com/xyz/photo.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # Stable filename for this URL
        name = tumblr._filename_for_url("blog", url)
        (out / name).write_bytes(b"already here")
        with patch("retina.tumblr.urlopen") as mock_urlopen:
            paths = tumblr.download_images([url], out, "blog", rate_limit_sec=0)
        assert len(paths) == 0
        mock_urlopen.assert_not_called()
        assert (out / name).read_bytes() == b"already here"


def test_download_images_skips_if_already_in_skip_dirs() -> None:
    """download_images skips when file exists in one of skip_dirs."""
    url = "https://64.media.tumblr.com/a/b.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        name = tumblr._filename_for_url("blog", url)
        (corpus / name).write_bytes(b"in corpus")
        with patch("retina.tumblr.urlopen") as mock_urlopen:
            paths = tumblr.download_images(
                [url], out, "blog", skip_dirs=[corpus], rate_limit_sec=0
            )
        assert len(paths) == 0
        mock_urlopen.assert_not_called()
        assert not (out / name).exists()
