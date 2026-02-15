"""Tests for core.mastodon: Mastodon client and post_image."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core import mastodon as mastodon_module


def test_create_client_returns_mastodon_instance() -> None:
    """create_client returns a Mastodon instance with given credentials."""
    client = mastodon_module.create_client("https://mastodon.example", "fake_token")
    assert client is not None
    assert client.access_token == "fake_token"


def test_post_image_uploads_with_alt_and_posts_status(
    tmp_path: Path,
) -> None:
    """post_image calls media_post with description and status_post with media_ids."""
    from tests.conftest import minimal_png_bytes

    img = tmp_path / "test.png"
    img.write_bytes(minimal_png_bytes())

    client = MagicMock()
    client.media_post.return_value = {"id": "123"}
    client.status_post.return_value = {"id": "456"}

    result = mastodon_module.post_image(
        client, img, alt_text="A test image", status_text=""
    )

    client.media_post.assert_called_once()
    call_kw = client.media_post.call_args[1]
    assert call_kw["mime_type"] == "image/png"
    assert call_kw["description"] == "A test image"
    client.status_post.assert_called_once_with(status="", media_ids=["123"])
    assert result == {"id": "456"}


def test_mime_for_path_maps_suffixes() -> None:
    """_mime_for_path returns correct MIME for known extensions."""
    assert mastodon_module._mime_for_path(Path("a.jpg")) == "image/jpeg"
    assert mastodon_module._mime_for_path(Path("b.jpeg")) == "image/jpeg"
    assert mastodon_module._mime_for_path(Path("c.png")) == "image/png"
    assert mastodon_module._mime_for_path(Path("d.webp")) == "image/webp"
    assert mastodon_module._mime_for_path(Path("e.gif")) == "image/gif"
    assert mastodon_module._mime_for_path(Path("f.unknown")) == "image/jpeg"
