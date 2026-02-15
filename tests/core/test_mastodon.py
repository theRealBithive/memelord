"""Tests for core.mastodon: media upload and status posting."""

from unittest.mock import Mock

import pytest

from core.mastodon import _wait_for_media_ready


def test_wait_for_media_ready_returns_immediately_when_url_set() -> None:
    """When media already has url, no polling is needed."""
    client = Mock()
    client.media.return_value = {"id": "123", "url": "https://example.com/media.png"}
    _wait_for_media_ready(client, "123", poll_interval=999.0, timeout=1.0)
    client.media.assert_called_once_with("123")


def test_wait_for_media_ready_polls_until_url_set() -> None:
    """Polls until media has url, then returns."""
    client = Mock()
    client.media.side_effect = [
        {"id": "123", "url": None},
        {"id": "123", "url": None},
        {"id": "123", "url": "https://example.com/media.png"},
    ]
    _wait_for_media_ready(client, "123", poll_interval=0.01, timeout=5.0)
    assert client.media.call_count == 3


def test_wait_for_media_ready_raises_on_timeout() -> None:
    """Raises MastodonAPIError when url never set within timeout."""
    from mastodon.errors import MastodonAPIError

    client = Mock()
    client.media.return_value = {"id": "123", "url": None}
    with pytest.raises(
        MastodonAPIError, match="did not finish processing before timeout"
    ):
        _wait_for_media_ready(client, "123", poll_interval=0.05, timeout=0.1)
