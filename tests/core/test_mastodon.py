"""Tests for core.mastodon: media upload and status posting."""

from unittest.mock import Mock

import pytest

from core.mastodon import (
    _wait_for_media_ready,
    engagement_from_status,
    fetch_status_engagement,
    post_status,
)


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


def test_engagement_from_status_dict() -> None:
    """engagement_from_status extracts counts from a status dict."""
    status = {
        "favourites_count": 5,
        "reblogs_count": 3,
        "replies_count": 1,
    }
    assert engagement_from_status(status) == {
        "favourites_count": 5,
        "reblogs_count": 3,
        "replies_count": 1,
    }


def test_engagement_from_status_dict_missing_keys_default_zero() -> None:
    """engagement_from_status uses 0 for missing keys."""
    assert engagement_from_status({}) == {
        "favourites_count": 0,
        "reblogs_count": 0,
        "replies_count": 0,
    }


def test_fetch_status_engagement_calls_client_and_returns_counts() -> None:
    """fetch_status_engagement calls client.status and returns engagement from response."""
    client = Mock()
    client.status.return_value = {
        "favourites_count": 12,
        "reblogs_count": 4,
        "replies_count": 0,
    }
    result = fetch_status_engagement(client, "12345")
    client.status.assert_called_once_with(12345)
    assert result == {
        "favourites_count": 12,
        "reblogs_count": 4,
        "replies_count": 0,
    }


def test_post_status_posts_text_only() -> None:
    """post_status calls status_post with the given text and returns the API response."""
    client = Mock()
    client.status_post.return_value = {"id": "99", "content": "Pondering."}
    result = post_status(client, "Pondering the means of discernment anew.")
    client.status_post.assert_called_once_with(status="Pondering the means of discernment anew.")
    assert result == {"id": "99", "content": "Pondering."}
