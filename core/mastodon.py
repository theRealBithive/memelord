"""Post images to Mastodon with alt text."""

import time
from pathlib import Path

from mastodon import Mastodon
from mastodon.errors import MastodonAPIError

# MIME types for supported image extensions (matches brain.IMAGE_EXTENSIONS)
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


def _mime_for_path(path: Path) -> str:
    """Return MIME type for path; default to image/jpeg if unknown."""
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")


def _wait_for_media_ready(
    client: Mastodon,
    media_id: str,
    *,
    poll_interval: float = 1.0,
    timeout: float = 30.0,
) -> None:
    """
    Poll until the media attachment has finished processing (url is set).

    Mastodon processes uploads asynchronously; status_post fails with 422
    if media is not ready. Raises MastodonAPIError if timeout is exceeded.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        media = client.media(media_id)
        url = (
            media.get("url") if isinstance(media, dict) else getattr(media, "url", None)
        )
        if url:
            return
        time.sleep(poll_interval)
    raise MastodonAPIError(
        "Media attachment did not finish processing before timeout. Try again in a moment."
    )


def create_client(base_url: str, access_token: str) -> Mastodon:
    """
    Create an authenticated Mastodon client.

    Args:
        base_url: Instance URL (e.g. https://mastodon.example).
        access_token: User access token.

    Returns:
        Mastodon instance ready for posting.
    """
    return Mastodon(access_token=access_token.strip(), api_base_url=base_url.strip())


def post_status(client: Mastodon, status_text: str) -> dict:
    """
    Post a text-only status (no media).

    Args:
        client: Authenticated Mastodon client from create_client().
        status_text: The status text to post.

    Returns:
        The Status dict returned by the API.

    Raises:
        MastodonAPIError: On post failure.
    """
    return client.status_post(status=status_text)


def post_image(
    client: Mastodon,
    image_path: Path | str,
    alt_text: str,
    status_text: str = "",
) -> dict:
    """
    Upload an image with alt text and post it as a new status.

    Args:
        client: Authenticated Mastodon client from create_client().
        image_path: Path to the image file.
        alt_text: Description for accessibility (alt text).
        status_text: Optional status text; default empty (image-only post).

    Returns:
        The Status dict returned by the API.

    Raises:
        MastodonAPIError: On upload or post failure.
    """
    path = Path(image_path)
    mime = _mime_for_path(path)
    media = client.media_post(
        str(path.resolve()),
        mime_type=mime,
        description=alt_text,
    )
    media_id = media["id"] if isinstance(media, dict) else media.id
    _wait_for_media_ready(client, media_id)
    return client.status_post(status=status_text or "", media_ids=[media_id])


def engagement_from_status(status: dict | object) -> dict[str, int]:
    """
    Extract engagement counts from a Status dict or entity (e.g. from status_post).

    Returns:
        Dict with keys favourites_count, reblogs_count, replies_count (all int).
    """
    if isinstance(status, dict):
        return {
            "favourites_count": int(status.get("favourites_count", 0)),
            "reblogs_count": int(status.get("reblogs_count", 0)),
            "replies_count": int(status.get("replies_count", 0)),
        }
    return {
        "favourites_count": getattr(status, "favourites_count", 0) or 0,
        "reblogs_count": getattr(status, "reblogs_count", 0) or 0,
        "replies_count": getattr(status, "replies_count", 0) or 0,
    }


def fetch_status_engagement(client: Mastodon, status_id: str | int) -> dict[str, int]:
    """
    Fetch a status by ID and return engagement counts.

    Args:
        client: Authenticated Mastodon client.
        status_id: The status ID (string or int).

    Returns:
        Dict with keys favourites_count, reblogs_count, replies_count (all int).

    Raises:
        MastodonAPIError: If the status cannot be fetched.
    """
    sid = int(status_id) if isinstance(status_id, str) else status_id
    status = client.status(sid)
    return engagement_from_status(status)
