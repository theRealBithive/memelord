"""
Mastodon-compatible API helpers shared by the Mastodon and Pixelfed scrapers.

Both Mastodon and Pixelfed implement the same /api/v1/accounts/* endpoints, so
a single pagination loop serves both. Callers pass a ``prefix`` string
(``"mastodon"`` or ``"pixelfed"``) to stamp filenames and log messages.

Not intended for direct import outside of retina.mastodon / retina.pixelfed.
"""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

from retina import image_validation

_RATE_LIMIT_SEC = 1.0
_IMAGE_TYPES = ("image",)
_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_PAGE_SIZE = 40


def _parse_account(handle: str) -> tuple[str, str]:
    """
    Parse a Mastodon/Pixelfed account handle into (username, instance).

    Accepts both ``@art@mastodon.social`` and ``art@mastodon.social``.
    Raises ValueError for malformed handles so callers can log and skip.
    """
    stripped = handle.lstrip("@")
    parts = stripped.split("@", 1)
    if len(parts) != 2 or not parts[0] or not parts[1] or "." not in parts[1]:
        raise ValueError(f"Invalid account handle: {handle!r}")
    return parts[0], parts[1]


def _get_json(url: str, *, access_token: str | None = None) -> list | dict:
    """Fetch URL and parse JSON. Raises on HTTP/connection errors or empty response."""
    headers: dict[str, str] = {"User-Agent": "Janulon/1.0", "Accept": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=15) as resp:
        body = resp.read().decode()
        if not body.strip():
            raise json.JSONDecodeError("Empty response", "", 0)
        return json.loads(body)


def _lookup_account_id(
    instance: str,
    username: str,
    *,
    access_token: str | None = None,
) -> str | None:
    """
    Resolve a username to an account ID via the /api/v1/accounts/lookup endpoint.

    Returns None on 404 (account not found or renamed) without clearing any stored
    cursor — preserving the cursor avoids a full re-download if the account comes back.
    """
    url = f"https://{instance}/api/v1/accounts/lookup?acct={username}"
    try:
        data = _get_json(url, access_token=access_token)
    except HTTPError as e:
        if e.code == 404:
            logger.warning("Account @{}@{} not found (404)", username, instance)
            return None
        logger.warning("HTTP {} looking up @{}@{}: {}", e.code, username, instance, e)
        return None
    except (URLError, json.JSONDecodeError) as e:
        logger.warning("Failed to look up @{}@{}: {}", username, instance, e)
        return None
    if not isinstance(data, dict):
        return None
    return data.get("id")


def _fetch_statuses_page(
    instance: str,
    account_id: str,
    *,
    max_id: str | None = None,
    since_id: str | None = None,
    min_id: str | None = None,
    limit: int = _PAGE_SIZE,
    access_token: str | None = None,
) -> list[dict]:
    """
    Fetch one page of media-only statuses for an account.

    Returns [] on 404, 429 (rate-limited), or any other error so the pagination
    loop terminates gracefully rather than raising.
    """
    url = f"https://{instance}/api/v1/accounts/{account_id}/statuses?only_media=true&limit={limit}"
    if max_id:
        url += f"&max_id={max_id}"
    elif since_id:
        url += f"&since_id={since_id}"
    elif min_id:
        url += f"&min_id={min_id}"
    try:
        data = _get_json(url, access_token=access_token)
    except HTTPError as e:
        if e.code == 429:
            logger.warning("Rate-limited by {}; stopping pagination", instance)
        elif e.code == 404:
            logger.warning("Account {} not found on {} (404)", account_id, instance)
        else:
            logger.warning("HTTP {} fetching statuses from {}: {}", e.code, instance, e)
        return []
    except (URLError, json.JSONDecodeError) as e:
        logger.warning("Failed to fetch statuses from {}: {}", instance, e)
        return []
    if not isinstance(data, list):
        return []
    return data


def _extract_items(statuses: list[dict], seen_images: set[str]) -> list[tuple[str, str]]:
    """
    Pull (image_url, post_url) from a page of statuses.

    Skips boosts (status["reblog"] is set) and non-image media. Takes the first
    image attachment per status to match the one-per-post convention.
    """
    out: list[tuple[str, str]] = []
    for status in statuses:
        if not isinstance(status, dict):
            continue
        if status.get("reblog") is not None:
            continue
        post_url = status.get("url") or status.get("uri")
        if not post_url or not isinstance(post_url, str):
            continue
        for att in status.get("media_attachments") or []:
            if not isinstance(att, dict):
                continue
            if att.get("type") not in _IMAGE_TYPES:
                continue
            image_url = att.get("url")
            if not image_url or not isinstance(image_url, str):
                continue
            if image_url in seen_images:
                continue
            seen_images.add(image_url)
            out.append((image_url, post_url))
            break
    return out


def iter_image_items(
    account_handle: str,
    *,
    since_id: str | None = None,
    max_pages: int = 10,
    access_token: str | None = None,
) -> tuple[list[tuple[str, str]], str | None]:
    """
    Collect (image_url, post_url) from an account's media timeline.

    Returns (items, new_cursor) where new_cursor is the highest status ID seen
    as a string. Uses integer comparison for Snowflake IDs of varying length.

    Two pagination modes:
    - First run (since_id=None): paginate backward with max_id to get full history.
    - Resume (since_id set): page forward from the cursor using since_id / min_id.

    Saves whatever was fetched before a rate-limit or error stops pagination so
    a partial run still advances the cursor.
    """
    try:
        username, instance = _parse_account(account_handle)
    except ValueError as e:
        logger.warning("{}", e)
        return [], None

    account_id = _lookup_account_id(instance, username, access_token=access_token)
    if account_id is None:
        return [], None

    items: list[tuple[str, str]] = []
    seen_images: set[str] = set()
    seen_ids: list[str] = []

    if since_id is None:
        cursor_max_id: str | None = None
        for page_num in range(max_pages):
            time.sleep(_RATE_LIMIT_SEC)
            page = _fetch_statuses_page(
                instance,
                account_id,
                max_id=cursor_max_id,
                access_token=access_token,
            )
            if not page:
                break
            page_ids = [s["id"] for s in page if isinstance(s, dict) and s.get("id")]
            seen_ids.extend(page_ids)
            items.extend(_extract_items(page, seen_images))
            if len(page) < _PAGE_SIZE:
                break
            cursor_max_id = min(page_ids, key=int)
            logger.debug(
                "{}: page {}/{}, {} items so far",
                account_handle,
                page_num + 1,
                max_pages,
                len(items),
            )
    else:
        # Resume: page forward chronologically with min_id, starting from the
        # stored cursor. since_id alone returns only the single newest page, so
        # if more than one page of posts accumulated since the last run the
        # middle would be skipped — and the cursor would jump past it for good.
        # min_id walks forward from the cursor and picks up every new post.
        cursor_min_id: str = since_id
        for page_num in range(max_pages):
            time.sleep(_RATE_LIMIT_SEC)
            page = _fetch_statuses_page(
                instance,
                account_id,
                min_id=cursor_min_id,
                access_token=access_token,
            )
            if not page:
                break
            page_ids = [s["id"] for s in page if isinstance(s, dict) and s.get("id")]
            seen_ids.extend(page_ids)
            items.extend(_extract_items(page, seen_images))
            if len(page) < _PAGE_SIZE:
                break
            cursor_min_id = max(page_ids, key=int)
            logger.debug(
                "{}: resume page {}/{}, {} items so far",
                account_handle,
                page_num + 1,
                max_pages,
                len(items),
            )

    new_cursor = max(seen_ids, key=int) if seen_ids else None
    logger.info(
        "{}: {} image items, new cursor: {}",
        account_handle,
        len(items),
        new_cursor,
    )
    return items, new_cursor


def _filename_for_item(prefix: str, account_handle: str, image_url: str) -> str:
    """
    Deterministic filename from prefix + account handle + image URL hash.

    Using a hash of the URL means the filename is stable across re-runs — if
    the file already exists, the download is skipped without a database lookup.
    """
    slug = _SAFE_SLUG_RE.sub("_", account_handle.lstrip("@"))[:40]
    digest = hashlib.sha256(image_url.encode()).hexdigest()[:12]
    ext = Path(image_url.split("?")[0]).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        ext = ".jpg"
    return f"{prefix}_{slug}_{digest}{ext}"


def download_images(
    items: list[tuple[str, str]],
    output_dir: Path,
    account_handle: str,
    *,
    prefix: str,
    rate_limit_sec: float = 0.5,
    skip_dirs: list[Path] | None = None,
    skip_paths: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Download each image to output_dir. source_label in the result is account_handle.

    Args:
        items: List of (image_url, post_url).
        output_dir: Directory to write files into.
        account_handle: Used as source_label in DB records and in filenames.
        prefix: Filename prefix — ``"mastodon"`` or ``"pixelfed"``.
        rate_limit_sec: Delay between downloads.
        skip_dirs: Directories already containing rated images.
        skip_paths: Resolved file paths already recorded in the DB; skip if present.

    Returns:
        List of (path, source_url, source_label); source_url is the post URL.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = [Path(d) for d in (skip_dirs or [])]
    total = len(items)
    logger.info(
        "Downloading {} {} images for {} to {}",
        total,
        prefix,
        account_handle,
        output_dir.resolve(),
    )
    written: list[tuple[Path, str, str]] = []
    skipped_output = skipped_sets = skipped_db = 0

    for image_url, post_url in items:
        filename = _filename_for_item(prefix, account_handle, image_url)
        path = output_dir / filename

        if path.exists():
            skipped_output += 1
            logger.debug("Skipped (already in output): {}", filename)
            continue
        if any((d / filename).exists() for d in skip_dirs):
            skipped_sets += 1
            logger.debug("Skipped (already rated): {}", filename)
            continue
        if skip_paths and str(path.resolve()) in skip_paths:
            skipped_db += 1
            logger.debug("Skipped (path in database): {}", filename)
            continue

        try:
            time.sleep(rate_limit_sec)
            req = Request(image_url, headers={"User-Agent": "Janulon/1.0"})
            with urlopen(req, timeout=30) as resp:
                path.write_bytes(resp.read())
            if not image_validation.is_readable_image(path):
                path.unlink(missing_ok=True)
                logger.warning("Removed corrupted download: {}", filename)
                continue
            written.append((path, post_url, account_handle))
            logger.info("Downloaded {}/{}: {}", len(written), total, filename)
        except (HTTPError, URLError, OSError) as e:
            logger.warning("Failed to download {}: {}", filename, e)

    if skipped_output or skipped_sets or skipped_db:
        logger.info(
            "Skipped {} (output), {} (rated), {} (DB)",
            skipped_output,
            skipped_sets,
            skipped_db,
        )
    logger.info(
        "Downloaded {} of {} {} images for {}",
        len(written),
        total,
        prefix,
        account_handle,
    )
    return written
