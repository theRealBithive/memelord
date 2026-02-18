"""Fetch images from Pixelfed public timeline (Mastodon-compatible API)."""

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
# Basename from URL may have query params; strip to get a safe filename stem
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
# Playwright fallback: wait for feed and scroll pause
_BROWSER_WAIT_MS = 4000
_BROWSER_SCROLL_PAUSE_MS = 1500
_BROWSER_MAX_SCROLLS = 25


def _normalize_base(base: str) -> str:
    """Return instance base URL without trailing slash."""
    return base.rstrip("/")


def _get_json(url: str, headers: dict | None = None) -> list | dict:
    """Fetch URL and parse JSON. Raises on HTTP or connection errors."""
    h = {"User-Agent": "Janulon/1.0", "Accept": "application/json"}
    if headers:
        h.update(headers)
    req = Request(url, headers=h)
    with urlopen(req, timeout=15) as resp:
        body = resp.read().decode()
        if not body.strip():
            raise json.JSONDecodeError("Empty response", "", 0)
        return json.loads(body)


def fetch_public_timeline(
    instance_base: str,
    limit: int = 40,
    *,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
    access_token: str | None = None,
) -> list[dict]:
    """
    Fetch public timeline from a Pixelfed instance (Mastodon-compatible API).

    Some instances return HTML (login page) unless Accept: application/json is set
    or an access_token is provided. Pass access_token if the instance requires auth.

    Args:
        instance_base: Instance URL (e.g. https://pixelfed.social).
        limit: Max number of statuses to request (API may cap at 40).
        rate_limit_sec: Delay before the request.
        access_token: Optional OAuth token (Bearer) for instances that require auth.

    Returns:
        List of status dicts (empty on error or non-list response).
    """
    base = _normalize_base(instance_base)
    url = f"{base}/api/v1/timelines/public?limit={limit}"
    time.sleep(rate_limit_sec)
    headers: dict | None = None
    if access_token and access_token.strip():
        headers = {"Authorization": f"Bearer {access_token.strip()}"}
    try:
        data = _get_json(url, headers=headers)
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        logger.warning("Failed to fetch Pixelfed timeline: {}", e)
        return []
    if not isinstance(data, list):
        logger.warning("Pixelfed timeline did not return a list")
        return []
    return data


def _fetch_timeline_with_browser(
    instance_base: str,
    limit: int = 40,
) -> list[dict]:
    """
    Use Playwright to load the explore page and capture timeline JSON or scrape DOM.

    Tries to intercept the API response (timelines/public or discover). If that
    fails, scrapes post links and image URLs from the page and builds minimal
    status-like dicts so the rest of the pipeline can stay the same.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.debug("Playwright not available for Pixelfed fallback")
        return []
    base = _normalize_base(instance_base)
    explore_url = f"{base}/web/explore"
    captured: list[list[dict]] = []

    def on_response(response: object) -> None:
        try:
            url = getattr(response, "url", "") or ""
            req = getattr(response, "request", None)
            if req and getattr(req, "method", "") != "GET":
                return
            if "api" not in url or ("timelines" not in url and "discover" not in url):
                return
            body = getattr(response, "json", lambda: None)()
            if body is None:
                return
            if isinstance(body, list) and body and isinstance(body[0], dict):
                captured.append(body[:limit])
        except Exception:
            pass

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("response", on_response)
            page.goto(explore_url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(_BROWSER_WAIT_MS)
            if captured:
                browser.close()
                return captured[0]
            # DOM scrape: find post links and their first image
            for _ in range(_BROWSER_MAX_SCROLLS):
                items = page.evaluate(_js_scrape_explore_posts(base))
                if isinstance(items, list) and len(items) >= limit:
                    browser.close()
                    return items[:limit]
                page.evaluate(
                    "() => { window.scrollTo(0, document.documentElement.scrollHeight); }"
                )
                page.wait_for_timeout(_BROWSER_SCROLL_PAUSE_MS)
            items = page.evaluate(_js_scrape_explore_posts(base))
            browser.close()
            if isinstance(items, list):
                return items[:limit]
            return []
    except Exception as e:
        logger.debug("Pixelfed browser fetch failed: {}", e)
        return []


def _js_scrape_explore_posts(instance_base: str) -> str:
    """Return JS that scrapes the explore page for (post_url, image_url) and returns status-like objects."""
    base = instance_base.rstrip("/")
    return f"""
    () => {{
        const statuses = [];
        const seen = new Set();
        // Links to post pages: /p/username/id or similar
        const links = document.querySelectorAll('a[href*="/p/"]');
        for (const a of links) {{
            const href = a.getAttribute('href') || '';
            const path = href.startsWith('http') ? new URL(href).pathname : href;
            if (!path.includes('/p/') || path.length < 5) continue;
            const postUrl = href.startsWith('http') ? href : ('{base}' + (path.startsWith('/') ? path : '/' + path));
            if (seen.has(postUrl)) continue;
            const img = a.querySelector('img[src]');
            if (!img || !img.src) continue;
            let imgUrl = img.src.split('?')[0];
            if (!imgUrl.startsWith('http')) imgUrl = '{base}' + (imgUrl.startsWith('/') ? imgUrl : '/' + imgUrl);
            seen.add(postUrl);
            statuses.push({{ url: postUrl, media_attachments: [{{ type: 'image', url: imgUrl }}] }});
        }}
        return statuses;
    }}
    """


def iter_image_items(
    instance_base: str,
    limit: int = 40,
    *,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
    access_token: str | None = None,
) -> list[tuple[str, str]]:
    """
    Collect (image_url, post_url) from the instance's public timeline.

    For each status with at least one image attachment, yields the first image
    URL and the status's post URL (for later boost). Skips statuses without
    image media or without a post URL.

    Returns:
        List of (image_url, post_url). post_url is the status url or uri.
    """
    statuses = fetch_public_timeline(
        instance_base,
        limit=limit,
        rate_limit_sec=rate_limit_sec,
        access_token=access_token,
    )
    if not statuses:
        logger.info("Pixelfed API returned no timeline; trying Playwright fallback.")
        statuses = _fetch_timeline_with_browser(instance_base, limit=limit)
        if statuses:
            logger.info(
                "Pixelfed browser fallback: {} statuses",
                len(statuses),
            )
    out: list[tuple[str, str]] = []
    seen_images: set[str] = set()
    for status in statuses:
        if not isinstance(status, dict):
            continue
        post_url = status.get("url") or status.get("uri")
        if not post_url or not isinstance(post_url, str):
            continue
        media = status.get("media_attachments") or []
        for att in media:
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
            break  # one image per status
    logger.info(
        "Pixelfed timeline: {} statuses, {} image items",
        len(statuses),
        len(out),
    )
    return out


def _filename_for_item(index: int, image_url: str) -> str:
    """Generate a safe filename for a downloaded image (pixelfed_0_abc.jpg)."""
    base = Path(image_url).name.split("?")[0].strip()
    if not base or base == ".":
        base = f"img_{index}"
    else:
        base = _SAFE_FILENAME_RE.sub("_", base)[:80] or f"img_{index}"
    ext = Path(base).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        base = f"{base}.jpg"
    return f"pixelfed_{index}_{base}"


def download_images(
    items: list[tuple[str, str]],
    output_dir: Path,
    *,
    rate_limit_sec: float = 0.5,
    skip_dirs: list[Path] | None = None,
    skip_paths: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Download each image to output_dir. source_url in the result is the post URL.

    Args:
        items: List of (image_url, post_url).
        output_dir: Directory to write files into.
        rate_limit_sec: Delay between downloads.
        skip_dirs: Directories to consider "already present" (e.g. corpus/void).
        skip_paths: Resolved paths already in DB; skip if path in this set.

    Returns:
        List of (path, source_url, source_label); source_label is "pixelfed".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = [Path(d) for d in (skip_dirs or [])]
    total = len(items)
    logger.info("Downloading {} Pixelfed images to {}", total, output_dir.resolve())
    written: list[tuple[Path, str, str]] = []
    skipped_output = 0
    skipped_sets = 0
    skipped_db = 0
    for i, (image_url, post_url) in enumerate(items):
        filename = _filename_for_item(i, image_url)
        path = output_dir / filename
        if path.exists():
            skipped_output += 1
            logger.debug("Skipped (already in output): {}", filename)
            continue
        in_sets = False
        for d in skip_dirs:
            if (d / filename).exists():
                in_sets = True
                break
        if in_sets:
            skipped_sets += 1
            logger.debug("Skipped (already in corpus/void): {}", filename)
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
            written.append((path, post_url, "pixelfed"))
            logger.info("Downloaded {}/{}: {}", len(written), total, filename)
        except (HTTPError, URLError, OSError) as e:
            logger.warning("Failed to download {}: {}", filename, e)
    if skipped_output or skipped_sets or skipped_db:
        logger.info(
            "Skipped {} (already in output), {} (corpus/void), {} (path in DB)",
            skipped_output,
            skipped_sets,
            skipped_db,
        )
    logger.info(
        "Downloaded {} of {} Pixelfed images to {}",
        len(written),
        total,
        output_dir.resolve(),
    )
    return written
