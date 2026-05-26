"""Scraper for Imgur topic galleries (e.g. imgur.com/t/funny). Uses scraping by default; optional API if Client ID set."""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

from retina import image_validation

_API_BASE = "https://api.imgur.com/3"
_WEB_BASE = "https://imgur.com"
_RATE_LIMIT_SEC = 1.0
_SCRAPE_RATE_LIMIT_SEC = 2.0
_ITEMS_PER_PAGE = 60
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
# Direct image URL pattern: https://i.imgur.com/ID.ext or https://i.imgur.com/ID.ext?params
_DIRECT_IMAGE_RE = re.compile(
    r"https?://i\.imgur\.com/[a-zA-Z0-9]+\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\"'\s]*)?"
)
# URL path substrings that indicate non-gallery assets (banner, logo, etc.)
_NON_GALLERY_URL_PATTERNS = ("logo", "banner", "favicon", "og-image", "avatar", "icon")
# Imgur size suffix: strip _X (e.g. _d) or single letter s,b,t,m,l,h before extension for original
_IMGUR_UNDERSCORE_SUFFIX_RE = re.compile(
    r"(https?://i\.imgur\.com/[a-zA-Z0-9]+)_[a-zA-Z0-9](\.(?:jpg|jpeg|png|gif|webp))",
    re.IGNORECASE,
)
_IMGUR_LETTER_SUFFIX_RE = re.compile(
    r"(https?://i\.imgur\.com/)([a-zA-Z0-9]*?)([sbtmlh])(\.(?:jpg|jpeg|png|gif|webp))",
    re.IGNORECASE,
)
# Browser-like User-Agent to reduce chance of redirect/block when scraping
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _get_json(url: str, client_id: str) -> dict:
    """Fetch URL with Imgur Client-ID header. Returns parsed JSON."""
    req = Request(
        url,
        headers={
            "User-Agent": "Janulon/1.0",
            "Authorization": f"Client-ID {client_id}",
        },
    )
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _fetch_html(url: str) -> str:
    """Fetch URL and return body as string. Uses browser-like User-Agent."""
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=15) as resp:
        return resp.read().decode(errors="replace")


def _is_gallery_url(url: str) -> bool:
    """Return False if URL looks like a non-gallery asset (logo, banner, etc.)."""
    url_lower = url.lower()
    return not any(pat in url_lower for pat in _NON_GALLERY_URL_PATTERNS)


def _imgur_to_original_url(url: str) -> str:
    """
    Convert an Imgur thumbnail/size-variant URL to the original full-size URL.
    Strips _X (e.g. _d) or single letter (s,b,t,m,l,h) before the extension.
    """
    base = url.split("?")[0]
    base = _IMGUR_UNDERSCORE_SUFFIX_RE.sub(r"\1\2", base)
    base = _IMGUR_LETTER_SUFFIX_RE.sub(r"\1\2\4", base)
    return base


def _extract_image_urls_from_html(html: str) -> list[str]:
    """
    Extract direct Imgur image URLs from HTML (and embedded JSON).
    Returns deduplicated list of https://i.imgur.com/... URLs, excluding non-gallery.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _DIRECT_IMAGE_RE.finditer(html):
        url = m.group(0).split("?")[0]
        if not _is_gallery_url(url):
            continue
        orig = _imgur_to_original_url(url)
        if orig not in seen:
            seen.add(orig)
            out.append(orig)
    for pattern in (
        r'"link"\s*:\s*"(https://i\.imgur\.com/[^"]+\.(?:jpg|jpeg|png|gif|webp))"',
        r'"url"\s*:\s*"(https://i\.imgur\.com/[^"]+\.(?:jpg|jpeg|png|gif|webp))"',
    ):
        for m in re.finditer(pattern, html):
            url = m.group(1).split("?")[0]
            if not _is_gallery_url(url):
                continue
            orig = _imgur_to_original_url(url)
            if orig not in seen:
                seen.add(orig)
                out.append(orig)
    return out


def _topic_slug(topic: str) -> str:
    """Normalize topic to slug (e.g. 't/funny' or 'funny' -> 'funny')."""
    return topic.lower().strip().removeprefix("t/").split("/")[0]


def _js_get_gallery_image_urls() -> str:
    """JavaScript: return gallery image URLs (main feed only, skip single banner/hero)."""
    return """
    () => {
        const sel = 'img[src*="i.imgur.com"]';
        const norm = (s) => (s || '').split('?')[0];
        const main = document.querySelector('main');
        let imgs = [];
        if (main) {
            imgs = Array.from(main.querySelectorAll(sel));
        }
        if (imgs.length <= 1) {
            imgs = Array.from(document.querySelectorAll(sel));
            if (imgs.length > 2) imgs = imgs.slice(1);
        }
        const out = [];
        const seen = new Set();
        imgs.forEach(img => {
            const src = norm(img.src);
            if (src && !seen.has(src)) { seen.add(src); out.push(src); }
        });
        return out;
    }
    """


def _scroll_to_bottom(page: object) -> None:
    """Scroll window and main to bottom to trigger infinite scroll."""
    page.evaluate(
        "() => { window.scrollTo(0, document.documentElement.scrollHeight); "
        "const m = document.querySelector('main'); "
        "if (m) m.scrollTop = m.scrollHeight; }"
    )


def _fetch_image_urls_with_browser(
    url: str,
    wait_sec: float = 6.0,
    target_count: int = 50,
    scroll_pause_ms: int = 2200,
    max_scroll_rounds: int = 60,
) -> list[str]:
    """
    Use Playwright to load the page, scroll to bottom repeatedly to trigger
    lazy-load, and collect gallery image URLs until target_count or feed stops.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            try:
                page.wait_for_selector(
                    "img[src*='i.imgur.com']",
                    timeout=15000,
                )
            except Exception:
                pass
            page.wait_for_timeout(int(wait_sec * 1000))
            seen: set[str] = set()
            stale_rounds = 0
            for _ in range(max_scroll_rounds):
                urls = page.evaluate(_js_get_gallery_image_urls())
                if isinstance(urls, list):
                    added = 0
                    for u in urls:
                        if not isinstance(u, str):
                            continue
                        u = u.split("?")[0]
                        if not _is_gallery_url(u):
                            continue
                        orig = _imgur_to_original_url(u)
                        if orig not in seen:
                            seen.add(orig)
                            added += 1
                    if added == 0:
                        stale_rounds += 1
                        if stale_rounds >= 10:
                            break
                    else:
                        stale_rounds = 0
                if len(seen) >= target_count:
                    break
                _scroll_to_bottom(page)
                page.wait_for_timeout(scroll_pause_ms)
            browser.close()
            return sorted(seen)
    except Exception as e:
        logger.debug("Browser fetch failed: {}", e)
        return []


def get_topic_page_scrape(
    topic: str,
    page: int = 0,
    *,
    use_browser: bool = False,
    target_urls: int | None = None,
) -> list[str]:
    """
    Fetch one topic page HTML and extract direct image URLs (no API).
    Page 0 = /t/slug, page 1+ = /t/slug/page/N (if supported).
    If use_browser is True and Playwright is installed, scroll until target_urls
    collected (default 50) or feed stops loading.
    """
    slug = _topic_slug(topic)
    if page <= 0:
        url = f"{_WEB_BASE}/t/{slug}"
    else:
        url = f"{_WEB_BASE}/t/{slug}/page/{page}"
    if use_browser:
        target = target_urls if target_urls is not None else 50
        urls = _fetch_image_urls_with_browser(url, target_count=target)
        if urls:
            return urls
    html = _fetch_html(url)
    return _extract_image_urls_from_html(html)


def get_topic_page(
    topic: str,
    page: int,
    client_id: str,
) -> dict:
    """
    Fetch one page of gallery items for a topic (API; requires Client ID).
    Returns API response with data.items (and data.total_items if present).
    """
    slug = _topic_slug(topic)
    url = f"{_API_BASE}/gallery/t/{slug}?page={page}"
    return _get_json(url, client_id)


def image_urls_from_item(item: dict) -> list[str]:
    """
    Return all image URLs for a gallery item (single image or album).
    Single image: item has 'link'. Album: item has 'images' with link each.
    """
    if not isinstance(item, dict):
        return []
    # Album: list of images
    images = item.get("images")
    if isinstance(images, list) and images:
        urls = []
        for img in images:
            if isinstance(img, dict) and img.get("link"):
                urls.append(img["link"])
        if urls:
            return urls
    # Single image
    link = item.get("link")
    if link and isinstance(link, str):
        return [link]
    return []


def iter_image_urls(
    topic: str,
    client_id: str = "",
    max_items: int = 120,
    *,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
) -> list[str]:
    """
    Collect image URLs from the topic gallery.
    If client_id is set, uses Imgur API (paginated). Otherwise scrapes topic pages.
    Deduplicates by URL.
    """
    if (client_id or "").strip():
        return _iter_image_urls_api(topic, client_id.strip(), max_items, rate_limit_sec)
    return _iter_image_urls_scrape(topic, max_items)


def _iter_image_urls_api(
    topic: str,
    client_id: str,
    max_items: int,
    rate_limit_sec: float,
) -> list[str]:
    """API path: paginated gallery/t/topic requests."""
    seen: set[str] = set()
    out: list[str] = []
    page = 0
    while len(out) < max_items:
        logger.info(
            "Fetching page {} for topic {} ({} so far)",
            page + 1,
            topic,
            len(out),
        )
        time.sleep(rate_limit_sec)
        try:
            data = get_topic_page(topic, page, client_id)
        except (HTTPError, URLError, json.JSONDecodeError) as e:
            logger.warning("Failed to fetch page {}: {}", page + 1, e)
            break
        if not isinstance(data, dict) or not data.get("success"):
            logger.warning("API returned success=false or invalid shape")
            break
        payload = data.get("data")
        if not isinstance(payload, dict):
            break
        items = payload.get("items")
        if not isinstance(items, list):
            break
        n_page = 0
        for item in items:
            for url in image_urls_from_item(item):
                if url not in seen:
                    seen.add(url)
                    out.append(url)
                    n_page += 1
                    if len(out) >= max_items:
                        break
            if len(out) >= max_items:
                break
        logger.info(
            "Page {}: {} items, {} new image URLs ({} total)",
            page + 1,
            len(items),
            n_page,
            len(out),
        )
        if len(items) < _ITEMS_PER_PAGE:
            break
        page += 1
    logger.info(
        "Scrape complete: {} unique image URLs from topic {}",
        len(out),
        topic,
    )
    return out


def _playwright_available() -> bool:
    """Return True if Playwright is installed."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401

        return True
    except ImportError:
        return False


def _iter_image_urls_scrape(topic: str, max_items: int) -> list[str]:
    """Scrape path: fetch topic page(s) HTML and extract direct image URLs."""
    seen: set[str] = set()
    out: list[str] = []
    page = 0
    max_pages = max(3, (max_items // 50) + 1)
    used_browser = False
    while len(out) < max_items and page < max_pages:
        logger.info(
            "Scraping page {} for topic {} ({} URLs so far)",
            page + 1,
            topic,
            len(out),
        )
        time.sleep(_SCRAPE_RATE_LIMIT_SEC)
        try:
            use_browser = page == 0 and len(out) == 0
            urls = get_topic_page_scrape(
                topic,
                page,
                use_browser=use_browser,
                target_urls=max_items if use_browser else None,
            )
            if use_browser and urls:
                used_browser = True
        except (HTTPError, URLError, OSError) as e:
            logger.warning("Failed to scrape page {}: {}", page + 1, e)
            break
        n_page = 0
        for url in urls:
            if url not in seen:
                seen.add(url)
                out.append(url)
                n_page += 1
                if len(out) >= max_items:
                    break
        logger.info(
            "Page {}: {} new image URLs ({} total)",
            page + 1,
            n_page,
            len(out),
        )
        if not urls:
            if page == 0 and not used_browser and not _playwright_available():
                logger.warning(
                    "Imgur topic pages load content via JavaScript. "
                    "For scraping, install: uv pip install 'janulon[imgur-browser]' "
                    "then run: playwright install chromium"
                )
            break
        if used_browser and page == 0:
            break
        page += 1
    logger.info(
        "Scrape complete: {} unique image URLs from topic {}",
        len(out),
        topic,
    )
    return out


def _filename_for_url(topic: str, url: str) -> str:
    """Stable unique filename from topic and URL."""
    path = url.split("?")[0].rstrip("/")
    base = Path(path).name
    ext = Path(base).suffix.lower() if base else ".jpg"
    if ext not in _IMAGE_EXTENSIONS:
        ext = ".jpg"
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return f"{topic}_{digest}{ext}"


def download_images(
    urls: list[str],
    output_dir: Path,
    topic: str,
    *,
    rate_limit_sec: float = 0.5,
    skip_dirs: list[Path] | None = None,
    skip_paths: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Download each URL into output_dir as {topic}_{hash}{ext}.
    Skips if file exists in output_dir, in skip_dirs, or path is in skip_paths (DB).
    Returns list of (path, source_url, source_label) for each file written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = [Path(d) for d in (skip_dirs or [])]
    total = len(urls)
    logger.info("Downloading {} images to {}", total, output_dir.resolve())
    written: list[tuple[Path, str, str]] = []
    skipped_output = 0
    skipped_sets = 0
    skipped_db = 0
    for url in urls:
        filename = _filename_for_url(topic, url)
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
            logger.debug("Skipped (already rated): {}", filename)
            continue
        if skip_paths and str(path.resolve()) in skip_paths:
            skipped_db += 1
            logger.debug("Skipped (path in database): {}", filename)
            continue
        try:
            time.sleep(rate_limit_sec)
            req = Request(url, headers={"User-Agent": "Janulon/1.0"})
            with urlopen(req, timeout=30) as resp:
                path.write_bytes(resp.read())
            if not image_validation.is_readable_image(path):
                path.unlink(missing_ok=True)
                logger.warning("Removed corrupted download: {}", filename)
                continue
            written.append((path, url, topic))
            logger.info("Downloaded {}/{}: {}", len(written), total, filename)
        except (HTTPError, URLError, OSError) as e:
            logger.warning("Failed to download {}: {}", filename, e)
    if skipped_output or skipped_sets or skipped_db:
        logger.info(
            "Skipped {} (already in output), {} (already rated), {} (path in DB)",
            skipped_output,
            skipped_sets,
            skipped_db,
        )
    logger.info(
        "Downloaded {} of {} images to {}",
        len(written),
        total,
        output_dir.resolve(),
    )
    return written
