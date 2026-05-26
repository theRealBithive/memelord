"""Scraper for Tumblr blogs via the public v1 read API; no API key."""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

from retina import image_validation

_RATE_LIMIT_SEC = 1.5
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
_PAGE_SIZE = 20
# Match img src="..." or src='...' (URLs only; no HTML entities in URL)
_IMG_SRC_RE = re.compile(r'src=["\']([^"\']+)["\']', re.IGNORECASE)


def _get_json(url: str) -> dict:
    """Fetch URL and parse JSON. Handles trailing junk and JSONP wrapper."""
    req = Request(url, headers={"User-Agent": "Janulon/1.0"})
    with urlopen(req, timeout=15) as resp:
        raw = resp.read().decode()
    decoder = json.JSONDecoder()
    # Parse first JSON value only (Tumblr often appends ";" or extra data)
    try:
        obj, _ = decoder.raw_decode(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # JSONP wrapper: var tumblr_api_read = {...};
    pattern = r"^\s*var\s+tumblr_api_read\s*=\s*(.+);?\s*$"
    match = re.match(pattern, raw, re.DOTALL)
    if match:
        try:
            obj, _ = decoder.raw_decode(match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Could not parse Tumblr API response", raw, 0)


def get_posts(blog: str, num: int = _PAGE_SIZE, start: int = 0) -> dict:
    """Fetch one page of posts. Returns API response with 'posts' list."""
    base = blog if ".tumblr.com" in blog else f"{blog}.tumblr.com"
    url = f"https://{base}/api/read/json?num={num}&start={start}"
    return _get_json(url)


def _image_urls_from_html(html: str) -> list[str]:
    """
    Extract image URLs from HTML (e.g. regular-body) via img src.
    Returns only URLs that look like images (Tumblr media or known extensions).
    """
    if not html or not isinstance(html, str):
        return []
    urls: list[str] = []
    for match in _IMG_SRC_RE.finditer(html):
        url = match.group(1).strip()
        if not url or url in urls:
            continue
        lower = url.split("?")[0].lower()
        if "media.tumblr.com" in lower or "tumblr.com" in lower:
            urls.append(url)
            continue
        if any(lower.endswith(ext) for ext in _IMAGE_EXTENSIONS):
            urls.append(url)
    return urls


def image_urls_from_post(post: dict) -> list[str]:
    """
    Return all image URLs for a post.
    Handles type "photo" (photo-url-* and photos[]) and type "regular" (img in regular-body).
    Empty list for other types or if no images found.
    """
    if not isinstance(post, dict):
        return []
    post_type = post.get("type")

    if post_type == "regular":
        body = post.get("regular-body") or ""
        return _image_urls_from_html(body)

    if post_type != "photo":
        return []

    urls: list[str] = []
    # Largest first: photo-url-1280, 500, 400, 250
    keys = ("photo-url-1280", "photo-url-500", "photo-url-400", "photo-url-250")
    for key in keys:
        u = post.get(key)
        if u and isinstance(u, str) and u not in urls:
            urls.append(u)
    if urls:
        return urls

    # Alternative: photos array with original_size
    photos = post.get("photos") or []
    for p in photos:
        if not isinstance(p, dict):
            continue
        orig_raw = p.get("original_size")
        orig = orig_raw if isinstance(orig_raw, dict) else None
        if orig and orig.get("url"):
            u = orig["url"]
            if u not in urls:
                urls.append(u)
        for alt in p.get("alt_sizes") or []:
            if not isinstance(alt, dict) or not alt.get("url"):
                continue
            if alt["url"] not in urls:
                urls.append(alt["url"])
    return urls


def iter_image_urls(
    blog: str,
    num_posts: int = 50,
    *,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
) -> list[str]:
    """
    Collect image URLs from the blog (v1 API, no auth).
    Pages of _PAGE_SIZE until num_posts. Deduplicates by URL.
    """
    seen: set[str] = set()
    out: list[str] = []
    start = 0
    page = 0
    while start < num_posts:
        page += 1
        logger.info(
            "Fetching page {} for blog {} (start={})",
            page,
            blog,
            start,
        )
        time.sleep(rate_limit_sec)
        try:
            data = get_posts(blog, num=_PAGE_SIZE, start=start)
        except (HTTPError, URLError, json.JSONDecodeError) as e:
            logger.warning("Failed to fetch page {}: {}", page, e)
            break
        posts = data.get("posts") if isinstance(data, dict) else []
        if not posts:
            break
        n_page = 0
        for post in posts:
            for url in image_urls_from_post(post):
                if url not in seen:
                    seen.add(url)
                    out.append(url)
                    n_page += 1
        logger.info(
            "Page {}: {} posts, {} image URLs ({} unique so far)",
            page,
            len(posts),
            n_page,
            len(out),
        )
        start += len(posts)
        if len(posts) < _PAGE_SIZE:
            break
    logger.info(
        "Scrape complete: {} unique image URLs from blog {}",
        len(out),
        blog,
    )
    return out


def _filename_for_url(blog: str, url: str) -> str:
    """Stable unique filename from blog and URL (same URL => same file)."""
    path = url.split("?")[0].rstrip("/")
    base = Path(path).name
    ext = Path(base).suffix.lower() if base else ".jpg"
    if ext not in _IMAGE_EXTENSIONS:
        ext = ".jpg"
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return f"{blog}_{digest}{ext}"


def download_images(
    urls: list[str],
    output_dir: Path,
    blog: str,
    *,
    rate_limit_sec: float = 0.5,
    skip_dirs: list[Path] | None = None,
    skip_paths: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Download each URL into output_dir as {blog}_{hash}{ext}.
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
        filename = _filename_for_url(blog, url)
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
            written.append((path, url, blog))
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
