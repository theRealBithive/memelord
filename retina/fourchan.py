"""Scraper for 4chan /wg/ (wallpaper) via the public JSON API; no API key."""

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

_BASE = "https://a.4cdn.org"
_IMAGE_BASE = "https://i.4cdn.org"
_RATE_LIMIT_SEC = 1.0
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")


def _get_json(url: str) -> dict | list:
    """Fetch URL and parse JSON. Raises on HTTP or connection errors."""
    req = Request(url, headers={"User-Agent": "Janulon/1.0"})
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def get_index(board: str, page: int = 1) -> dict:
    """Fetch one index page for a board. Page is 1-based."""
    url = f"{_BASE}/{board}/{page}.json"
    return _get_json(url)


def get_thread(board: str, thread_no: int) -> dict:
    """Fetch full thread JSON for a board."""
    url = f"{_BASE}/{board}/thread/{thread_no}.json"
    return _get_json(url)


def image_url_from_post(post: dict, board: str) -> str | None:
    """
    Return the full image URL for a post if it has a valid image attachment.
    Returns None if the post has no image, or the file was deleted.
    """
    if post.get("filedeleted"):
        return None
    tim = post.get("tim")
    ext = post.get("ext")
    if tim is None or not ext or not ext.lower().startswith("."):
        return None
    ext_lower = ext.lower()
    if ext_lower not in _IMAGE_EXTENSIONS:
        return None
    return f"{_IMAGE_BASE}/{board}/{tim}{ext}"


def iter_image_urls(
    board: str = "wg",
    index_pages: int = 2,
    *,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
) -> list[str]:
    """
    Collect image URLs from the board's index pages (no thread fetching).
    Respects 4chan API: at most one request per second.
    """
    seen: set[str] = set()
    out: list[str] = []
    page = 0
    for page in range(1, index_pages + 1):
        logger.info(
            "Fetching index page {}/{} for board /{}/", page, index_pages, board
        )
        time.sleep(rate_limit_sec)
        try:
            data = get_index(board, page)
        except (HTTPError, URLError) as e:
            logger.warning("Failed to fetch page {}: {}", page, e)
            break
        threads = data.get("threads") if isinstance(data, dict) else []
        n_page = 0
        for thread in threads:
            posts = thread.get("posts") if isinstance(thread, dict) else []
            for post in posts:
                if not isinstance(post, dict):
                    continue
                url = image_url_from_post(post, board)
                if url and url not in seen:
                    seen.add(url)
                    out.append(url)
                    n_page += 1
        logger.info(
            "Page {}: found {} image URLs ({} unique so far)", page, n_page, len(out)
        )
    logger.info("Scrape complete: {} unique image URLs from {} pages", len(out), page)
    return out


def download_images(
    urls: list[str],
    output_dir: Path,
    board: str,
    *,
    rate_limit_sec: float = 0.5,
    skip_dirs: list[Path] | None = None,
) -> list[Path]:
    """
    Download each URL into output_dir. Filename is {board}_{tim}{ext} (e.g. wg_123.png).
    Skips if file already exists in output_dir or in any of skip_dirs (e.g. data/corpus,
    data/void). Returns list of paths written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_dirs = [Path(d) for d in (skip_dirs or [])]
    total = len(urls)
    logger.info("Downloading {} images to {}", total, output_dir.resolve())
    written: list[Path] = []
    skipped_output = 0
    skipped_sets = 0
    for url in urls:
        name = Path(url).name
        if not name or name == ".":
            continue
        filename = f"{board}_{name}"
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
        try:
            time.sleep(rate_limit_sec)
            req = Request(url, headers={"User-Agent": "Janulon/1.0"})
            with urlopen(req, timeout=30) as resp:
                path.write_bytes(resp.read())
            written.append(path)
            logger.info("Downloaded {}/{}: {}", len(written), total, filename)
        except (HTTPError, URLError, OSError) as e:
            logger.warning("Failed to download {}: {}", filename, e)
    if skipped_output or skipped_sets:
        logger.info(
            "Skipped {} (already in output), {} (already in corpus/void)",
            skipped_output,
            skipped_sets,
        )
    logger.info(
        "Downloaded {} of {} images to {}", len(written), total, output_dir.resolve()
    )
    return written
