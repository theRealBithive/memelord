"""Scraper for 4chan /wg/ (wallpaper) via the public JSON API; no API key."""

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

from retina import image_validation

_BASE = "https://a.4cdn.org"
_IMAGE_BASE = "https://i.4cdn.org"
_RATE_LIMIT_SEC = 1.0
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")


def _get_json(url: str) -> dict | list:
    """Fetch URL and parse JSON. Raises on HTTP or connection errors."""
    req = Request(url, headers={"User-Agent": "Janulon/1.0"})
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def get_thread_list(board: str) -> list[dict]:
    """
    Fetch every live thread on a board via threads.json.

    Why threads.json rather than paging {page}.json: a single request returns
    every thread number across all index pages (each with a ``last_modified``
    stamp), which we then expand into full-thread fetches. The index and catalog
    endpoints only ever expose the OP plus the last few replies — the remainder
    is counted in ``omitted_images`` and never reappears no matter how often you
    poll — so an index-only scrape silently drops the bulk of the images on
    dump-heavy boards like /wg/ (measured ~94% missed).
    """
    data = _get_json(f"{_BASE}/{board}/threads.json")
    threads: list[dict] = []
    if isinstance(data, list):
        for page in data:
            if not isinstance(page, dict):
                continue
            for thread in page.get("threads", []):
                if isinstance(thread, dict) and thread.get("no") is not None:
                    threads.append(thread)
    return threads


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
    *,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
    max_threads: int | None = None,
) -> list[str]:
    """
    Collect image URLs from every post of every live thread on a board.

    Why full-thread fetching instead of reading index pages: 4chan's index and
    catalog JSON truncate each thread to the OP plus the last few replies, hiding
    the rest in ``omitted_images``. On image-dump boards (/wg/) that is the great
    majority of the content, and the buried middle never resurfaces in the index
    — so scraping more frequently cannot recover it; only fetching the full
    thread can. We enumerate threads once via threads.json, then fetch each.

    Rate limiting: the 4chan API permits at most one request per second, applied
    across the whole a.4cdn.org host (not per board). Every API call here — the
    thread list and each individual thread — is preceded by a ``rate_limit_sec``
    sleep, including the first call, so back-to-back board scrapes stay within
    budget too. Image downloads hit the separate i.4cdn.org CDN and are paced in
    ``download_images``.
    """
    try:
        time.sleep(rate_limit_sec)
        threads = get_thread_list(board)
    except (HTTPError, URLError, OSError) as e:
        # OSError covers socket read timeouts (socket.timeout/TimeoutError),
        # which are NOT wrapped in URLError; without it a slow response here
        # would propagate out and abort the entire multi-source scrape, since
        # scraper.run has no per-source guard.
        logger.warning("Failed to fetch thread list for /{}/: {}", board, e)
        return []
    if max_threads is not None:
        threads = threads[:max_threads]
    logger.info("Board /{}/: {} live threads to scan", board, len(threads))

    seen: set[str] = set()
    out: list[str] = []
    for i, thread in enumerate(threads, start=1):
        thread_no = thread.get("no")
        time.sleep(rate_limit_sec)
        try:
            data = get_thread(board, thread_no)
        except (HTTPError, URLError, OSError) as e:
            # Threads 404 routinely (pruned between the list fetch and now), and
            # a read timeout (socket.timeout/TimeoutError, an OSError not wrapped
            # in URLError) is increasingly likely across hundreds of fetches.
            # Skip the thread and keep scanning rather than aborting the board —
            # and, since scraper.run has no per-source guard, the whole scrape.
            logger.warning("Failed to fetch thread {}: {}", thread_no, e)
            continue
        posts = data.get("posts") if isinstance(data, dict) else []
        n_thread = 0
        for post in posts:
            if not isinstance(post, dict):
                continue
            url = image_url_from_post(post, board)
            if url and url not in seen:
                seen.add(url)
                out.append(url)
                n_thread += 1
        logger.info(
            "Thread {}/{} (#{}): {} new images ({} unique total)",
            i,
            len(threads),
            thread_no,
            n_thread,
            len(out),
        )
    logger.info(
        "Scrape complete: {} unique image URLs from {} threads on /{}/",
        len(out),
        len(threads),
        board,
    )
    return out


def download_images(
    urls: list[str],
    output_dir: Path,
    board: str,
    *,
    rate_limit_sec: float = 0.5,
    skip_dirs: list[Path] | None = None,
    skip_paths: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Download each URL into output_dir. Filename is {board}_{tim}{ext} (e.g. wg_123.png).
    Skips if file already exists in output_dir, in skip_dirs, or path is in skip_paths (DB).
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
            written.append((path, url, board))
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
        "Downloaded {} of {} images to {}", len(written), total, output_dir.resolve()
    )
    return written
