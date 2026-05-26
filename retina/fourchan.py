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
    since_modified: int | None = None,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
    max_threads: int | None = None,
) -> tuple[list[str], int | None]:
    """
    Collect image URLs from every post of every thread modified since last scrape.

    Returns ``(urls, new_cursor)``. ``new_cursor`` is a ``last_modified`` unix
    stamp to persist as the source cursor and pass back as ``since_modified`` next
    run; on a thread-list failure it returns the cursor unchanged so a transient
    error never resets incremental progress.

    Why full-thread fetching instead of reading index pages: 4chan's index and
    catalog JSON truncate each thread to the OP plus the last few replies, hiding
    the rest in ``omitted_images``. On image-dump boards (/wg/) that is the great
    majority of the content, and the buried middle never resurfaces in the index
    — so scraping more frequently cannot recover it; only fetching the full
    thread can. We enumerate threads once via threads.json, then fetch each.

    Why incremental via ``last_modified``: threads.json stamps every thread with
    the time of its last post/edit/delete. Adding a post always advances that
    stamp, so a thread whose stamp is ``<= since_modified`` cannot hold an image
    we have not already captured — we skip it and save the per-thread request.
    Without this, every scrape re-fetched every live thread (hundreds of requests
    on /wg/) regardless of what changed.

    The cursor is a *watermark over the contiguous run of successfully-fetched
    threads, oldest first*, NOT ``max(last_modified)`` over successes. Threads are
    sorted ascending and the cursor only advances while no fetch has errored yet;
    on the first error it freezes (we keep fetching the rest to capture what we
    can, but don't promise it via the cursor). This is deliberate: a plain max
    would jump the cursor past a failed thread sitting *between* two successes,
    permanently skipping it next run. Freezing on first error costs at most some
    redundant refetches after the error, never a lost image. Do not "simplify" it
    back to a max. The ascending order also lets ``max_threads`` act as a safety
    valve that drains the oldest backlog first across runs (trading freshness for
    eventual completeness) rather than stranding the threads it doesn't reach.

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
        # scraper.run has no per-source guard. Return the cursor unchanged.
        logger.warning("Failed to fetch thread list for /{}/: {}", board, e)
        return [], since_modified

    prev = since_modified or 0
    # Oldest-first so the cursor watermark and any max_threads cap both advance
    # from the bottom of the changed range (see docstring).
    changed = sorted(
        (t for t in threads if (t.get("last_modified") or 0) > prev),
        key=lambda t: t.get("last_modified") or 0,
    )
    if max_threads is not None:
        changed = changed[:max_threads]
    logger.info(
        "Board /{}/: {} live threads, {} changed since last scrape",
        board,
        len(threads),
        len(changed),
    )

    seen: set[str] = set()
    out: list[str] = []
    new_cursor = prev
    cursor_advancing = True
    for i, thread in enumerate(changed, start=1):
        thread_no = thread.get("no")
        last_modified = thread.get("last_modified") or 0
        time.sleep(rate_limit_sec)
        try:
            data = get_thread(board, thread_no)
        except (HTTPError, URLError, OSError) as e:
            # Threads 404 routinely (pruned between the list fetch and now), and
            # a read timeout (socket.timeout/TimeoutError, an OSError not wrapped
            # in URLError) is increasingly likely across hundreds of fetches.
            # Skip the thread and keep scanning rather than aborting the board —
            # and, since scraper.run has no per-source guard, the whole scrape.
            # Freeze the cursor here so the failed thread is retried next run.
            cursor_advancing = False
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
        if cursor_advancing:
            new_cursor = last_modified
        logger.info(
            "Thread {}/{} (#{}): {} new images ({} unique total)",
            i,
            len(changed),
            thread_no,
            n_thread,
            len(out),
        )
    logger.info(
        "Scrape complete: {} unique image URLs from {} threads on /{}/",
        len(out),
        len(changed),
        board,
    )
    return out, new_cursor


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
