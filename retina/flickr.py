"""
Scraper for Flickr group pools and user photostreams.

Two ways in, chosen per run by whether FLICKR_API_KEY is set:

- With a key, the REST API (flickr.groups.pools.getPhotos /
  flickr.people.getPublicPhotos) pages through the pool newest first. The
  source's cursor is the range fetched without a gap; each run first takes
  what was added since and then works down the backlog (V4, V12–V14, see
  _iter_api).
- Without a key, the public feed (services/feeds/*.gne) is the only anonymous
  endpoint. It lists the 20 newest photos and cannot page, so it never writes
  a cursor: a cursor taken from the feed would make a later key-backed run
  skip the backlog the feed never reached (V3).

Sources are stored in one canonical spelling, "group/<id>" or "user/<id>",
produced by normalize_source() from a pasted flickr.com URL. The canonical name
is also the images' source_label, so every Flickr source is its own taste
category (V10).

Everything a response contains is hostile (OWASP A10, SSRF): an image is only
fetched from https://*.staticflickr.com, checked before the request and again
on every redirect, and the photo id that names the file must be all digits.
The API key travels in the query string, so no request URL is ever logged (V9).
"""

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from loguru import logger

from retina import image_validation

GROUP = "group"
USER = "user"

# Upper bound on the photos one API run collects. The first run of a large
# pool would otherwise download and encode thousands of images inside one
# scrape, and the CPU host encodes about 2000 per hour at 448 px. Nothing is
# lost to the cap: what a run leaves out stays outside the fetched range and
# comes in a later run (V13).
RUN_LIMIT = 1000

_API_URL = "https://www.flickr.com/services/rest/"
_GROUP_FEED_URL = "https://www.flickr.com/services/feeds/groups_pool.gne"
_USER_FEED_URL = "https://www.flickr.com/services/feeds/photos_public.gne"
_API_PAGE_SIZE = 500
_RATE_LIMIT_SEC = 1.0
_USER_AGENT = "Janulon/1.0"

# Largest first: url_l is 1024 px on the long edge, url_c 800, url_z 640 (V6).
_API_SIZE_FIELDS = ("url_l", "url_c", "url_z")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")

_NSID_RE = re.compile(r"\d+@N\d{2}")
_ALIAS_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_PHOTO_ID_RE = re.compile(r"\d{1,20}")
_FEED_PHOTO_LINK_RE = re.compile(r"/photos/[^/]+/(\d{1,20})(?:/|$)")
_FLICKR_HOSTS = ("flickr.com", "www.flickr.com", "m.flickr.com")
_URL_KIND_BY_SECTION = {"groups": GROUP, "photos": USER, "people": USER}

# Path words under /photos/ and /groups/ that are Flickr pages, not a user or
# a group: /photos/tags/cats is a tag search, not the user "tags" (V2).
_RESERVED_IDS = frozenset(
    {
        "browse",
        "camera",
        "create",
        "explore",
        "friends",
        "me",
        "organize",
        "places",
        "search",
        "tags",
        "upload",
    }
)

# HTTPException covers a truncated body (IncompleteRead), which is neither an
# OSError nor a URLError.
_FETCH_ERRORS = (HTTPError, URLError, HTTPException, json.JSONDecodeError, OSError, ValueError)


class FlickrApiError(Exception):
    """The API answered, but with stat=fail (bad key, unknown group, …)."""


def is_nsid(identifier: str) -> bool:
    """True for a numeric Flickr id such as 419512@N22."""
    return _NSID_RE.fullmatch(identifier) is not None


def _is_valid_identifier(identifier: str) -> bool:
    if identifier.lower() in _RESERVED_IDS:
        return False
    if is_nsid(identifier):
        return True
    return _ALIAS_RE.fullmatch(identifier) is not None


def _canonical(kind: str, identifier: str) -> str | None:
    if not _is_valid_identifier(identifier):
        return None
    return f"{kind}/{identifier}"


def _normalize_short_form(text: str) -> str | None:
    """"group/<id>" or "user/<id>"; the caller has checked the prefix."""
    kind, _, identifier = text.partition("/")
    return _canonical(kind, identifier)


def _normalize_url(text: str) -> str | None:
    if "://" not in text:
        text = "https://" + text
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https"):
        return None
    if parts.hostname not in _FLICKR_HOSTS:
        return None
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) < 2:
        return None
    kind = _URL_KIND_BY_SECTION.get(segments[0])
    if kind is None:
        return None
    return _canonical(kind, segments[1])


def normalize_source(text: str) -> str | None:
    """
    Turn what the operator pasted into "group/<id>" or "user/<id>", or None.

    Pure (no network), so the add-source view can run it on every request and
    the scraper can re-check a stored name before building a request from it.
    A bare id without "group/" or "user/" is refused: a group NSID and a user
    NSID look alike, and guessing wrong scrapes the wrong thing silently. An
    alias (a name instead of a number) is kept as typed; only the API can
    resolve it, which happens at scrape time (V11).
    """
    text = text.strip()
    if not text:
        return None
    if text.startswith((GROUP + "/", USER + "/")):
        return _normalize_short_form(text)
    return _normalize_url(text)


def is_allowed_image_url(url: str) -> bool:
    """True only for https URLs on Flickr's image servers (V7, OWASP A10)."""
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    if parts.scheme != "https":
        return False
    if parts.port is not None:
        return False
    host = parts.hostname
    if host is None:
        return False
    return host == "staticflickr.com" or host.endswith(".staticflickr.com")


class _AllowlistRedirectHandler(HTTPRedirectHandler):
    """
    Follow a redirect only when its target is an allowed image URL.

    urlopen follows 3xx to any host by default, so the pre-request check alone
    would not stop a redirect into the internal network. It is also how Flickr
    answers for a size that does not exist: a redirect to a placeholder picture
    on another host, which would pass the readability check and land in the
    library. Returning None makes urllib raise HTTPError for the 3xx instead.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not is_allowed_image_url(newurl):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_image_opener = build_opener(_AllowlistRedirectHandler)


def _get_json(url: str) -> dict:
    """Fetch a Flickr API or feed URL and parse the JSON object."""
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode())
    if not isinstance(payload, dict):
        raise ValueError("Flickr answered with something other than a JSON object")
    return payload


def _call_api(method: str, params: dict, api_key: str) -> dict:
    query = {
        "method": method,
        "api_key": api_key,
        "format": "json",
        "nojsoncallback": "1",
        **params,
    }
    payload = _get_json(f"{_API_URL}?{urlencode(query)}")
    if payload.get("stat") != "ok":
        code = payload.get("code")
        message = str(payload.get("message", ""))[:200]
        raise FlickrApiError(f"Flickr API error {code}: {message}")
    return payload


def _lookup_nsid(kind: str, alias: str, api_key: str) -> str | None:
    """Resolve an alias to the numeric id through flickr.urls.lookup*."""
    if kind == GROUP:
        payload = _call_api(
            "flickr.urls.lookupGroup",
            {"url": f"https://www.flickr.com/groups/{alias}/"},
            api_key,
        )
        found = payload.get("group")
    else:
        payload = _call_api(
            "flickr.urls.lookupUser",
            {"url": f"https://www.flickr.com/photos/{alias}/"},
            api_key,
        )
        found = payload.get("user")
    if not isinstance(found, dict):
        return None
    nsid = found.get("id")
    if not isinstance(nsid, str) or not is_nsid(nsid):
        return None
    return nsid


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _api_item(entry: dict) -> tuple[str, str, None] | None:
    """(photo_id, largest allowed URL, no fallback) for one API photo entry."""
    photo_id = entry.get("id")
    if not isinstance(photo_id, str) or _PHOTO_ID_RE.fullmatch(photo_id) is None:
        return None
    for size_field in _API_SIZE_FIELDS:
        url = entry.get(size_field)
        if is_allowed_image_url(url):
            return (photo_id, url, None)
    return None


def _api_page_request(kind: str, nsid: str, page: int) -> tuple[str, dict]:
    if kind == GROUP:
        method = "flickr.groups.pools.getPhotos"
        params = {"group_id": nsid, "extras": ",".join(_API_SIZE_FIELDS)}
    else:
        method = "flickr.people.getPublicPhotos"
        extras = ",".join((*_API_SIZE_FIELDS, "date_upload"))
        params = {"user_id": nsid, "extras": extras}
    params["per_page"] = str(_API_PAGE_SIZE)
    params["page"] = str(page)
    return method, params


def _timestamp_field(kind: str) -> str:
    """
    The time the cursor follows.

    A pool is ordered by when a photo was added to the pool, not when it was
    uploaded: an old photo posted to the pool today is new to the pool, and a
    cursor on the upload date would skip it. So the pool follows dateadded and
    a photostream follows dateupload.
    """
    if kind == GROUP:
        return "dateadded"
    return "dateupload"


class _WalkInterrupted(Exception):
    """A page could not be read; the walk stopped before the end of the pool."""


def _api_entries(kind: str, nsid: str, api_key: str, rate_limit_sec: float) -> Iterator[dict]:
    """
    Yield the photo entries of a pool or photostream newest first, page by page.

    Returns normally only after the last page Flickr reports. Anything else,
    a failed request, a page without a photo list or page count, or an empty
    page anywhere but in an empty pool, raises _WalkInterrupted, so a broken
    or cut-off answer can never be taken for "the backlog is complete" (V14).
    Flickr fills every page up to the reported count, so an empty one means
    the answer was cut off, which is how a pagination limit deep in a big
    pool would show itself; it must not end the backlog silently.
    """
    page = 1
    while True:
        method, params = _api_page_request(kind, nsid, page)
        time.sleep(rate_limit_sec)
        try:
            payload = _call_api(method, params, api_key)
            photos = payload.get("photos")
            if not isinstance(photos, dict) or not isinstance(photos.get("photo"), list):
                raise ValueError("Flickr answered a page without a photo list")
            total_pages = _int_or_none(photos.get("pages"))
            if total_pages is None:
                raise ValueError("Flickr answered a page without a page count")
            entries = photos["photo"]
            empty_pool = page == 1 and total_pages <= 1
            if not entries and not empty_pool:
                raise ValueError(f"Flickr answered page {page} of {total_pages} empty")
        except (FlickrApiError, *_FETCH_ERRORS) as error:
            logger.warning("Flickr: page {} of {}/{} failed: {}", page, kind, nsid, error)
            raise _WalkInterrupted from error
        yield from entries
        if page >= total_pages:
            return
        page += 1


def _parse_cursor(cursor: str | None) -> tuple[int, int] | None:
    """
    "<oldest>:<newest>" → (oldest, newest), or None for no or an unreadable mark.

    The pair describes the range the source has fetched without a gap (V12);
    oldest = 0 means the backlog is complete. An unreadable mark counts as
    "nothing fetched yet": the walk starts from the top again, and the
    per-photo filename check keeps that from downloading anything twice.
    """
    if not isinstance(cursor, str):
        return None
    oldest_text, separator, newest_text = cursor.partition(":")
    if separator != ":":
        return None
    oldest = _int_or_none(oldest_text)
    newest = _int_or_none(newest_text)
    if oldest is None or newest is None:
        return None
    if oldest < 0 or oldest > newest:
        return None
    return oldest, newest


def _format_cursor(oldest: int, newest: int) -> str:
    return f"{oldest}:{newest}"


# (timestamp, item) for one photo of an API walk.
_Entry = tuple[int, tuple[str, str, None]]


def _take_whole_groups(entries: list[_Entry], budget: int, must_progress: bool) -> list[_Entry]:
    """
    Take entries from the front, whole groups of one timestamp at a time.

    A group of photos sharing one second is never split: the range boundary
    then always falls between two seconds, so everything inside the range is
    fetched (V12). When must_progress is set and the very first group alone is
    larger than the budget, it is taken whole anyway, or a run could never get
    past it.
    """
    taken: list[_Entry] = []
    start = 0
    while start < len(entries):
        group_stamp = entries[start][0]
        end = start
        while end < len(entries) and entries[end][0] == group_stamp:
            end += 1
        group = entries[start:end]
        fits = len(taken) + len(group) <= budget
        oversized_first_group = must_progress and not taken and not fits
        if not fits and not oversized_first_group:
            break
        taken.extend(group)
        if oversized_first_group:
            break
        start = end
    return taken


def _backlog_cut_is_known(backlog: list[_Entry], budget_left: int) -> bool:
    """True once the group that straddles the budget has been read to its end."""
    if len(backlog) <= budget_left:
        return False
    stamp_at_budget = backlog[budget_left - 1][0]
    return backlog[-1][0] < stamp_at_budget


@dataclass
class _Walk:
    """What one API walk read, sorted against the range fetched so far."""

    at_newest: list[_Entry] = field(default_factory=list)
    new: list[_Entry] = field(default_factory=list)
    backlog: list[_Entry] = field(default_factory=list)
    taken_new: list[_Entry] = field(default_factory=list)
    all_new_read: bool = False
    reached_end: bool = False
    failed: bool = False


def _backlog_budget(walk: _Walk) -> int:
    """
    What the backlog may take this run: the budget the new photos left over,
    but nothing while new photos are still waiting (V4: new ones first; a new
    second that did not fit must not be overtaken by the backlog).
    """
    if len(walk.taken_new) < len(walk.new):
        return 0
    return RUN_LIMIT - len(walk.taken_new)


def _read_walk(entries: Iterator[dict], fetched: tuple[int, int] | None, time_field: str) -> _Walk:
    """
    Read the newest-first walk only as far as this run needs it.

    Each entry falls into one of four places: above the newest mark (new),
    on the newest mark (seen again, see _iter_api), inside the range (skipped)
    or below the oldest mark (backlog). Before anything is fetched, everything
    is backlog. Reading stops once all new photos are known and the backlog
    has been read past the group that straddles the remaining budget.
    """
    walk = _Walk()
    if fetched is None:
        oldest, newest = None, None
        backlog_open = True
        walk.all_new_read = True
    else:
        oldest, newest = fetched
        backlog_open = oldest > 0

    try:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            stamp = _int_or_none(entry.get(time_field))
            item = _api_item(entry)
            if stamp is None or item is None:
                continue

            if fetched is None:
                walk.backlog.append((stamp, item))
            elif stamp > newest:
                walk.new.append((stamp, item))
            elif stamp == newest:
                walk.at_newest.append((stamp, item))
            else:
                if not walk.all_new_read:
                    walk.all_new_read = True
                    oldest_new_first = list(reversed(walk.new))
                    walk.taken_new = _take_whole_groups(oldest_new_first, RUN_LIMIT, True)
                if backlog_open and stamp < oldest:
                    walk.backlog.append((stamp, item))

            if not walk.all_new_read:
                continue
            budget_left = _backlog_budget(walk)
            if not backlog_open or budget_left <= 0:
                break
            if _backlog_cut_is_known(walk.backlog, budget_left):
                break
        else:
            walk.reached_end = True
    except _WalkInterrupted:
        walk.failed = True

    if walk.reached_end and not walk.all_new_read:
        walk.all_new_read = True
        oldest_new_first = list(reversed(walk.new))
        walk.taken_new = _take_whole_groups(oldest_new_first, RUN_LIMIT, True)
    return walk


def _next_cursor(
    fetched: tuple[int, int] | None,
    taken_new: list[_Entry],
    taken_backlog: list[_Entry],
    backlog_complete: bool,
) -> str | None:
    """
    The range fetched without a gap after this run, or None to keep the mark.

    The new photos taken are the oldest whole seconds above the old range and
    the backlog photos taken the newest whole seconds below it, so the old
    range widened by both still has no gap (V12).
    """
    stamps = [stamp for stamp, _ in taken_new + taken_backlog]
    if fetched is None and not stamps and not backlog_complete:
        return None

    if fetched is not None:
        newest = max([fetched[1], *stamps])
    elif stamps:
        newest = max(stamps)
    else:
        newest = 0

    if backlog_complete:
        oldest = 0
    elif taken_backlog:
        oldest = min(stamp for stamp, _ in taken_backlog)
    else:
        # Also keeps a complete backlog complete: its oldest mark is already 0
        # and no backlog is read for it.
        oldest = fetched[0]
    return _format_cursor(oldest, newest)


def _iter_api(
    kind: str, identifier: str, since: str | None, api_key: str, rate_limit_sec: float
) -> tuple[list[tuple[str, str, None]], str | None]:
    """
    One run over the pool or photostream: new photos first, then the backlog.

    The cursor is the range fetched so far without a gap (V12). Each run takes
    at most RUN_LIMIT photos (V4):

    1. New are the photos above the newest mark. The pool lists newest first,
       so the walk reads until the first entry below the mark and takes the
       *oldest* new seconds; taking the newest would leave a gap above the old
       mark whenever more than RUN_LIMIT photos arrived (V13).
    2. What is left of the budget goes to the backlog below the oldest mark,
       top down. Reaching the real end of the pool with everything taken sets
       the oldest mark to 0, after which runs fetch only new photos (V14).

    The lower boundary compares strictly: no photo can ever appear below it,
    because dateadded is the time the photo was added. The upper boundary is
    inclusive: a photo added in the same second as the previous run's newest,
    but after that run, carries the boundary's own timestamp. That second is
    returned again on every run and does not count against the budget (the
    files of its photos are usually there already and are skipped).

    A page failure stops the walk. If the new photos were not all read yet,
    the run takes nothing and keeps the old mark (V5, V8). Otherwise it takes
    what it read, minus the last backlog second (it may be cut off), which
    still joins the range without a gap; the oldest mark never becomes 0 on a
    failure.
    """
    if is_nsid(identifier):
        nsid = identifier
    else:
        nsid = _lookup_nsid(kind, identifier, api_key)
        if nsid is None:
            logger.warning("Flickr: could not resolve {}/{} to an id", kind, identifier)
            return [], None

    fetched = _parse_cursor(since)
    entries = _api_entries(kind, nsid, api_key, rate_limit_sec)
    walk = _read_walk(entries, fetched, _timestamp_field(kind))
    if not walk.all_new_read:
        return [], None

    backlog = walk.backlog
    if walk.failed and backlog:
        last_stamp = backlog[-1][0]
        backlog = [entry for entry in backlog if entry[0] != last_stamp]
    budget_left = _backlog_budget(walk)
    must_progress = not walk.taken_new
    taken_backlog = _take_whole_groups(backlog, budget_left, must_progress)
    backlog_complete = walk.reached_end and len(taken_backlog) == len(walk.backlog)

    new_cursor = _next_cursor(fetched, walk.taken_new, taken_backlog, backlog_complete)
    taken = walk.at_newest + walk.taken_new + taken_backlog
    items = [item for _, item in taken]

    if backlog_complete:
        backlog_note = " (backlog complete)"
    else:
        backlog_note = ""
    logger.info(
        "Flickr: {}/{}: {} new, {} from the backlog{}",
        kind,
        identifier,
        len(walk.taken_new),
        len(taken_backlog),
        backlog_note,
    )
    return items, new_cursor


def _large_feed_url(small_url: str) -> str:
    """The feed only lists the 240 px "_m" picture; "_b" is the 1024 px one."""
    if small_url.endswith("_m.jpg"):
        return small_url[: -len("_m.jpg")] + "_b.jpg"
    return small_url


def _feed_item(entry: dict) -> tuple[str, str, str | None] | None:
    link = entry.get("link")
    media = entry.get("media")
    if not isinstance(link, str) or not isinstance(media, dict):
        return None
    match = _FEED_PHOTO_LINK_RE.search(urlsplit(link).path)
    if match is None:
        return None
    small_url = media.get("m")
    if not is_allowed_image_url(small_url):
        return None
    large_url = _large_feed_url(small_url)
    if large_url == small_url:
        return (match.group(1), small_url, None)
    return (match.group(1), large_url, small_url)


def _iter_feed(kind: str, nsid: str) -> list[tuple[str, str, str | None]]:
    if kind == GROUP:
        base = _GROUP_FEED_URL
    else:
        base = _USER_FEED_URL
    query = urlencode({"id": nsid, "format": "json", "nojsoncallback": "1"})
    try:
        payload = _get_json(f"{base}?{query}")
    except _FETCH_ERRORS as error:
        logger.warning("Flickr: feed of {}/{} failed: {}", kind, nsid, error)
        return []
    entries = payload.get("items")
    if not isinstance(entries, list):
        return []
    items = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = _feed_item(entry)
        if item is not None:
            items.append(item)
    logger.info("Flickr: feed of {}/{} lists {} photos", kind, nsid, len(items))
    return items


def iter_image_items(
    source: str,
    *,
    since: str | None = None,
    api_key: str | None = None,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
) -> tuple[list[tuple[str, str, str | None]], str | None]:
    """
    Collect (photo_id, image_url, fallback_url) for one canonical source.

    Returns the items and the new cursor (None = leave the stored one alone).
    The stored name is checked again before anything is requested: the DB row
    may predate the validation, and config.toml bypasses the view entirely.
    """
    if normalize_source(source) != source:
        logger.warning("Flickr: {!r} is not a valid source name, skipped", source)
        return [], None
    kind, _, identifier = source.partition("/")
    if api_key:
        # Deliberately broad: the key sits in the local variables of the API
        # frames, and an exception that escapes to scraper.run is logged with
        # logger.exception, whose diagnose mode prints those locals (V9). Only
        # the exception's type and text leave this module.
        try:
            return _iter_api(kind, identifier, since, api_key, rate_limit_sec)
        except Exception as error:
            logger.warning("Flickr: {} failed: {}: {}", source, type(error).__name__, error)
            return [], None
    if not is_nsid(identifier):
        logger.warning(
            "Flickr: {} needs FLICKR_API_KEY or the numeric ID (e.g. group/419512@N22)",
            source,
        )
        return [], None
    return _iter_feed(kind, identifier), None


def _extension_for(url: str) -> str:
    suffix = Path(urlsplit(url).path).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return suffix
    return ".jpg"


def _already_downloaded(photo_id: str, directories: list[Path]) -> bool:
    """Any size of this photo, in any known directory, counts (V6)."""
    existing: list[Path] = []
    for directory in directories:
        existing.extend(directory.glob(f"flickr_{photo_id}.*"))
    return len(existing) > 0


def _fetch_image(url: str, path: Path) -> bool:
    """Download one allowed URL to path; False if refused, failed or unreadable."""
    if not is_allowed_image_url(url):
        return False
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with _image_opener.open(request, timeout=30) as response:
            if not is_allowed_image_url(response.geturl()):
                return False
            path.write_bytes(response.read())
    except (HTTPError, URLError, OSError) as error:
        logger.warning("Flickr: download of {} failed: {}", path.name, error)
        return False
    if not image_validation.is_readable_image(path):
        path.unlink()
        logger.warning("Flickr: removed unreadable download {}", path.name)
        return False
    return True


def download_images(
    items: list[tuple[str, str, str | None]],
    output_dir: Path,
    source: str,
    *,
    rate_limit_sec: float = 0.5,
    skip_dirs: list[Path] | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Download each photo once as flickr_<photo_id><ext>.

    The file is named by the photo, not by the URL, so the 1024 px and the
    240 px version of one photo, or url_l and url_c of it, can never both be
    stored (V6). The fallback URL (the feed's "_m") is tried only when the
    large one fails. Returns (path, source_url, source_label) per file written;
    the label is the canonical source name (V10).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    known_dirs = [output_dir] + [Path(d) for d in (skip_dirs or [])]
    written: list[tuple[Path, str, str]] = []
    skipped = 0
    for photo_id, url, fallback_url in items:
        if _PHOTO_ID_RE.fullmatch(photo_id) is None:
            continue
        if _already_downloaded(photo_id, known_dirs):
            skipped += 1
            continue
        time.sleep(rate_limit_sec)
        path = output_dir / f"flickr_{photo_id}{_extension_for(url)}"
        if _fetch_image(url, path):
            written.append((path, url, source))
            continue
        if fallback_url is None:
            continue
        fallback_path = output_dir / f"flickr_{photo_id}{_extension_for(fallback_url)}"
        if _fetch_image(fallback_url, fallback_path):
            written.append((fallback_path, fallback_url, source))
    logger.info(
        "Flickr: downloaded {} of {} photos of {} ({} already present)",
        len(written),
        len(items),
        source,
        skipped,
    )
    return written
