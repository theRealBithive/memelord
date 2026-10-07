"""
Tests for retina.flickr: source names, API and feed walks, downloads.

Contract (confirmed 2026-10-07; V4 revised, its boundary rules and V12–V14
added and confirmed the same day):
- V1 Ein Flickr-Gruppen-Pool oder Nutzer-Photostream lässt sich in Config über seine
  flickr.com-Adresse oder als group/<id> / user/<id> eintragen; jede Schreibweise
  desselben Ziels ergibt dieselbe Quelle.
- V2 Eingaben, die keine Flickr-Gruppe und kein Flickr-Nutzer sind (andere Hosts, andere
  Pfade, Flickr-Sonderseiten wie Tags oder Suche, Zeichen außerhalb einer ID, nackte NSID),
  werden mit Meldung abgelehnt; es entsteht keine Quelle.
- V3 Ohne API-Key holt ein Scrape die neuesten Bilder, die der öffentliche Feed nennt
  (höchstens 20), und verändert die Merkmarke der Quelle nicht.
- V4 Mit API-Key holt jeder Lauf höchstens 1000 Bilder: zuerst die seit dem vorigen Lauf
  hinzugekommenen (von den ältesten her), mit dem restlichen Kontingent arbeitet er den
  Altbestand von oben nach unten ab. Zwei Grenzregeln: Eine Sekunde wird nie geteilt (ist
  eine einzelne Sekunde größer als das ganze Kontingent, wird sie ganz genommen), und die
  Sekunde auf der oberen Marke kommt in jedem Lauf außerhalb des Kontingents mit.
- V5 Die Merkmarke rückt erst vor, wenn die Bilder des Laufs in der DB sind; ein
  gescheiterter Lauf wird ab der alten Marke wiederholt.
- V6 Jedes Foto wird höchstens einmal geladen, in der größten der Stufen 1024/800/640 px,
  die es gibt; das 240-px-Vorschaubild nur, wenn es keine größere Fassung gibt.
- V7 Geladen werden nur Bilder von Flickrs Bildservern über https; jede andere Adresse in
  einer Antwort und jede Weiterleitung woandershin wird ignoriert.
- V8 Ein Flickr-Fehler (Netz, ungültiger Key, unbekannte Gruppe, kaputte Antwort) beendet nur
  diese Quelle; die übrigen Quellen des Laufs laufen weiter, die Merkmarke bleibt.
- V9 Der API-Key erscheint nie in einer Logzeile, einer DB-Zeile oder einer Antwort.
- V10 Jedes Bild einer Flickr-Quelle trägt den Namen der Quelle als Herkunft und bildet damit
  eine eigene Taste-Kategorie.
- V11 Ein Alias (Name statt Nummer) wird mit Key in die ID aufgelöst; ohne Key wird die Quelle
  mit einer Logzeile übersprungen, die Key oder numerische ID verlangt.
- V12 Was eine Quelle bisher geholt hat, ist lückenlos: jedes Foto zwischen dem ältesten und
  dem neuesten geholten Zeitpunkt ist geholt.
- V13 Über genug Läufe kommt jedes Foto des Pools genau einmal dran, auch wenn zwischen zwei
  Läufen mehr als 1000 neue dazukommen.
- V14 Ist das Ende des Pools einmal erreicht, holen spätere Läufe nur noch Neues.

Why the boundary rules of V4: a group of photos sharing one second is never
split, so the range boundary always falls between two seconds (V12), and a
single such group larger than the whole budget is taken whole so a run can
always progress (V13).
The second on the newest mark is returned again by every run and does not
count against the budget: a photo added in that second after the previous
run must not be lost, and "genau einmal" in V13 means downloaded once, which
the per-photo filename check guarantees for the re-returned photos.

V5 and V8 (the cursor and the per-source isolation inside a scrape run) are
pinned in tests/ratings/test_scraper_flickr.py; V2's "no source is created" in
tests/ratings/test_source_add.py. This module covers the retina side.
"""

import io
import json
import tempfile
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest
from hypothesis import given
from hypothesis import strategies as st
from loguru import logger

from retina import flickr
from tests.conftest import minimal_png_bytes

API_KEY = "SECRETKEY123"
STATIC = "https://live.staticflickr.com/65535"

# ── generators ────────────────────────────────────────────────────────────────

nsids = st.builds(
    lambda number, suffix: f"{number}@N{suffix:02d}",
    st.integers(min_value=1, max_value=10**12),
    st.integers(min_value=0, max_value=99),
)
# A reserved word is a Flickr page by definition (V2), not an alias.
aliases = st.from_regex(r"[A-Za-z0-9_-]{1,64}", fullmatch=True).filter(
    lambda alias: alias.lower() not in flickr._RESERVED_IDS
)
identifiers = st.one_of(nsids, aliases)
kinds = st.sampled_from([flickr.GROUP, flickr.USER])
schemes = st.sampled_from(["https://", "http://", ""])
flickr_hosts = st.sampled_from(["www.flickr.com", "flickr.com", "m.flickr.com"])
path_tails = st.sampled_from(
    ["", "/", "/pool/", "/pool/with/54645046861", "/discuss/", "/albums/72157", "/favorites"]
)
queries = st.sampled_from(["", "?page=2", "?foo=bar&x=1"])
fragments = st.sampled_from(["", "#top"])
padding = st.sampled_from(["", " ", "\t", "  \n"])


def _section_for(kind: str, draw) -> str:
    if kind == flickr.GROUP:
        return "groups"
    return draw(st.sampled_from(["photos", "people"]))


@st.composite
def spellings(draw):
    """(expected canonical name, one way the operator may type it)."""
    kind = draw(kinds)
    identifier = draw(identifiers)
    expected = f"{kind}/{identifier}"
    pad = draw(padding)
    if draw(st.booleans()):
        return expected, f"{pad}{expected}{pad}"
    section = _section_for(kind, draw)
    url = (
        f"{draw(schemes)}{draw(flickr_hosts)}/{section}/{identifier}"
        f"{draw(path_tails)}{draw(queries)}{draw(fragments)}"
    )
    return expected, f"{pad}{url}{pad}"


bad_identifier_chars = st.sampled_from(list(".%<>'\"!$~:;,= +*()[]{}|\\^`"))


@st.composite
def bad_identifiers(draw):
    alias = draw(st.from_regex(r"[A-Za-z0-9_-]{1,20}", fullmatch=True))
    position = draw(st.integers(min_value=0, max_value=len(alias)))
    return alias[:position] + draw(bad_identifier_chars) + alias[position:]


@st.composite
def invalid_inputs(draw):
    """Things that are not a Flickr group or user (V2)."""
    identifier = draw(identifiers)
    choice = draw(st.integers(min_value=0, max_value=6))
    if choice == 0:
        host = draw(
            st.sampled_from(
                ["example.com", "flickr.com.evil.io", "evilflickr.com", "flickr.org", "127.0.0.1"]
            )
        )
        return f"https://{host}/groups/{identifier}/pool/"
    if choice == 1:
        reserved = draw(st.sampled_from(sorted(flickr._RESERVED_IDS)))
        section = draw(st.sampled_from(["photos", "groups", "people"]))
        return f"https://www.flickr.com/{section}/{reserved}/cats"
    if choice == 2:
        return f"user/{draw(bad_identifiers())}"
    if choice == 3:
        section = draw(st.sampled_from(["photos", "groups"]))
        return f"https://www.flickr.com/{section}/{draw(bad_identifiers())}/"
    if choice == 4:
        return identifier
    if choice == 5:
        page = draw(st.sampled_from(["search/?text=x", "explore", "photos/", "groups", "", "/"]))
        return f"https://www.flickr.com/{page}"
    scheme = draw(st.sampled_from(["ftp://", "javascript:", "file://"]))
    return f"{scheme}www.flickr.com/groups/{identifier}/"


# ── V1 / V2: source names ─────────────────────────────────────────────────────


@given(spellings())
def test_every_spelling_of_one_target_gives_the_same_source(case) -> None:
    """Contract: V1"""
    expected, typed = case
    assert flickr.normalize_source(typed) == expected
    # The canonical name is itself an accepted spelling of the same target.
    assert flickr.normalize_source(expected) == expected


def test_the_example_link_becomes_the_group() -> None:
    """Contract: V1 (the link from the request)."""
    link = "https://www.flickr.com/groups/419512@N22/pool/with/54645046861"
    assert flickr.normalize_source(link) == "group/419512@N22"


@given(invalid_inputs())
def test_non_flickr_input_is_rejected(text) -> None:
    """Contract: V2"""
    assert flickr.normalize_source(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "https://www.flickr.com/search/people/?q=x",
        "https://www.flickr.com/explore/2026/10/07",
        "https://www.flickr.com/photos/",
        "group/",
        "user/a/b",
    ],
)
def test_flickr_pages_that_name_no_group_or_user_are_rejected(text) -> None:
    """Contract: V2"""
    assert flickr.normalize_source(text) is None


def test_bare_nsid_is_rejected() -> None:
    """Contract: V2 (group and user NSIDs look alike, so neither is guessed)."""
    assert flickr.normalize_source("419512@N22") is None


# ── V7: image host allowlist ──────────────────────────────────────────────────


@given(st.sampled_from(["live", "farm1", "farm66", "c1", ""]))
def test_static_flickr_https_is_allowed(subdomain) -> None:
    """Contract: V7"""
    host = f"{subdomain}.staticflickr.com" if subdomain else "staticflickr.com"
    assert flickr.is_allowed_image_url(f"https://{host}/65535/1_a_b.jpg")


@pytest.mark.parametrize(
    "url",
    [
        "http://live.staticflickr.com/65535/1_a_b.jpg",
        "https://live.staticflickr.com:8443/1_a_b.jpg",
        "https://evil.staticflickr.com.attacker.io/1.jpg",
        "https://staticflickr.com@evil.io/1.jpg",
        "https://evilstaticflickr.com/1.jpg",
        "http://169.254.169.254/latest/meta-data",
        "file:///etc/passwd",
        "https:///65535/1_a_b.jpg",
        "",
        None,
        42,
    ],
)
def test_other_urls_are_not_allowed(url) -> None:
    """Contract: V7"""
    assert not flickr.is_allowed_image_url(url)


def test_redirect_to_foreign_host_is_not_followed() -> None:
    """Contract: V7 (urlopen would otherwise follow any 3xx)."""
    handler = flickr._AllowlistRedirectHandler()
    request = Request(f"{STATIC}/1_a_b.jpg")
    for target in ("http://127.0.0.1/x.jpg", "https://example.com/photo_unavailable.png"):
        assert handler.redirect_request(request, None, 302, "Found", {}, target) is None
    followed = handler.redirect_request(request, None, 302, "Found", {}, f"{STATIC}/2_a_b.jpg")
    assert followed is not None
    assert followed.full_url == f"{STATIC}/2_a_b.jpg"


# ── fake Flickr ───────────────────────────────────────────────────────────────


def _query(url: str) -> dict:
    return {key: values[0] for key, values in parse_qs(urlsplit(url).query).items()}


class FakeFlickr:
    """Answers _get_json for the API and the feed; records every request."""

    def __init__(
        self, entries=(), page_size=500, lookup=None, feed_items=(), fail_on_page=None, empty_after_page=None
    ):
        self.entries = list(entries)
        self.page_size = page_size
        self.lookup = lookup or {}
        self.feed_items = list(feed_items)
        self.fail_on_page = fail_on_page
        # Serve empty pages beyond this one while still reporting all pages:
        # what a pagination limit deep in a big pool would look like.
        self.empty_after_page = empty_after_page
        self.failed = False
        self.urls: list[str] = []

    def __call__(self, url: str) -> dict:
        self.urls.append(url)
        query = _query(url)
        # Both endpoints answer JSONP or XML unless asked for plain JSON.
        if query.get("format") != "json" or query.get("nojsoncallback") != "1":
            raise ValueError("Flickr answered with something other than a JSON object")
        if "/services/feeds/" in url:
            return {"items": self.feed_items}
        if query.get("api_key") != API_KEY:
            return {"stat": "fail", "code": 100, "message": "Invalid API Key"}
        method = query["method"]
        if method in ("flickr.urls.lookupGroup", "flickr.urls.lookupUser"):
            alias = query["url"].rstrip("/").rsplit("/", 1)[-1]
            key = "group" if method.endswith("Group") else "user"
            if alias not in self.lookup:
                return {"stat": "fail", "code": 1, "message": "Group not found"}
            return {"stat": "ok", key: {"id": self.lookup[alias]}}
        page = int(query["page"])
        if self.fail_on_page == page:
            self.failed = True
            raise URLError("connection reset")
        per_page = int(query["per_page"])
        start = (page - 1) * per_page
        pages = max(1, -(-len(self.entries) // per_page))
        photo = self.entries[start : start + per_page]
        if self.empty_after_page is not None and page > self.empty_after_page:
            photo = []
        return {"stat": "ok", "photos": {"page": page, "pages": pages, "photo": photo}}

    def api_page_requests(self) -> list[dict]:
        return [_query(url) for url in self.urls if "/services/rest/" in url and "page" in _query(url)]


def _pool_entry(photo_id: int, added: int, uploaded: int = 0) -> dict:
    return {
        "id": str(photo_id),
        "dateadded": str(added),
        "dateupload": str(uploaded),
        "url_l": f"{STATIC}/{photo_id}_s_b.jpg",
    }


def _user_entry(photo_id: int, uploaded: int) -> dict:
    return {"id": str(photo_id), "dateupload": str(uploaded), "url_l": f"{STATIC}/{photo_id}_s_b.jpg"}


def _walk(fake: FakeFlickr, source: str, since=None, api_key=API_KEY):
    with patch.object(flickr, "_get_json", side_effect=fake):
        return flickr.iter_image_items(source, since=since, api_key=api_key, rate_limit_sec=0)


# ── V4, V12–V14: the fetched range ────────────────────────────────────────────


def _range(cursor: str | None) -> tuple[int, int] | None:
    """The test's own reading of "<oldest>:<newest>"."""
    if cursor is None:
        return None
    oldest, newest = cursor.split(":")
    return int(oldest), int(newest)


def _inside(stamp: int, fetched: tuple[int, int]) -> bool:
    oldest, newest = fetched
    return oldest <= stamp <= newest


@st.composite
def timelines(draw):
    """
    A pool that grows between runs, with a small budget and small pages.

    The initial pool has ties; every addition lands at or above the current
    newest second (increment 0 = same second as the previous photo, which
    hits the inclusive upper boundary). Some runs fail on one page.
    """
    initial = sorted(
        draw(st.lists(st.integers(min_value=1, max_value=40), max_size=20)), reverse=True
    )
    runs = draw(
        st.lists(
            st.tuples(
                st.lists(st.integers(min_value=0, max_value=2), max_size=10),
                st.one_of(st.none(), st.integers(min_value=1, max_value=6)),
            ),
            min_size=1,
            max_size=7,
        )
    )
    limit = draw(st.integers(min_value=1, max_value=8))
    page_size = draw(st.integers(min_value=1, max_value=6))
    return initial, runs, limit, page_size


class PoolModel:
    """The pool, the cursor the scraper would store, and what was returned."""

    def __init__(self, initial_stamps: list[int]):
        self.next_id = 1000
        self.pool: list[dict] = []
        for stamp in initial_stamps:
            self.pool.append(self._entry(stamp))
        self.cursor: str | None = None
        self.returned: set[str] = set()
        # The pool as it was when the cursor was last written; see _check_gapless.
        self.pool_at_cursor: list[dict] = []

    def _entry(self, stamp: int) -> dict:
        self.next_id += 1
        return _pool_entry(self.next_id, stamp)

    def stamp(self, photo_id: str) -> int:
        for entry in self.pool:
            if entry["id"] == photo_id:
                return int(entry["dateadded"])
        raise KeyError(photo_id)

    def add(self, increments: list[int]) -> None:
        stamp = max((int(e["dateadded"]) for e in self.pool), default=0)
        added = []
        for increment in increments:
            stamp += increment
            added.append(self._entry(stamp))
        self.pool = list(reversed(added)) + self.pool

    def run(self, page_size: int, limit: int, fail_on_page=None):
        fake = FakeFlickr(self.pool, fail_on_page=fail_on_page)
        with patch.object(flickr, "_API_PAGE_SIZE", page_size), patch.object(flickr, "RUN_LIMIT", limit):
            items, new_cursor = _walk(fake, "group/419512@N22", since=self.cursor)
        return [item[0] for item in items], new_cursor, fake.failed

    def commit(self, ids: list[str], new_cursor: str | None) -> None:
        self.returned |= set(ids)
        if new_cursor is not None:
            self.cursor = new_cursor
            self.pool_at_cursor = list(self.pool)


def _check_run(model: PoolModel, before: tuple[int, int] | None, ids: list[str], new_cursor, failed, limit):
    stamps = [model.stamp(photo_id) for photo_id in ids]

    # V4: at most the budget, beyond the second on the old newest mark,
    # unless the run took one oversized second.
    counted = [stamp for stamp in stamps if before is None or stamp != before[1]]
    assert len(counted) <= limit or len(set(counted)) == 1

    if before is not None:
        oldest, newest = before
        # V4: new photos first: a backlog photo only once every new one is in.
        took_backlog = any(stamp < oldest for stamp in stamps)
        if took_backlog:
            returned_now = model.returned | set(ids)
            new_ids = {e["id"] for e in model.pool if int(e["dateadded"]) > newest}
            assert new_ids <= returned_now
        # V13: a photo comes again only as part of the second on the newest mark.
        for photo_id, stamp in zip(ids, stamps, strict=True):
            if photo_id in model.returned:
                assert stamp == newest
        # V14: once the backlog is complete, nothing older than the mark comes.
        if oldest == 0:
            assert all(stamp >= newest for stamp in stamps)
        # A failure never marks the backlog as complete.
        if failed and new_cursor is not None:
            assert _range(new_cursor)[0] > 0 or oldest == 0
    elif failed and new_cursor is not None:
        assert _range(new_cursor)[0] > 0


def _check_gapless(model: PoolModel) -> None:
    """
    Contract: V12, against the pool as it was when the range was written.

    A photo added later in the same second as the newest mark carries a
    timestamp inside the range without having been there when the range was
    fetched; if the following run fails before it has read the new photos, the
    range stays as it is. The range is a statement about what was fetched, so
    it is checked against the pool of that moment; V13 (everything comes in
    eventually) and the re-returned upper second cover the late arrival.
    """
    fetched = _range(model.cursor)
    if fetched is None:
        return
    for entry in model.pool_at_cursor:
        if _inside(int(entry["dateadded"]), fetched):
            assert entry["id"] in model.returned


@given(timelines())
def test_runs_fetch_new_first_then_backlog_without_gaps_until_everything_is_in(case) -> None:
    """Contract: V4, V12, V13, V14"""
    initial, runs, limit, page_size = case
    model = PoolModel(initial)

    for increments, fail_on_page in runs:
        model.add(increments)
        before = _range(model.cursor)
        ids, new_cursor, failed = model.run(page_size, limit, fail_on_page)
        _check_run(model, before, ids, new_cursor, failed, limit)
        model.commit(ids, new_cursor)
        _check_gapless(model)

    # V13: without further additions, every photo is in after enough runs;
    # each run takes at least one second, so the pool size bounds the count.
    for _ in range(len(model.pool) + 1):
        before = _range(model.cursor)
        ids, new_cursor, failed = model.run(page_size, limit)
        _check_run(model, before, ids, new_cursor, failed, limit)
        model.commit(ids, new_cursor)
        _check_gapless(model)
    assert model.returned == {entry["id"] for entry in model.pool}

    # V14: the backlog is complete and a further run brings only the mark's second.
    final = _range(model.cursor)
    assert final[0] == 0
    ids, _, _ = model.run(page_size, limit)
    assert all(model.stamp(photo_id) == final[1] for photo_id in ids)


@given(
    st.lists(st.integers(min_value=1, max_value=40), min_size=1, max_size=25),
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=6),
    st.integers(min_value=1, max_value=8),
)
def test_empty_pages_before_the_last_never_complete_the_backlog(stamps, empty_after, page_size, limit) -> None:
    """Contract: V12, V14 (a pagination limit must not pass for the end of the pool)."""
    pool = [_pool_entry(1000 + index, stamp) for index, stamp in enumerate(sorted(stamps, reverse=True))]
    cursor = None
    returned: set[str] = set()
    for _ in range(len(pool) + 1):
        fake = FakeFlickr(pool, empty_after_page=empty_after)
        with patch.object(flickr, "_API_PAGE_SIZE", page_size), patch.object(flickr, "RUN_LIMIT", limit):
            items, new_cursor = _walk(fake, "group/419512@N22", since=cursor)
        returned |= {item[0] for item in items}
        if new_cursor is not None:
            cursor = new_cursor
        fetched = _range(cursor)
        if fetched is not None and fetched[0] == 0:
            assert returned == {entry["id"] for entry in pool}


def test_first_run_takes_the_newest_1000_and_marks_their_range() -> None:
    """Contract: V4, V12"""
    entries = [_pool_entry(10_000 + index, 5000 - index) for index in range(1500)]
    fake = FakeFlickr(entries)
    items, new_cursor = _walk(fake, "group/419512@N22")
    assert [item[0] for item in items] == [str(10_000 + index) for index in range(1000)]
    assert new_cursor == "4001:5000"
    # The third page is read to see that the 1000th photo ends its second.
    assert [request["page"] for request in fake.api_page_requests()] == ["1", "2", "3"]


def test_more_new_photos_than_the_budget_are_taken_from_the_oldest_up() -> None:
    """Contract: V4, V13 (2500 new photos: the 1000 just above the old mark;
    the backlog below waits while new photos are left)."""
    old = [_pool_entry(1, 100), _pool_entry(0, 50)]
    new = [_pool_entry(10_000 + index, 3600 - index) for index in range(2500)]
    items, new_cursor = _walk(FakeFlickr(new + old), "group/419512@N22", since="100:100")
    taken = [item[0] for item in items]
    assert taken[0] == "1"  # the mark's own second comes along, outside the budget
    assert taken[1:] == [str(10_000 + index) for index in range(2499, 1499, -1)]
    assert new_cursor == "100:2100"


def test_budget_left_after_new_photos_goes_to_the_backlog_top_down() -> None:
    """Contract: V4"""
    pool = (
        [_pool_entry(30, 300), _pool_entry(29, 290)]
        + [_pool_entry(20, 200)]
        + [_pool_entry(10 - index, 99 - index) for index in range(10)]
    )
    with patch.object(flickr, "RUN_LIMIT", 5):
        items, new_cursor = _walk(FakeFlickr(pool), "group/419512@N22", since="200:200")
    assert [item[0] for item in items] == ["20", "29", "30", "10", "9", "8"]
    assert new_cursor == "97:300"


def test_reaching_the_end_marks_the_backlog_complete(log_lines) -> None:
    """Contract: V14"""
    pool = [_pool_entry(3, 300), _pool_entry(2, 200), _pool_entry(1, 100)]
    items, new_cursor = _walk(FakeFlickr(pool), "group/419512@N22", since="300:300")
    assert [item[0] for item in items] == ["3", "2", "1"]
    assert new_cursor == "0:300"
    assert log_lines[-1] == "Flickr: group/419512@N22: 0 new, 2 from the backlog (backlog complete)"


def test_after_the_end_only_new_photos_are_read() -> None:
    """Contract: V14 (the walk stops at the first photo below the mark)."""
    pool = [_pool_entry(9, 900), _pool_entry(8, 800)] + [
        _pool_entry(100 + index, 700 - index) for index in range(20)
    ]
    fake = FakeFlickr(pool)
    with patch.object(flickr, "_API_PAGE_SIZE", 3):
        items, new_cursor = _walk(fake, "group/419512@N22", since="0:700")
    assert [item[0] for item in items] == ["100", "8", "9"]
    assert new_cursor == "0:900"
    assert [request["page"] for request in fake.api_page_requests()] == ["1", "2"]


def test_photo_added_in_the_marks_second_is_still_fetched() -> None:
    """Contract: V12, V13 (the upper boundary is inclusive)."""
    pool = [_pool_entry(2, 100), _pool_entry(1, 100), _pool_entry(0, 99)]
    with patch.object(flickr, "RUN_LIMIT", 1):
        items, new_cursor = _walk(FakeFlickr(pool), "group/419512@N22", since="100:100")
    assert [item[0] for item in items] == ["2", "1", "0"]
    assert new_cursor == "0:100"


def test_a_second_is_never_split_at_the_budget() -> None:
    """Contract: V12 (the boundary falls between two seconds)."""
    pool = [_pool_entry(5, 50), _pool_entry(4, 40), _pool_entry(3, 40), _pool_entry(2, 30)]
    with patch.object(flickr, "RUN_LIMIT", 2):
        items, new_cursor = _walk(FakeFlickr(pool), "group/419512@N22")
    assert [item[0] for item in items] == ["5"]
    assert new_cursor == "50:50"


def test_one_second_larger_than_the_budget_is_taken_whole() -> None:
    """Contract: V4 bound plus one oversized second (see the module docstring)."""
    pool = [_pool_entry(3, 10), _pool_entry(2, 10), _pool_entry(1, 10), _pool_entry(0, 9)]
    with patch.object(flickr, "RUN_LIMIT", 2):
        items, new_cursor = _walk(FakeFlickr(pool), "group/419512@N22")
    assert [item[0] for item in items] == ["3", "2", "1"]
    assert new_cursor == "10:10"


def test_the_walk_stops_once_the_second_at_the_budget_is_complete() -> None:
    """Contract: V4 (no more pages than needed to see where the budget ends)."""
    pool = [_pool_entry(10 + index, 50 - index) for index in range(5)]
    fake = FakeFlickr(pool)
    with patch.object(flickr, "_API_PAGE_SIZE", 1), patch.object(flickr, "RUN_LIMIT", 1):
        items, new_cursor = _walk(fake, "group/419512@N22")
    assert [item[0] for item in items] == ["10"]
    assert new_cursor == "50:50"
    assert [request["page"] for request in fake.api_page_requests()] == ["1", "2"]


def test_a_larger_budget_also_stops_right_after_its_last_second() -> None:
    """Contract: V4"""
    pool = [_pool_entry(10 + index, 50 - index) for index in range(6)]
    fake = FakeFlickr(pool)
    with patch.object(flickr, "_API_PAGE_SIZE", 1), patch.object(flickr, "RUN_LIMIT", 3):
        items, new_cursor = _walk(fake, "group/419512@N22")
    assert [item[0] for item in items] == ["10", "11", "12"]
    assert new_cursor == "48:50"
    assert [request["page"] for request in fake.api_page_requests()] == ["1", "2", "3", "4"]


def test_a_range_down_to_the_first_second_still_completes() -> None:
    """Contract: V14 (any oldest mark above 0 leaves the backlog open)."""
    pool = [_pool_entry(6, 6), _pool_entry(5, 5), _pool_entry(3, 3)]
    items, new_cursor = _walk(FakeFlickr(pool), "group/419512@N22", since="1:5")
    assert [item[0] for item in items] == ["5", "6"]
    assert new_cursor == "0:6"


def test_a_budget_filled_by_new_photos_stops_the_walk() -> None:
    """Contract: V4 (the range below is not read when nothing is left for it)."""
    pool = [_pool_entry(30, 300), _pool_entry(20, 200)] + [
        _pool_entry(10 - index, 100 - index * 10) for index in range(6)
    ]
    fake = FakeFlickr(pool)
    with patch.object(flickr, "_API_PAGE_SIZE", 2), patch.object(flickr, "RUN_LIMIT", 2):
        items, new_cursor = _walk(fake, "group/419512@N22", since="50:100")
    assert [item[0] for item in items] == ["10", "20", "30"]
    assert new_cursor == "50:300"
    assert [request["page"] for request in fake.api_page_requests()] == ["1", "2"]


def test_an_oversized_new_second_at_the_end_of_the_pool_is_taken_whole() -> None:
    """Contract: V4 bound plus one oversized second, V13 (a run always progresses)."""
    pool = [_pool_entry(3, 20), _pool_entry(2, 20), _pool_entry(1, 20)]
    with patch.object(flickr, "RUN_LIMIT", 2):
        items, new_cursor = _walk(FakeFlickr(pool), "group/419512@N22", since="0:10")
    assert [item[0] for item in items] == ["1", "2", "3"]
    assert new_cursor == "0:20"


def test_cursor_compares_as_number_not_text() -> None:
    """Contract: V4 (risk 2: "999" > "1000" as text)."""
    pool = [_pool_entry(7, 1000), _pool_entry(6, 5)]
    items, new_cursor = _walk(FakeFlickr(pool), "group/419512@N22", since="999:999")
    assert [item[0] for item in items] == ["7", "6"]
    assert new_cursor == "0:1000"


@pytest.mark.parametrize("cursor", [None, "", "5", "abc:5", "5:abc", "9:3", "-1:5", "1:2:3"])
def test_an_unreadable_mark_starts_from_the_top(cursor) -> None:
    """Contract: V12 (no mark means nothing counts as fetched)."""
    pool = [_pool_entry(2, 20), _pool_entry(1, 10)]
    items, new_cursor = _walk(FakeFlickr(pool), "group/419512@N22", since=cursor)
    assert [item[0] for item in items] == ["2", "1"]
    assert new_cursor == "0:20"


def test_pool_follows_the_time_added_not_the_upload() -> None:
    """Contract: V4 (risk 1: an old photo newly added to the pool is new)."""
    entries = [_pool_entry(7, added=2000, uploaded=10)]
    items, new_cursor = _walk(FakeFlickr(entries), "group/419512@N22", since="0:1500")
    assert [item[0] for item in items] == ["7"]
    assert new_cursor == "0:2000"


def test_photostream_follows_the_upload_time() -> None:
    """Contract: V4"""
    entries = [_user_entry(9, 3000), _user_entry(8, 1000)]
    fake = FakeFlickr(entries)
    items, new_cursor = _walk(fake, "user/16099490@N00", since="0:2000")
    assert [item[0] for item in items] == ["9"]
    assert new_cursor == "0:3000"
    request = fake.api_page_requests()[0]
    assert request["method"] == "flickr.people.getPublicPhotos"
    assert request["user_id"] == "16099490@N00"
    assert "date_upload" in request["extras"].split(",")


def test_entry_without_a_timestamp_is_skipped() -> None:
    """Contract: V12 (a photo that cannot be placed in the range is not taken)."""
    entries = [{"id": "5", "url_l": f"{STATIC}/5_s_b.jpg"}, _pool_entry(4, 40)]
    items, new_cursor = _walk(FakeFlickr(entries), "group/419512@N22")
    assert [item[0] for item in items] == ["4"]
    assert new_cursor == "0:40"


def test_group_request_names_the_pool_method_and_sizes() -> None:
    """Contract: V4, V6 (the sizes are requested as extras)."""
    fake = FakeFlickr([_pool_entry(1, 1)])
    _walk(fake, "group/419512@N22")
    request = fake.api_page_requests()[0]
    assert request["method"] == "flickr.groups.pools.getPhotos"
    assert request["group_id"] == "419512@N22"
    assert request["api_key"] == API_KEY
    assert request["extras"].split(",")[:3] == ["url_l", "url_c", "url_z"]
    assert request["per_page"] == "500"


# ── V6: size choice ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"url_l": "L", "url_c": "C", "url_z": "Z"}, "L"),
        ({"url_c": "C", "url_z": "Z"}, "C"),
        ({"url_z": "Z"}, "Z"),
        ({"url_l": "https://example.com/L.jpg", "url_c": "C"}, "C"),
    ],
)
def test_api_takes_the_largest_allowed_size(fields, expected) -> None:
    """Contract: V6, V7"""
    entry = {"id": "5", "dateadded": "1"}
    for field, value in fields.items():
        entry[field] = value if value.startswith("https://") else f"{STATIC}/5_{value}.jpg"
    items, _ = _walk(FakeFlickr([entry]), "group/419512@N22")
    assert items == [("5", f"{STATIC}/5_{expected}.jpg", None)]


def test_entry_without_any_allowed_size_is_dropped() -> None:
    """Contract: V7"""
    entry = {"id": "5", "dateadded": "1", "url_l": "http://169.254.169.254/x.jpg"}
    items, _ = _walk(FakeFlickr([entry]), "group/419512@N22")
    assert items == []


@pytest.mark.parametrize("photo_id", ["../../etc/passwd", "12a", "", "1/2"])
def test_entry_with_a_non_numeric_id_is_dropped(photo_id) -> None:
    """Contract: V7 (the id names the file; a hostile one must not reach the path)."""
    entry = {"id": photo_id, "dateadded": "1", "url_l": f"{STATIC}/1_s_b.jpg"}
    items, _ = _walk(FakeFlickr([entry]), "group/419512@N22")
    assert items == []


# ── V3: feed without key ──────────────────────────────────────────────────────


def _feed_entry(photo_id: int, user: str = "alexgee") -> dict:
    return {
        "link": f"https://www.flickr.com/photos/{user}/{photo_id}/in/pool-419512@N22",
        "media": {"m": f"{STATIC}/{photo_id}_6aed83a1f0_m.jpg"},
    }


def test_feed_lists_the_newest_photos_large_with_small_fallback_and_no_cursor() -> None:
    """Contract: V3, V6"""
    fake = FakeFlickr(feed_items=[_feed_entry(100 + n) for n in range(20)])
    items, new_cursor = _walk(fake, "group/419512@N22", since="555", api_key=None)
    assert new_cursor is None
    assert len(items) == 20
    assert items[0] == (
        "100",
        f"{STATIC}/100_6aed83a1f0_b.jpg",
        f"{STATIC}/100_6aed83a1f0_m.jpg",
    )
    assert len(fake.urls) == 1
    query = _query(fake.urls[0])
    assert urlsplit(fake.urls[0]).path == "/services/feeds/groups_pool.gne"
    assert query["id"] == "419512@N22"


def test_user_feed_uses_the_public_photos_feed() -> None:
    """Contract: V3"""
    fake = FakeFlickr(feed_items=[_feed_entry(1)])
    _walk(fake, "user/16099490@N00", api_key=None)
    assert urlsplit(fake.urls[0]).path == "/services/feeds/photos_public.gne"
    assert _query(fake.urls[0])["id"] == "16099490@N00"


def test_feed_entries_on_foreign_hosts_or_without_photo_link_are_dropped() -> None:
    """Contract: V7"""
    foreign = {"link": "https://www.flickr.com/photos/a/1/", "media": {"m": "https://evil.io/1_m.jpg"}}
    no_link = {"link": "https://www.flickr.com/groups/x/", "media": {"m": f"{STATIC}/2_s_m.jpg"}}
    fake = FakeFlickr(feed_items=[foreign, no_link, "junk", _feed_entry(3)])
    items, _ = _walk(fake, "group/419512@N22", api_key=None)
    assert [item[0] for item in items] == ["3"]


def test_feed_entry_without_m_suffix_has_no_fallback() -> None:
    """Contract: V6"""
    entry = {"link": "https://www.flickr.com/photos/a/4/", "media": {"m": f"{STATIC}/4_s.jpg"}}
    items, _ = _walk(FakeFlickr(feed_items=[entry]), "group/419512@N22", api_key=None)
    assert items == [("4", f"{STATIC}/4_s.jpg", None)]


# ── V11: aliases ──────────────────────────────────────────────────────────────


def test_alias_is_resolved_with_key() -> None:
    """Contract: V11"""
    fake = FakeFlickr([_pool_entry(1, 1)], lookup={"controlrooms": "419512@N22"})
    items, _ = _walk(fake, "group/controlrooms")
    assert [item[0] for item in items] == ["1"]
    assert _query(fake.urls[0])["method"] == "flickr.urls.lookupGroup"
    assert fake.api_page_requests()[0]["group_id"] == "419512@N22"


def test_user_alias_is_resolved_through_lookup_user() -> None:
    """Contract: V11"""
    fake = FakeFlickr([_user_entry(1, 1)], lookup={"alexgee": "16099490@N00"})
    _walk(fake, "user/alexgee")
    assert _query(fake.urls[0])["method"] == "flickr.urls.lookupUser"
    assert fake.api_page_requests()[0]["user_id"] == "16099490@N00"


def test_unknown_alias_with_key_fetches_nothing() -> None:
    """Contract: V8, V11"""
    fake = FakeFlickr([_pool_entry(1, 1)])
    assert _walk(fake, "group/nosuchgroup") == ([], None)
    assert fake.api_page_requests() == []


def test_alias_without_key_is_skipped_with_a_hint(log_lines) -> None:
    """Contract: V11"""
    fake = FakeFlickr(feed_items=[_feed_entry(1)])
    assert _walk(fake, "group/controlrooms", api_key=None) == ([], None)
    assert fake.urls == []
    assert (
        "Flickr: group/controlrooms needs FLICKR_API_KEY or the numeric ID (e.g. group/419512@N22)"
        in log_lines
    )


@pytest.mark.parametrize("answer", [{"stat": "ok"}, {"stat": "ok", "group": {"id": "not-an-nsid"}}])
def test_lookup_without_a_usable_id_fetches_nothing(answer, log_lines) -> None:
    """Contract: V8, V11"""
    with patch.object(flickr, "_get_json", return_value=answer) as mock_get:
        result = flickr.iter_image_items("group/controlrooms", api_key=API_KEY, rate_limit_sec=0)
    assert result == ([], None)
    assert mock_get.call_count == 1
    assert "Flickr: could not resolve group/controlrooms to an id" in log_lines


# ── V8 / V9: failures ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "answer",
    [
        {"stat": "ok"},
        {"stat": "ok", "photos": "junk"},
        {"stat": "ok", "photos": {"photo": "junk", "pages": 3}},
    ],
)
def test_malformed_api_page_ends_the_walk(answer) -> None:
    """Contract: V8, V14 (a broken page is a failure, never the end of the pool)."""
    with patch.object(flickr, "_get_json", return_value=answer) as mock_get:
        result = flickr.iter_image_items("group/419512@N22", api_key=API_KEY, rate_limit_sec=0)
    assert result == ([], None)
    assert mock_get.call_count == 1


@pytest.mark.parametrize("pages", [0, 1])
def test_an_empty_pool_is_complete(pages) -> None:
    """Contract: V14 (an empty last page is the real end)."""
    answer = {"stat": "ok", "photos": {"photo": [], "pages": pages}}
    with patch.object(flickr, "_get_json", return_value=answer) as mock_get:
        result = flickr.iter_image_items("group/419512@N22", api_key=API_KEY, rate_limit_sec=0)
    assert result == ([], "0:0")
    assert mock_get.call_count == 1


@pytest.mark.parametrize(
    ("photos", "reason"),
    [
        ({"photo": [], "pages": 3}, "Flickr answered page 1 of 3 empty"),
        ({"photo": [], "pages": 2}, "Flickr answered page 1 of 2 empty"),
        ({"photo": [{"id": "1", "dateadded": "5"}]}, "Flickr answered a page without a page count"),
        ({"photo": "junk", "pages": 3}, "Flickr answered a page without a photo list"),
    ],
)
def test_an_empty_or_unnumbered_page_is_a_failure_not_the_end(photos, reason, log_lines) -> None:
    """Contract: V8, V14 (a cut-off answer never completes the backlog)."""
    answer = {"stat": "ok", "photos": photos}
    with patch.object(flickr, "_get_json", return_value=answer):
        result = flickr.iter_image_items("group/419512@N22", api_key=API_KEY, rate_limit_sec=0)
    assert result == ([], None)
    assert log_lines[0] == f"Flickr: page 1 of group/419512@N22 failed: {reason}"


def test_junk_entries_in_an_api_page_are_skipped() -> None:
    """Contract: V8"""
    items, new_cursor = _walk(FakeFlickr(["junk", None, _pool_entry(3, 30)]), "group/419512@N22")
    assert [item[0] for item in items] == ["3"]
    assert new_cursor == "0:30"


@pytest.mark.parametrize(
    "answer",
    [{}, {"items": "junk"}, {"items": [{"link": 5, "media": {}}, {"link": "x", "media": "y"}]}],
)
def test_malformed_feed_gives_nothing(answer) -> None:
    """Contract: V8"""
    with patch.object(flickr, "_get_json", return_value=answer):
        assert flickr.iter_image_items("group/419512@N22", rate_limit_sec=0) == ([], None)


class _JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(("body", "ok"), [(b'{"stat": "ok"}', True), (b"[1, 2]", False)])
def test_get_json_accepts_only_an_object(body, ok) -> None:
    """Contract: V8 (an unexpected shape is a broken answer, not a crash later)."""
    with patch.object(flickr, "urlopen", return_value=_JsonResponse(body)) as mock_open:
        if ok:
            assert flickr._get_json("https://www.flickr.com/x") == {"stat": "ok"}
        else:
            with pytest.raises(ValueError):
                flickr._get_json("https://www.flickr.com/x")
    request = mock_open.call_args[0][0]
    assert request.get_header("User-agent") == "Janulon/1.0"


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    yield lines
    logger.remove(sink_id)


@pytest.mark.parametrize(
    "error",
    [
        HTTPError(f"https://www.flickr.com/services/rest/?api_key={API_KEY}", 500, "boom", {}, None),
        URLError("Name or service not known"),
        TimeoutError("read timed out"),
        json.JSONDecodeError("Expecting value", "", 0),
        ValueError("not an object"),
        IncompleteRead(b"partial"),
    ],
)
def test_network_and_parse_failures_end_the_source_quietly(error, log_lines) -> None:
    """Contract: V8, V9"""
    with patch.object(flickr, "_get_json", side_effect=error):
        api_result = flickr.iter_image_items("group/419512@N22", api_key=API_KEY, rate_limit_sec=0)
        feed_result = flickr.iter_image_items("group/419512@N22", api_key=None, rate_limit_sec=0)
    assert api_result == ([], None)
    assert feed_result == ([], None)
    assert log_lines
    assert not any(API_KEY in line for line in log_lines)


def test_api_error_answer_ends_the_source(log_lines) -> None:
    """Contract: V8, V9 (bad key)"""
    answer = {"stat": "fail", "code": 100, "message": "Invalid API Key (Key has invalid format)"}
    with patch.object(flickr, "_get_json", return_value=answer):
        result = flickr.iter_image_items("group/419512@N22", api_key=API_KEY, rate_limit_sec=0)
    assert result == ([], None)
    assert any("Invalid API Key" in line for line in log_lines)
    assert not any(API_KEY in line for line in log_lines)


def test_failure_in_the_backlog_keeps_what_joins_the_range() -> None:
    """Contract: V5, V8, V12 (the last second read may be cut off and is dropped;
    the backlog is never marked complete on a failure)."""
    entries = [_pool_entry(100 + index, 50 - index // 2) for index in range(10)]
    fake = FakeFlickr(entries, fail_on_page=2)
    with patch.object(flickr, "_API_PAGE_SIZE", 3):
        items, new_cursor = _walk(fake, "group/419512@N22")
    # Page 1 held stamps 50, 50, 49; the second 49 continues on the failed page.
    assert [item[0] for item in items] == ["100", "101"]
    assert new_cursor == "50:50"


def test_failure_before_all_new_photos_are_read_takes_nothing() -> None:
    """Contract: V5, V8 (the oldest new ones are unknown, so nothing is taken)."""
    entries = [_pool_entry(100 + index, 90 - index) for index in range(10)] + [_pool_entry(1, 10)]
    fake = FakeFlickr(entries, fail_on_page=2)
    with patch.object(flickr, "_API_PAGE_SIZE", 4):
        assert _walk(fake, "group/419512@N22", since="0:10") == ([], None)


def test_invalid_stored_name_is_never_requested(log_lines) -> None:
    """Contract: V2 (a row that predates validation is re-checked)."""
    fake = FakeFlickr()
    assert _walk(fake, "group/../../x") == ([], None)
    assert _walk(fake, "https://www.flickr.com/groups/419512@N22/") == ([], None)
    assert fake.urls == []


def test_the_key_never_reaches_the_returned_data() -> None:
    """Contract: V9"""
    items, new_cursor = _walk(FakeFlickr([_pool_entry(1, 1)]), "group/419512@N22")
    assert API_KEY not in repr((items, new_cursor))


# ── downloads ─────────────────────────────────────────────────────────────────


class _FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, final_url: str):
        super().__init__(data)
        self._final_url = final_url

    def geturl(self) -> str:
        return self._final_url


class FakeImageServer:
    """Answers the image opener: png for known URLs, 410 for the rest."""

    def __init__(self, ok_urls=(), redirects=None, broken=()):
        self.ok_urls = set(ok_urls)
        self.redirects = redirects or {}
        self.broken = set(broken)
        self.requested: list[str] = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.requested.append(url)
        if url in self.redirects:
            return _FakeResponse(minimal_png_bytes(), self.redirects[url])
        if url in self.broken:
            return _FakeResponse(b"not an image", url)
        if url in self.ok_urls:
            return _FakeResponse(minimal_png_bytes(), url)
        raise HTTPError(url, 410, "Gone", {}, None)


def _download(server: FakeImageServer, items, directory: Path, skip_dirs=None):
    with patch.object(flickr._image_opener, "open", side_effect=server):
        return flickr.download_images(
            items, directory, "group/419512@N22", rate_limit_sec=0, skip_dirs=skip_dirs
        )


def test_download_names_the_file_by_photo_and_labels_it_with_the_source() -> None:
    """Contract: V6, V10"""
    url = f"{STATIC}/1_s_b.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        written = _download(FakeImageServer([url]), [("1", url, None)], Path(tmp))
        assert written == [(Path(tmp) / "flickr_1.jpg", url, "group/419512@N22")]
        assert (Path(tmp) / "flickr_1.jpg").read_bytes() == minimal_png_bytes()


def test_large_size_failing_falls_back_to_the_small_one() -> None:
    """Contract: V6 (Flickr answers 410 for a size that does not exist)."""
    large, small = f"{STATIC}/1_s_b.jpg", f"{STATIC}/1_s_m.jpg"
    server = FakeImageServer([small])
    with tempfile.TemporaryDirectory() as tmp:
        written = _download(server, [("1", large, small)], Path(tmp))
        assert written == [(Path(tmp) / "flickr_1.jpg", small, "group/419512@N22")]
    assert server.requested == [large, small]


def test_small_size_is_not_fetched_when_the_large_one_works() -> None:
    """Contract: V6"""
    large, small = f"{STATIC}/1_s_b.jpg", f"{STATIC}/1_s_m.jpg"
    server = FakeImageServer([large, small])
    with tempfile.TemporaryDirectory() as tmp:
        _download(server, [("1", large, small)], Path(tmp))
    assert server.requested == [large]


def test_a_photo_already_on_disk_in_any_size_is_not_fetched_again() -> None:
    """Contract: V6 (risk 3b: url_l one run, url_c the next)."""
    first, second = f"{STATIC}/1_s_b.jpg", f"{STATIC}/1_s_c.jpg"
    server = FakeImageServer([first, second])
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
        _download(server, [("1", first, None)], Path(tmp))
        assert _download(server, [("1", second, None)], Path(tmp)) == []
        (Path(other) / "flickr_2.png").write_bytes(minimal_png_bytes())
        assert _download(server, [("2", f"{STATIC}/2_s_b.jpg", None)], Path(tmp), [Path(other)]) == []
        # A neighbouring id with the same prefix is a different photo.
        third = f"{STATIC}/12_s_b.jpg"
        server.ok_urls.add(third)
        assert len(_download(server, [("12", third, None)], Path(tmp))) == 1
    assert server.requested == [first, third]


def test_foreign_url_is_never_requested() -> None:
    """Contract: V7"""
    server = FakeImageServer()
    items = [("1", "http://169.254.169.254/x.jpg", "https://evil.io/x_m.jpg")]
    with tempfile.TemporaryDirectory() as tmp:
        assert _download(server, items, Path(tmp)) == []
        assert list(Path(tmp).iterdir()) == []
    assert server.requested == []


def test_response_that_ended_on_a_foreign_host_is_not_stored() -> None:
    """Contract: V7 (risk 3: the placeholder behind a redirect)."""
    url = f"{STATIC}/1_s_b.jpg"
    server = FakeImageServer(redirects={url: "https://s.yimg.com/photo_unavailable.png"})
    with tempfile.TemporaryDirectory() as tmp:
        assert _download(server, [("1", url, None)], Path(tmp)) == []
        assert list(Path(tmp).iterdir()) == []


def test_unreadable_download_is_removed_and_the_fallback_tried() -> None:
    """Contract: V6"""
    large, small = f"{STATIC}/1_s_b.jpg", f"{STATIC}/1_s_m.jpg"
    server = FakeImageServer([small], broken=[large])
    with tempfile.TemporaryDirectory() as tmp:
        written = _download(server, [("1", large, small)], Path(tmp))
        assert [entry[1] for entry in written] == [small]
        assert [p.name for p in Path(tmp).iterdir()] == ["flickr_1.jpg"]


def test_download_keeps_the_image_extension_and_defaults_to_jpg() -> None:
    """Contract: V6"""
    png, odd = f"{STATIC}/1_s_o.png", f"{STATIC}/2_s_b.webp"
    with tempfile.TemporaryDirectory() as tmp:
        written = _download(FakeImageServer([png, odd]), [("1", png, None), ("2", odd, None)], Path(tmp))
        assert sorted(entry[0].name for entry in written) == ["flickr_1.png", "flickr_2.jpg"]


def test_download_skips_a_hostile_photo_id() -> None:
    """Contract: V7"""
    url = f"{STATIC}/1_s_b.jpg"
    server = FakeImageServer([url])
    with tempfile.TemporaryDirectory() as tmp:
        assert _download(server, [("../evil", url, None)], Path(tmp)) == []
    assert server.requested == []


# ── exact messages and request details ────────────────────────────────────────


def test_api_walk_logs_the_failing_page_and_the_tally(log_lines) -> None:
    """Contract: V8 (the log says which source and page broke, and what was taken)."""
    entries = [_pool_entry(100 + index, 50 - index) for index in range(10)]
    with patch.object(flickr, "_API_PAGE_SIZE", 4):
        _walk(FakeFlickr(entries, fail_on_page=2), "group/419512@N22")
    assert log_lines == [
        "Flickr: page 2 of group/419512@N22 failed: <urlopen error connection reset>",
        "Flickr: group/419512@N22: 0 new, 3 from the backlog",
    ]


def test_walk_requests_exactly_the_pages_flickr_reports() -> None:
    """Contract: V4"""
    fake = FakeFlickr([_pool_entry(3, 3), _pool_entry(2, 2), _pool_entry(1, 1)])
    with patch.object(flickr, "_API_PAGE_SIZE", 2):
        items, _ = _walk(fake, "group/419512@N22")
    assert len(items) == 3
    assert [request["page"] for request in fake.api_page_requests()] == ["1", "2"]


def test_feed_logs_its_size_and_its_failure(log_lines) -> None:
    """Contract: V3, V8"""
    _walk(FakeFlickr(feed_items=[_feed_entry(1)]), "group/419512@N22", api_key=None)
    with patch.object(flickr, "_get_json", side_effect=URLError("down")):
        flickr.iter_image_items("user/16099490@N00", rate_limit_sec=0)
    assert log_lines == [
        "Flickr: feed of group/419512@N22 lists 1 photos",
        "Flickr: feed of user/16099490@N00 failed: <urlopen error down>",
    ]


def test_invalid_stored_name_is_logged_as_such(log_lines) -> None:
    """Contract: V2"""
    flickr.iter_image_items("group/../../x", api_key=API_KEY)
    assert log_lines == ["Flickr: 'group/../../x' is not a valid source name, skipped"]


_NOTHING_TAKEN = "Flickr: group/419512@N22: 0 new, 0 from the backlog"


@pytest.mark.parametrize(
    ("source", "answer", "expected"),
    [
        (
            "group/419512@N22",
            {"stat": "fail", "code": 100, "message": "Invalid API Key"},
            [
                "Flickr: page 1 of group/419512@N22 failed: Flickr API error 100: Invalid API Key",
                _NOTHING_TAKEN,
            ],
        ),
        (
            "group/419512@N22",
            {"stat": "fail", "code": 105},
            ["Flickr: page 1 of group/419512@N22 failed: Flickr API error 105: ", _NOTHING_TAKEN],
        ),
        (
            "group/419512@N22",
            {"stat": "fail", "code": 1, "message": "x" * 300},
            [
                "Flickr: page 1 of group/419512@N22 failed: Flickr API error 1: " + "x" * 200,
                _NOTHING_TAKEN,
            ],
        ),
        (
            "group/controlrooms",
            {"stat": "fail", "code": 1, "message": "Group not found"},
            ["Flickr: group/controlrooms failed: FlickrApiError: Flickr API error 1: Group not found"],
        ),
    ],
)
def test_api_error_is_logged_with_code_and_a_bounded_message(source, answer, expected, log_lines) -> None:
    """Contract: V8 (bad key, unknown group): the reason reaches the log, bounded."""
    with patch.object(flickr, "_get_json", return_value=answer):
        flickr.iter_image_items(source, api_key=API_KEY, rate_limit_sec=0)
    assert log_lines == expected


def test_get_json_uses_a_timeout_and_names_a_wrong_shape() -> None:
    """Contract: V8 (a hanging Flickr must not hold the scrape)."""
    with patch.object(flickr, "urlopen", return_value=_JsonResponse(b"[]")) as mock_open:
        with pytest.raises(ValueError) as raised:
            flickr._get_json("https://www.flickr.com/x")
    assert str(raised.value) == "Flickr answered with something other than a JSON object"
    assert mock_open.call_args.kwargs == {"timeout": 15}


def test_image_request_sends_the_user_agent_with_a_timeout() -> None:
    """Contract: V8"""
    url = f"{STATIC}/1_s_b.jpg"
    server = FakeImageServer([url])
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(flickr._image_opener, "open", side_effect=server) as mock_open:
            flickr.download_images([("1", url, None)], Path(tmp), "group/419512@N22", rate_limit_sec=0)
    request = mock_open.call_args.args[0]
    assert request.get_header("User-agent") == "Janulon/1.0"
    assert mock_open.call_args.kwargs == {"timeout": 30}


def test_download_goes_on_after_skips_and_failures_and_logs_the_tally(log_lines) -> None:
    """Contract: V6, V7, V8"""
    good, gone, broken = f"{STATIC}/4_s_b.jpg", f"{STATIC}/3_s_b.jpg", f"{STATIC}/5_s_b.jpg"
    server = FakeImageServer([good], broken=[broken])
    items = [
        ("../x", good, None),
        ("2", good, None),
        ("6", good, None),
        ("3", gone, None),
        ("5", broken, None),
        ("4", good, None),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "flickr_2.jpg").write_bytes(minimal_png_bytes())
        (Path(tmp) / "flickr_6.jpg").write_bytes(minimal_png_bytes())
        written = _download(server, items, Path(tmp))
        assert [entry[0].name for entry in written] == ["flickr_4.jpg"]
    assert log_lines == [
        "Flickr: download of flickr_3.jpg failed: HTTP Error 410: Gone",
        "Flickr: removed unreadable download flickr_5.jpg",
        "Flickr: downloaded 1 of 6 photos of group/419512@N22 (2 already present)",
    ]


def test_download_waits_half_a_second_per_photo_by_default() -> None:
    """Contract: V8 (a polite pace keeps Flickr from blocking the host)."""
    url = f"{STATIC}/1_s_b.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        with (
            patch.object(flickr._image_opener, "open", side_effect=FakeImageServer([url])),
            patch.object(flickr.time, "sleep") as mock_sleep,
        ):
            flickr.download_images([("1", url, None)], Path(tmp), "group/419512@N22")
    mock_sleep.assert_called_once_with(0.5)


def test_download_creates_a_missing_nested_directory() -> None:
    """Contract: V6"""
    url = f"{STATIC}/1_s_b.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "data" / "images"
        written = _download(FakeImageServer([url]), [("1", url, None)], target)
        assert [entry[0] for entry in written] == [target / "flickr_1.jpg"]


def test_allowed_redirect_keeps_urllib_refusing_unsafe_methods() -> None:
    """Contract: V7 (an allowed target still goes through urllib's own checks)."""
    handler = flickr._AllowlistRedirectHandler()
    request = Request(f"{STATIC}/1_s_b.jpg", data=b"x")
    headers = {"Location": f"{STATIC}/2_s_b.jpg"}
    fp = io.BytesIO(b"")
    with pytest.raises(HTTPError) as raised:
        handler.redirect_request(request, fp, 307, "Temporary Redirect", headers, f"{STATIC}/2_s_b.jpg")
    assert raised.value.code == 307
    assert raised.value.msg == "Temporary Redirect"
    assert raised.value.hdrs is headers
    assert raised.value.fp is fp
