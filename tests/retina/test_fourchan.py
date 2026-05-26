"""Tests for retina.fourchan."""

import tempfile
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from retina import fourchan
from tests.conftest import minimal_png_bytes


def test_fourchan_module_imports() -> None:
    """Fourchan scraper module can be imported."""
    assert fourchan is not None


def test_image_url_from_post_returns_url_when_valid() -> None:
    """Valid post with tim and image ext yields full image URL."""
    post = {"tim": 1234567890123, "ext": ".jpg"}
    assert fourchan.image_url_from_post(post, "wg") == (
        "https://i.4cdn.org/wg/1234567890123.jpg"
    )


def test_image_url_from_post_returns_none_when_filedeleted() -> None:
    """Post with filedeleted does not yield URL."""
    post = {"tim": 1, "ext": ".png", "filedeleted": 1}
    assert fourchan.image_url_from_post(post, "wg") is None


def test_image_url_from_post_returns_none_when_no_tim() -> None:
    """Post without tim does not yield URL."""
    post = {"ext": ".jpg"}
    assert fourchan.image_url_from_post(post, "wg") is None


def test_image_url_from_post_returns_none_when_non_image_ext() -> None:
    """Post with .webm does not yield URL (only static images)."""
    post = {"tim": 1, "ext": ".webm"}
    assert fourchan.image_url_from_post(post, "wg") is None


def test_image_url_from_post_accepts_png_and_gif() -> None:
    """PNG and GIF extensions are accepted."""
    assert fourchan.image_url_from_post({"tim": 1, "ext": ".png"}, "wg") == (
        "https://i.4cdn.org/wg/1.png"
    )
    assert fourchan.image_url_from_post({"tim": 2, "ext": ".gif"}, "wg") == (
        "https://i.4cdn.org/wg/2.gif"
    )


def test_get_thread_list_flattens_pages() -> None:
    """get_thread_list flattens every index page into one thread list."""
    payload = [
        {"page": 1, "threads": [{"no": 1}, {"no": 2}]},
        {"page": 2, "threads": [{"no": 3}]},
    ]
    with patch.object(fourchan, "_get_json", return_value=payload):
        threads = fourchan.get_thread_list("wg")
    assert [t["no"] for t in threads] == [1, 2, 3]


def test_iter_image_urls_fetches_full_threads() -> None:
    """iter_image_urls expands every live thread into its full post list."""
    thread_list = [{"no": 1, "last_modified": 10}, {"no": 2, "last_modified": 20}]
    threads = {
        1: {"posts": [{"tim": 100, "ext": ".jpg"}, {"tim": 101, "ext": ".png"}]},
        2: {
            "posts": [
                {"tim": 102, "ext": ".jpg", "filedeleted": 1},
                {"tim": 103, "ext": ".gif"},
            ]
        },
    }
    with (
        patch.object(fourchan, "get_thread_list", return_value=thread_list),
        patch.object(fourchan, "get_thread", side_effect=lambda board, no: threads[no]),
    ):
        urls, cursor = fourchan.iter_image_urls(board="wg", rate_limit_sec=0)
    # Every post is read, not just the OP + tail; filedeleted is skipped.
    assert urls == [
        "https://i.4cdn.org/wg/100.jpg",
        "https://i.4cdn.org/wg/101.png",
        "https://i.4cdn.org/wg/103.gif",
    ]
    # Cursor advances to the newest fully-fetched thread.
    assert cursor == 20


def test_iter_image_urls_deduplicates_across_threads() -> None:
    """The same image reposted in two threads appears only once."""
    thread_list = [{"no": 1, "last_modified": 10}, {"no": 2, "last_modified": 20}]
    threads = {
        1: {"posts": [{"tim": 42, "ext": ".jpg"}]},
        2: {"posts": [{"tim": 42, "ext": ".jpg"}]},
    }
    with (
        patch.object(fourchan, "get_thread_list", return_value=thread_list),
        patch.object(fourchan, "get_thread", side_effect=lambda board, no: threads[no]),
    ):
        urls, _ = fourchan.iter_image_urls(board="wg", rate_limit_sec=0)
    assert urls == ["https://i.4cdn.org/wg/42.jpg"]


def test_iter_image_urls_skips_unchanged_threads() -> None:
    """Threads with last_modified <= since_modified are not re-fetched."""
    thread_list = [
        {"no": 1, "last_modified": 100},  # unchanged since cursor
        {"no": 2, "last_modified": 200},  # changed
    ]
    fetched: list[int] = []

    def fake_get_thread(board: str, no: int) -> dict:
        fetched.append(no)
        return {"posts": [{"tim": no, "ext": ".jpg"}]}

    with (
        patch.object(fourchan, "get_thread_list", return_value=thread_list),
        patch.object(fourchan, "get_thread", side_effect=fake_get_thread),
    ):
        urls, cursor = fourchan.iter_image_urls(
            board="wg", since_modified=100, rate_limit_sec=0
        )
    assert fetched == [2]  # only the changed thread hit the network
    assert urls == ["https://i.4cdn.org/wg/2.jpg"]
    assert cursor == 200


def test_iter_image_urls_skips_failed_thread() -> None:
    """A thread that 404s (pruned mid-scan) is skipped without aborting."""
    thread_list = [{"no": 1, "last_modified": 10}, {"no": 2, "last_modified": 20}]

    def fake_get_thread(board: str, no: int) -> dict:
        if no == 1:
            raise HTTPError("url", 404, "gone", {}, None)  # type: ignore[arg-type]
        return {"posts": [{"tim": 50, "ext": ".jpg"}]}

    with (
        patch.object(fourchan, "get_thread_list", return_value=thread_list),
        patch.object(fourchan, "get_thread", side_effect=fake_get_thread),
    ):
        urls, _ = fourchan.iter_image_urls(board="wg", rate_limit_sec=0)
    assert urls == ["https://i.4cdn.org/wg/50.jpg"]


def test_iter_image_urls_skips_thread_read_timeout() -> None:
    """A socket read timeout (TimeoutError/OSError, not URLError) skips the thread.

    Read timeouts on resp.read() raise socket.timeout/TimeoutError, which is an
    OSError and is NOT wrapped in URLError — the original (HTTPError, URLError)
    catch would have let it abort the whole scrape.
    """
    thread_list = [{"no": 1, "last_modified": 10}, {"no": 2, "last_modified": 20}]

    def fake_get_thread(board: str, no: int) -> dict:
        if no == 1:
            raise TimeoutError("timed out")
        return {"posts": [{"tim": 50, "ext": ".jpg"}]}

    with (
        patch.object(fourchan, "get_thread_list", return_value=thread_list),
        patch.object(fourchan, "get_thread", side_effect=fake_get_thread),
    ):
        urls, _ = fourchan.iter_image_urls(board="wg", rate_limit_sec=0)
    assert urls == ["https://i.4cdn.org/wg/50.jpg"]


def test_iter_image_urls_cursor_freezes_on_mid_range_error() -> None:
    """A failed fetch between two successes freezes the cursor at the last success.

    Watermark semantics: advancing the cursor to max(successful last_modified)
    would jump past the failed thread (lm=100), permanently skipping it next run.
    Freezing at the pre-error success (lm=80) makes both the failed thread and
    the later success eligible for refetch — wasteful but never lossy.
    """
    thread_list = [
        {"no": 1, "last_modified": 80},  # ok
        {"no": 2, "last_modified": 100},  # errors
        {"no": 3, "last_modified": 120},  # ok, but after the error
    ]

    def fake_get_thread(board: str, no: int) -> dict:
        if no == 2:
            raise TimeoutError("timed out")
        return {"posts": [{"tim": no, "ext": ".jpg"}]}

    with (
        patch.object(fourchan, "get_thread_list", return_value=thread_list),
        patch.object(fourchan, "get_thread", side_effect=fake_get_thread),
    ):
        urls, cursor = fourchan.iter_image_urls(board="wg", rate_limit_sec=0)
    # We still capture what we can from threads 1 and 3.
    assert urls == ["https://i.4cdn.org/wg/1.jpg", "https://i.4cdn.org/wg/3.jpg"]
    # Cursor frozen at the last success before the error, not at 120.
    assert cursor == 80


def test_iter_image_urls_max_threads_takes_oldest_first() -> None:
    """A max_threads cap drains the oldest changed threads first across runs."""
    thread_list = [
        {"no": 1, "last_modified": 30},
        {"no": 2, "last_modified": 10},
        {"no": 3, "last_modified": 20},
    ]
    fetched: list[int] = []

    def fake_get_thread(board: str, no: int) -> dict:
        fetched.append(no)
        return {"posts": [{"tim": no, "ext": ".jpg"}]}

    with (
        patch.object(fourchan, "get_thread_list", return_value=thread_list),
        patch.object(fourchan, "get_thread", side_effect=fake_get_thread),
    ):
        urls, cursor = fourchan.iter_image_urls(
            board="wg", max_threads=2, rate_limit_sec=0
        )
    # Sorted ascending by last_modified, capped to 2 → threads 2 (lm10), 3 (lm20).
    assert fetched == [2, 3]
    assert urls == ["https://i.4cdn.org/wg/2.jpg", "https://i.4cdn.org/wg/3.jpg"]
    # Cursor stops at the newest fetched; thread 1 (lm30) is left for next run.
    assert cursor == 20


def test_iter_image_urls_thread_list_failure_preserves_cursor() -> None:
    """A thread-list fetch failure yields [] and returns the cursor unchanged."""
    with patch.object(fourchan, "get_thread_list", side_effect=TimeoutError("nope")):
        urls, cursor = fourchan.iter_image_urls(
            board="wg", since_modified=500, rate_limit_sec=0
        )
    assert urls == []
    assert cursor == 500


def test_iter_image_urls_rate_limits_every_request() -> None:
    """One sleep precedes the thread list and one precedes each thread fetch."""
    thread_list = [{"no": 1, "last_modified": 10}, {"no": 2, "last_modified": 20}]
    with (
        patch.object(fourchan, "get_thread_list", return_value=thread_list),
        patch.object(fourchan, "get_thread", return_value={"posts": []}),
        patch("retina.fourchan.time.sleep") as mock_sleep,
    ):
        fourchan.iter_image_urls(board="wg", rate_limit_sec=1.0)
    # 1 for threads.json + 1 per thread = never more than one request/second.
    assert mock_sleep.call_count == 3


def test_download_images_writes_files_with_board_prefix() -> None:
    """download_images writes files named {board}_{tim}{ext} and returns (path, url, label)."""
    content = minimal_png_bytes()
    url = "https://i.4cdn.org/wg/999.png"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = content
            result = fourchan.download_images([url], out, "wg", rate_limit_sec=0)
        assert len(result) == 1
        path, source_url, source_label = result[0]
        assert path == out / "wg_999.png"
        assert path.read_bytes() == content
        assert source_url == url
        assert source_label == "wg"


def test_download_images_creates_directory() -> None:
    """download_images creates output_dir if it does not exist."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "dir"
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = minimal_png_bytes()
            fourchan.download_images(
                ["https://i.4cdn.org/wg/1.jpg"], out, "wg", rate_limit_sec=0
            )
        assert (out / "wg_1.jpg").exists()


def test_download_images_skips_if_already_in_output() -> None:
    """download_images skips URL when file already exists in output_dir."""
    url = "https://i.4cdn.org/wg/42.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "wg_42.jpg").write_bytes(b"already here")
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            paths = fourchan.download_images([url], out, "wg", rate_limit_sec=0)
        assert len(paths) == 0
        mock_urlopen.assert_not_called()
        assert (out / "wg_42.jpg").read_bytes() == b"already here"


def test_download_images_skips_if_already_in_skip_dirs() -> None:
    """download_images skips URL when file exists in one of skip_dirs (e.g. corpus)."""
    url = "https://i.4cdn.org/wg/99.png"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        (corpus / "wg_99.png").write_bytes(b"in corpus")
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            paths = fourchan.download_images(
                [url], out, "wg", skip_dirs=[corpus], rate_limit_sec=0
            )
        assert len(paths) == 0
        mock_urlopen.assert_not_called()
        assert not (out / "wg_99.png").exists()


def test_download_images_skips_if_path_in_skip_paths() -> None:
    """download_images skips URL when destination path is in skip_paths (e.g. from DB)."""
    url = "https://i.4cdn.org/wg/77.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        dest = (out / "wg_77.jpg").resolve()
        skip_paths = {str(dest)}
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            paths = fourchan.download_images(
                [url], out, "wg", skip_paths=skip_paths, rate_limit_sec=0
            )
        assert len(paths) == 0
        mock_urlopen.assert_not_called()
        assert not (out / "wg_77.jpg").exists()


def test_download_images_removes_corrupted_download() -> None:
    """download_images removes file and does not return it when content is not a valid image."""
    url = "https://i.4cdn.org/wg/123.png"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with patch("retina.fourchan.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = b"corrupted or not an image"
            paths = fourchan.download_images([url], out, "wg", rate_limit_sec=0)
        assert len(paths) == 0
        assert not (out / "wg_123.png").exists()
