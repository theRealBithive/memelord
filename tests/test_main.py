"""Tests for main CLI."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import main


def test_main_import_data_calls_import_data_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """main import-data calls db.import_data with data_dir and db, logs result."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir()
        db_path = data_dir / "janulon.db"
        with patch("main.db.import_data", return_value=(3, 1)) as import_data_mock:
            with patch(
                "sys.argv",
                [
                    "main.py",
                    "import-data",
                    "--data_dir",
                    str(data_dir),
                    "--db",
                    str(db_path),
                ],
            ):
                main()
        import_data_mock.assert_called_once_with(data_dir, db_path)
    assert "3 inserted" in caplog.text
    assert "1 skipped" in caplog.text


def test_main_cleanup_calls_cleanup_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    """main cleanup calls db.cleanup_void_files with db, logs count."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "janulon.db"
        with patch("main.db.cleanup_void_files", return_value=5) as cleanup_mock:
            with patch(
                "sys.argv",
                ["main.py", "cleanup", "--db", str(db_path)],
            ):
                main()
        cleanup_mock.assert_called_once_with(db_path, Path("data"))
    assert "5" in caplog.text and "void" in caplog.text
    assert "Cleanup" in caplog.text or "removed" in caplog.text.lower()


def test_main_post_saves_status_id_and_engagement_then_refreshes_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """main post saves mastodon status ID and engagement from response, then refreshes all posted."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir()
        db_path = data_dir / "janulon.db"
        config_path = data_dir / "config.toml"
        config_path.write_text(
            '[mastodon]\nbase_url = "https://example.com"\naccess_token = "token"\n'
        )
        mock_row = MagicMock()
        mock_row.content_hash = "abc123"
        mock_row.file_path = "corpus/img.jpg"
        mock_row.source_label = "wg"
        status_response = {
            "id": 98765,
            "favourites_count": 1,
            "reblogs_count": 0,
            "replies_count": 0,
        }
        with patch("main.db.init_db"):
            with patch(
                "main.db.get_random_unposted_corpus_image",
                return_value=mock_row,
            ):
                with patch(
                    "main.db.resolve_file_path",
                    return_value=data_dir / "corpus" / "img.jpg",
                ):
                    with patch(
                        "main.caption.describe_for_alt",
                        return_value="alt text",
                    ):
                        with patch(
                            "main.mastodon_module.post_image",
                            return_value=status_response,
                        ):
                            with patch("main.db.update_image_engagement") as update_eng:
                                with patch(
                                    "main._refresh_posted_engagement",
                                ) as refresh_mock:
                                    with patch("main._log_top_posts_by_engagement"):
                                        with patch(
                                            "sys.argv",
                                            [
                                                "main.py",
                                                "post",
                                                "--config",
                                                str(config_path),
                                                "--db",
                                                str(db_path),
                                                "--data_dir",
                                                str(data_dir),
                                            ],
                                        ):
                                            main()
        update_eng.assert_called_once()
        assert update_eng.call_args[0][1] == 1  # favourites_count
        assert update_eng.call_args[0][2] == 0  # reblogs_count
        assert update_eng.call_args[0][3] == 0  # replies_count
        refresh_mock.assert_called_once()
        assert refresh_mock.call_args[1]["exclude_content_hash"] == "abc123"
    assert "Posted" in caplog.text


def test_main_post_boosts_when_source_label_is_pixelfed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When chosen corpus image has source_label pixelfed and source_url, post resolves URL and boosts instead of uploading."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir()
        db_path = data_dir / "janulon.db"
        config_path = data_dir / "config.toml"
        config_path.write_text(
            '[mastodon]\nbase_url = "https://example.com"\naccess_token = "token"\n'
        )
        mock_row = MagicMock()
        mock_row.content_hash = "pix123"
        mock_row.file_path = "corpus/pixelfed_0_photo.jpg"
        mock_row.source_label = "pixelfed"
        mock_row.source_url = "https://pixelfed.social/p/user/42"
        status_response = {
            "id": 111222,
            "favourites_count": 0,
            "reblogs_count": 1,
            "replies_count": 0,
        }
        with patch("main.db.init_db"):
            with patch(
                "main.db.get_random_unposted_corpus_image",
                return_value=mock_row,
            ):
                with patch(
                    "main.db.resolve_file_path",
                    return_value=data_dir / "corpus" / "pixelfed_0_photo.jpg",
                ):
                    with patch(
                        "main.mastodon_module.resolve_remote_url",
                        return_value="98765",
                    ) as resolve_mock:
                        with patch(
                            "main.mastodon_module.boost_status",
                            return_value=status_response,
                        ) as boost_mock:
                            with patch(
                                "main.mastodon_module.post_image"
                            ) as post_image_mock:
                                with patch(
                                    "main.db.update_image_engagement"
                                ) as update_eng:
                                    with patch("main._refresh_posted_engagement"):
                                        with patch("main._log_top_posts_by_engagement"):
                                            with patch(
                                                "sys.argv",
                                                [
                                                    "main.py",
                                                    "post",
                                                    "--config",
                                                    str(config_path),
                                                    "--db",
                                                    str(db_path),
                                                    "--data_dir",
                                                    str(data_dir),
                                                ],
                                            ):
                                                main()
        resolve_mock.assert_called_once()
        assert resolve_mock.call_args[0][1] == "https://pixelfed.social/p/user/42"
        boost_mock.assert_called_once()
        assert boost_mock.call_args[0][1] == "98765"
        post_image_mock.assert_not_called()
        update_eng.assert_called_once()
        assert "Posted" in caplog.text


def test_post_run_summary_posts_artefacts_message_when_mastodon_configured() -> None:
    """_post_run_summary posts text-only status with corpus/void counts when config has mastodon."""
    from main import _post_run_summary

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        config_path = Path(f.name)
    try:
        with patch(
            "main._load_config",
            return_value={
                "mastodon": {
                    "base_url": "https://mastodon.example",
                    "access_token": "token",
                }
            },
        ):
            with patch("main.mastodon_module.create_client") as create_mock:
                with patch("main.mastodon_module.post_status") as post_mock:
                    create_mock.return_value = MagicMock()
                    _post_run_summary(
                        config_path,
                        corpus_count=3,
                        void_count=7,
                        unposted_count=0,
                    )
        post_mock.assert_called_once()
        status_text = post_mock.call_args[0][1]
        assert "3" in status_text and "7" in status_text
        assert "corpus" in status_text and "void" in status_text
        assert "artefacts" in status_text.lower()
        assert "vault stands empty" in status_text
    finally:
        config_path.unlink(missing_ok=True)


def test_post_run_summary_includes_unposted_vault_when_positive() -> None:
    """_post_run_summary includes 'remain in the vault' and count when unposted_count > 0."""
    from main import _post_run_summary

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        config_path = Path(f.name)
    try:
        with patch(
            "main._load_config",
            return_value={
                "mastodon": {
                    "base_url": "https://mastodon.example",
                    "access_token": "token",
                }
            },
        ):
            with patch("main.mastodon_module.create_client") as create_mock:
                with patch("main.mastodon_module.post_status") as post_mock:
                    create_mock.return_value = MagicMock()
                    _post_run_summary(
                        config_path,
                        corpus_count=1,
                        void_count=2,
                        unposted_count=42,
                    )
        status_text = post_mock.call_args[0][1]
        assert "42" in status_text
        assert "remain in the vault" in status_text
        assert "awaiting the hour" in status_text
    finally:
        config_path.unlink(missing_ok=True)


def test_post_run_summary_skips_when_mastodon_config_missing() -> None:
    """_post_run_summary does not call post_status when mastodon base_url or token is empty."""
    from main import _post_run_summary

    config_path = Path("/nonexistent/config.toml")
    with patch(
        "main._load_config",
        return_value={"mastodon": {"base_url": "", "access_token": "token"}},
    ):
        with patch("main.mastodon_module.post_status") as post_mock:
            _post_run_summary(config_path, corpus_count=1, void_count=0)
    post_mock.assert_not_called()


def test_post_retrain_summary_posts_ominous_message_when_mastodon_configured() -> None:
    """_post_retrain_summary posts text-only status with retrain wording when config has mastodon."""
    from main import _post_retrain_summary

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        config_path = Path(f.name)
    try:
        with patch(
            "main._load_config",
            return_value={
                "mastodon": {
                    "base_url": "https://mastodon.example",
                    "access_token": "token",
                }
            },
        ):
            with patch("main.mastodon_module.create_client") as create_mock:
                with patch("main.mastodon_module.post_status") as post_mock:
                    create_mock.return_value = MagicMock()
                    _post_retrain_summary(config_path)
        post_mock.assert_called_once()
        status_text = post_mock.call_args[0][1]
        assert "discernment" in status_text or "Pondering" in status_text
    finally:
        config_path.unlink(missing_ok=True)


def test_post_retrain_summary_skips_when_mastodon_config_missing() -> None:
    """_post_retrain_summary does not call post_status when mastodon config is empty."""
    from main import _post_retrain_summary

    config_path = Path("/nonexistent/config.toml")
    with patch(
        "main._load_config",
        return_value={"mastodon": {"base_url": "https://x.com", "access_token": ""}},
    ):
        with patch("main.mastodon_module.post_status") as post_mock:
            _post_retrain_summary(config_path)
    post_mock.assert_not_called()


def test_post_sync_summary_skips_when_mastodon_config_missing() -> None:
    """_post_sync_summary does not call post_status when mastodon config is empty."""
    from main import _post_sync_summary

    config_path = Path("/nonexistent/config.toml")
    with patch(
        "main._load_config",
        return_value={"mastodon": {"base_url": "https://x.com", "access_token": ""}},
    ):
        with patch("main.mastodon_module.post_status") as post_mock:
            _post_sync_summary(config_path)
    post_mock.assert_not_called()


def test_get_schedule_from_config_returns_defaults_when_missing() -> None:
    """_get_schedule_from_config returns default intervals (in minutes) when file or [schedule] missing."""
    from main import _get_schedule_from_config

    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "missing.toml"
        out = _get_schedule_from_config(missing)
    assert out["scrape_every_minutes"] == 360
    assert out["post_every_minutes"] == 1440
    assert out["retrain_every_minutes"] == 10080  # 168h default
    assert out["sync_every_minutes"] == 2880  # 48h default


def test_get_schedule_from_config_returns_values_from_file() -> None:
    """_get_schedule_from_config returns [schedule] values as minutes when present."""
    from main import _get_schedule_from_config

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        f.write(
            b"[schedule]\n"
            b"scrape_every_hours = 2\n"
            b"post_every_hours = 12\n"
            b"retrain_every_hours = 24\n"
        )
        path = Path(f.name)
    try:
        out = _get_schedule_from_config(path)
        assert out["scrape_every_minutes"] == 120
        assert out["post_every_minutes"] == 720
        assert out["retrain_every_minutes"] == 1440
        assert out["sync_every_minutes"] == 2880
    finally:
        path.unlink(missing_ok=True)


def test_get_schedule_from_config_accepts_fractional_hours() -> None:
    """_get_schedule_from_config converts fractional hours to minutes (e.g. 0.5 -> 30)."""
    from main import _get_schedule_from_config

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        f.write(b"[schedule]\n" b"scrape_every_hours = 6\n" b"post_every_hours = 0.5\n")
        path = Path(f.name)
    try:
        out = _get_schedule_from_config(path)
        assert out["scrape_every_minutes"] == 360
        assert out["post_every_minutes"] == 30
        assert out["retrain_every_minutes"] == 10080
        assert out["sync_every_minutes"] == 2880
    finally:
        path.unlink(missing_ok=True)


def test_main_train_calls_trainer_run() -> None:
    """main train invokes trainer.run with data_dir, weights, db_path."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "corpus").mkdir()
        (data_dir / "void").mkdir()
        (data_dir / "corpus" / "a.jpg").write_bytes(b"x")
        (data_dir / "void" / "b.jpg").write_bytes(b"y")
        weights = data_dir / "w.pkl"
        db_path = data_dir / "janulon.db"
        with patch("core.trainer.run") as run_mock:
            with patch(
                "sys.argv",
                [
                    "main.py",
                    "train",
                    "--data_dir",
                    str(data_dir),
                    "--weights",
                    str(weights),
                    "--db",
                    str(db_path),
                ],
            ):
                main()
        run_mock.assert_called_once_with(
            data_dir=data_dir,
            weights_path=weights,
            db_path=db_path,
        )


def test_main_schedule_calls_run_schedule_with_paths() -> None:
    """main schedule invokes _run_schedule with config, db, weights, output, data_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config = tmp_path / "config.toml"
        config.write_text("[schedule]\nscrape_every_hours = 1\n")
        with patch("main._run_schedule") as run_schedule_mock:
            with patch(
                "sys.argv",
                [
                    "main.py",
                    "schedule",
                    "--config",
                    str(config),
                    "--db",
                    str(tmp_path / "db"),
                    "--weights",
                    str(tmp_path / "w.pkl"),
                    "--output_folder",
                    str(tmp_path / "out"),
                    "--data_dir",
                    str(tmp_path),
                ],
            ):
                main()
        run_schedule_mock.assert_called_once()
        call_kw = run_schedule_mock.call_args[1]
        assert call_kw["config_path"] == config
        assert call_kw["db_path"] == tmp_path / "db"
        assert call_kw["weights_path"] == tmp_path / "w.pkl"
        assert call_kw["output_folder"] == tmp_path / "out"
        assert call_kw["data_dir"] == tmp_path


def test_main_4chan_downloads_to_output_folder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """main --source 4chan --output_folder X downloads to X/inbox and uses skip_dirs."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pics"
        with patch(
            "main.fourchan.iter_image_urls",
            return_value=["https://i.4cdn.org/wg/1.jpg"],
        ):
            with patch(
                "main.fourchan.download_images",
                return_value=[
                    (out / "inbox" / "1.jpg", "https://i.4cdn.org/wg/1.jpg", "wg")
                ],
            ) as dl:
                with patch(
                    "sys.argv",
                    [
                        "main.py",
                        "run",
                        "--source",
                        "4chan",
                        "--board",
                        "wg",
                        "--output_folder",
                        str(out),
                        "--index_pages",
                        "1",
                        "--no_judge",
                    ],
                ):
                    main()
                dl.assert_called_once()
                call_args = dl.call_args
                assert call_args[0][0] == ["https://i.4cdn.org/wg/1.jpg"]
                assert call_args[0][1] == out / "inbox"
                assert call_args[0][2] == "wg"
                assert "skip_dirs" in call_args[1]
                # data corpus/void + output corpus/void
                assert len(call_args[1]["skip_dirs"]) == 4
    assert "Downloaded 1 images" in caplog.text
    assert "inbox" in caplog.text


def test_main_4chan_accepts_defaults() -> None:
    """main --source 4chan uses default board and output_folder."""
    with patch("main.fourchan.iter_image_urls", return_value=[]):
        with patch(
            "sys.argv",
            ["main.py", "run", "--source", "4chan", "--index_pages", "1", "--no_judge"],
        ):
            main()
        from main import fourchan

        fourchan.iter_image_urls.assert_called_once_with(board="wg", index_pages=1)


def test_main_tumblr_requires_blog(caplog: pytest.LogCaptureFixture) -> None:
    """main --source tumblr without --blog exits 1 and logs error."""
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["main.py", "run", "--source", "tumblr"]):
            main()
    assert exc_info.value.code == 1
    assert "blog" in caplog.text.lower()


def test_main_tumblr_downloads_to_output_folder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """main --source tumblr --blog X --output_folder Y downloads to Y/inbox."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tumblr_pics"
        with patch(
            "main.tumblr.iter_image_urls",
            return_value=["https://64.media.tumblr.com/abc/photo.jpg"],
        ):
            with patch(
                "main.tumblr.download_images",
                return_value=[
                    (
                        out / "inbox" / "blog_abc.jpg",
                        "https://64.media.tumblr.com/abc/photo.jpg",
                        "staff",
                    )
                ],
            ) as dl:
                with patch(
                    "sys.argv",
                    [
                        "main.py",
                        "run",
                        "--source",
                        "tumblr",
                        "--blog",
                        "staff",
                        "--output_folder",
                        str(out),
                        "--num_posts",
                        "20",
                        "--no_judge",
                    ],
                ):
                    main()
                dl.assert_called_once()
                call_args = dl.call_args
                assert call_args[0][0] == ["https://64.media.tumblr.com/abc/photo.jpg"]
                assert call_args[0][1] == out / "inbox"
                assert call_args[0][2] == "staff"
    assert "Downloaded 1 images" in caplog.text


def test_main_imgur_requires_topic(caplog: pytest.LogCaptureFixture) -> None:
    """main --source imgur without --topic exits 1 and logs error."""
    with pytest.raises(SystemExit) as exc_info:
        with patch.dict("os.environ", {"IMGUR_CLIENT_ID": "test"}, clear=False):
            with patch("sys.argv", ["main.py", "run", "--source", "imgur"]):
                main()
    assert exc_info.value.code == 1
    assert "topic" in caplog.text.lower()


def test_main_imgur_downloads_to_output_folder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """main --source imgur --topic X --output_folder Y downloads to Y/inbox."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "imgur_pics"
        with patch(
            "main.imgur.iter_image_urls",
            return_value=["https://i.imgur.com/abc.jpg"],
        ):
            with patch(
                "main.imgur.download_images",
                return_value=[
                    (
                        out / "inbox" / "funny_abc.jpg",
                        "https://i.imgur.com/abc.jpg",
                        "funny",
                    )
                ],
            ) as dl:
                with patch(
                    "sys.argv",
                    [
                        "main.py",
                        "run",
                        "--source",
                        "imgur",
                        "--topic",
                        "funny",
                        "--output_folder",
                        str(out),
                        "--no_judge",
                    ],
                ):
                    main()
                dl.assert_called_once()
                call_args = dl.call_args
                assert call_args[0][0] == ["https://i.imgur.com/abc.jpg"]
                assert call_args[0][1] == out / "inbox"
                assert call_args[0][2] == "funny"
    assert "Downloaded 1 images" in caplog.text


def test_main_reddit_exits_with_message(caplog: pytest.LogCaptureFixture) -> None:
    """main --source reddit logs not implemented and exits 1."""
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["main.py", "run", "--source", "reddit"]):
            main()
    assert exc_info.value.code == 1
    assert "not implemented" in caplog.text.lower()


def test_main_source_all_missing_config_exits(caplog: pytest.LogCaptureFixture) -> None:
    """main --source all with missing config file exits 1 and logs error."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "nonexistent.toml"
        with pytest.raises(SystemExit) as exc_info:
            with patch(
                "sys.argv",
                [
                    "main.py",
                    "run",
                    "--source",
                    "all",
                    "--config",
                    str(config_path),
                    "--no_judge",
                ],
            ):
                main()
        assert exc_info.value.code == 1
        assert "not found" in caplog.text.lower() or "config" in caplog.text.lower()


def test_main_source_all_loads_config_and_runs_sources(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """main --source all loads config and runs 4chan/tumblr/imgur for each entry."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "config.toml"
        config_path.write_text(
            """
[4chan]
boards = ["wg", "a"]
[tumblr]
blogs = ["staff"]
[imgur]
topics = ["funny"]
""",
            encoding="utf-8",
        )
        out = Path(tmp) / "out"
        with patch("main._run_4chan") as run_4chan:
            with patch("main._run_imgur") as run_imgur:
                with patch("main._run_tumblr") as run_tumblr:
                    with patch(
                        "sys.argv",
                        [
                            "main.py",
                            "run",
                            "--source",
                            "all",
                            "--config",
                            str(config_path),
                            "--output_folder",
                            str(out),
                            "--no_judge",
                        ],
                    ):
                        main()
        assert run_4chan.call_count == 2
        boards = [c[1]["board"] for c in run_4chan.call_args_list]
        assert boards == ["wg", "a"]
        assert run_imgur.call_count == 1
        assert run_imgur.call_args[1]["topic"] == "funny"
        assert run_tumblr.call_count == 1
        assert run_tumblr.call_args[1]["blog"] == "staff"


def test_remove_inbox_duplicates_by_hash_keeps_inbox_when_only_row_is_inbox() -> None:
    """Inbox files are not removed when the only matching row is location=inbox (grace period)."""
    from main import _remove_inbox_duplicates_by_hash

    from core import db
    from tests.conftest import minimal_png_bytes

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        inbox = out / "inbox"
        inbox.mkdir(parents=True)
        img = inbox / "photo.png"
        img.write_bytes(minimal_png_bytes())
        db.init_db(out / "janulon.db")
        db.insert_inbox_image(out, img, "https://example.com/1.png", "pixelfed")
        _remove_inbox_duplicates_by_hash(inbox)
        assert img.exists(), "Inbox file must not be removed when only row is inbox"


def test_remove_inbox_duplicates_by_hash_removes_when_content_in_corpus() -> None:
    """Inbox files are removed when same content already exists in corpus."""
    from main import _remove_inbox_duplicates_by_hash

    from core import db
    from tests.conftest import minimal_png_bytes

    png = minimal_png_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        inbox = out / "inbox"
        corpus = out / "corpus"
        inbox.mkdir(parents=True)
        corpus.mkdir(parents=True)
        db.init_db(out / "janulon.db")
        corpus_file = corpus / "existing.png"
        corpus_file.write_bytes(png)
        db.Image.create(
            content_hash=hashlib.sha256(png).hexdigest(),
            file_path="corpus/existing.png",
            source_url=None,
            source_label="wg",
            location="corpus",
            file_deleted=False,
        )
        dup = inbox / "duplicate.png"
        dup.write_bytes(png)
        _remove_inbox_duplicates_by_hash(inbox)
        assert not dup.exists(), "Inbox duplicate of corpus content must be removed"


def test_main_judge_and_sort_moves_to_corpus_and_void(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With weights present, main runs judge and moves inbox images to corpus/void."""
    import numpy as np

    from core import brain
    from tests.conftest import minimal_png_bytes

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        inbox = out / "inbox"
        inbox.mkdir(parents=True)
        (inbox / "test.png").write_bytes(minimal_png_bytes())
        weights = out / "weights.pkl"
        clf = __import__(
            "sklearn.linear_model", fromlist=["LogisticRegression"]
        ).LogisticRegression(max_iter=100, random_state=42)
        clf.fit(np.random.randn(2, 768), [0, 1])
        brain.save_classifier(clf, weights)

        with patch("main.brain.get_encoder"):
            with patch(
                "main.brain.encode", return_value=np.zeros((1, 768), dtype=np.float32)
            ):
                with patch("main.fourchan.iter_image_urls", return_value=[]):
                    with patch(
                        "sys.argv",
                        [
                            "main.py",
                            "run",
                            "--source",
                            "4chan",
                            "--output_folder",
                            str(out),
                            "--weights",
                            str(weights),
                            "--db",
                            str(out / "janulon.db"),
                            "--index_pages",
                            "1",
                        ],
                    ):
                        main()
        corpus_dir = out / "corpus"
        void_dir = out / "void"
        assert corpus_dir.exists()
        assert void_dir.exists()
        assert not (inbox / "test.png").exists()
        assert (corpus_dir / "test.png").exists() or (void_dir / "test.png").exists()
        assert "Judged 1 images" in caplog.text
        from core import db

        rows = list(db.Image.select())
        assert len(rows) == 1
        assert rows[0].location in ("corpus", "void")
        assert "test.png" in rows[0].file_path


def test_main_judge_preserves_source_from_inbox_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When inbox row has source_url/source_label, record_judged_image preserves them."""
    import numpy as np

    from core import brain, db
    from tests.conftest import minimal_png_bytes

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        inbox = out / "inbox"
        inbox.mkdir(parents=True)
        img_path = inbox / "wg_42.png"
        img_path.write_bytes(minimal_png_bytes())
        db.init_db(out / "janulon.db")
        db.insert_inbox_image(out, img_path, "https://i.4cdn.org/wg/42.png", "wg")
        weights = out / "weights.pkl"
        clf = __import__(
            "sklearn.linear_model", fromlist=["LogisticRegression"]
        ).LogisticRegression(max_iter=100, random_state=42)
        clf.fit(np.random.randn(2, 768), [0, 1])
        brain.save_classifier(clf, weights)

        with patch("main.brain.get_encoder"):
            with patch(
                "main.brain.encode", return_value=np.zeros((1, 768), dtype=np.float32)
            ):
                with patch("main.fourchan.iter_image_urls", return_value=[]):
                    with patch(
                        "sys.argv",
                        [
                            "main.py",
                            "run",
                            "--source",
                            "4chan",
                            "--output_folder",
                            str(out),
                            "--weights",
                            str(weights),
                            "--db",
                            str(out / "janulon.db"),
                            "--index_pages",
                            "1",
                        ],
                    ):
                        main()
        rows = list(db.Image.select())
        assert len(rows) == 1
        assert rows[0].source_url == "https://i.4cdn.org/wg/42.png"
        assert rows[0].source_label == "wg"
