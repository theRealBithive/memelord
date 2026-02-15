"""Tests for main CLI."""

import tempfile
from pathlib import Path
from unittest.mock import patch

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
                "main.fourchan.download_images", return_value=[out / "inbox" / "1.jpg"]
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
                return_value=[out / "inbox" / "blog_abc.jpg"],
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
                return_value=[out / "inbox" / "funny_abc.jpg"],
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
