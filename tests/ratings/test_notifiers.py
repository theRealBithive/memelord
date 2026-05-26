"""Tests for Mattermost and Signal notification clients."""

import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image as PilImage

from ratings import notifiers


def test_encode_signal_attachment_uses_data_uri_with_filename() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = __import__("pathlib").Path(tmp) / "photo.jpg"
        PilImage.new("RGB", (4, 4), color=(255, 0, 0)).save(path)

        encoded = notifiers._encode_signal_attachment(path)

    assert encoded.startswith("data:image/jpeg;filename=photo.jpg;base64,")
    assert len(encoded.split(",", 1)[1]) > 0


def test_send_to_signal_posts_file_attachment(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = __import__("pathlib").Path(tmp) / "photo.png"
        PilImage.new("RGB", (4, 4)).save(path)

        captured: dict = {}

        def fake_post(url: str, json: dict, timeout: int) -> MagicMock:
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            response = MagicMock()
            response.raise_for_status = MagicMock()
            return response

        monkeypatch.setattr(notifiers.requests, "post", fake_post)

        ch = SimpleNamespace(
            signal_api_url="http://signal.local:8080",
            signal_sender="+15550001",
            signal_recipients="+15550002",
            signal_message_prefix="",
        )
        notifiers.send_to_signal(ch, path, "imgur/memes")

    assert captured["url"] == "http://signal.local:8080/v2/send"
    assert captured["json"]["message"] == "imgur/memes"
    assert captured["json"]["recipients"] == ["+15550002"]
    assert captured["json"]["base64_attachments"][0].startswith("data:image/png;")
    assert captured["timeout"] == 60


def test_send_to_mattermost_uploads_then_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mattermost send does a file upload then a post referencing the returned file_id."""
    with tempfile.TemporaryDirectory() as tmp:
        path = __import__("pathlib").Path(tmp) / "photo.jpg"
        PilImage.new("RGB", (4, 4), color=(0, 128, 255)).save(path)

        calls: list[dict] = []

        def fake_post(url: str, **kwargs) -> MagicMock:
            call = {"url": url, **kwargs}
            calls.append(call)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            # First call is file upload — return a file_id.
            if "/files" in url:
                resp.json.return_value = {"file_infos": [{"id": "abc123"}]}
            return resp

        monkeypatch.setattr(notifiers.requests, "post", fake_post)

        ch = SimpleNamespace(
            mm_base_url="https://chat.example.com",
            mm_token="tok_xyz",
            mm_channel_id="chan1",
            mm_message_prefix="look at this",
        )
        notifiers.send_to_mattermost(ch, path, "pixelfed/art")

    assert len(calls) == 2
    assert calls[0]["url"] == "https://chat.example.com/api/v4/files"
    assert calls[0]["data"] == {"channel_id": "chan1"}
    assert calls[1]["url"] == "https://chat.example.com/api/v4/posts"
    assert calls[1]["json"]["file_ids"] == ["abc123"]
    assert calls[1]["json"]["message"] == "look at this"


def test_send_to_mattermost_converts_webp_to_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WebP uploads are transcoded to JPEG so Mattermost mobile can preview them."""
    with tempfile.TemporaryDirectory() as tmp:
        path = __import__("pathlib").Path(tmp) / "meme.webp"
        PilImage.new("RGB", (8, 8), color=(10, 20, 30)).save(path, "WEBP")

        upload: dict = {}

        def fake_post(url: str, **kwargs) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/files" in url:
                name, fh = kwargs["files"]["files"]
                upload["name"] = name
                upload["data"] = fh.read()
                resp.json.return_value = {"file_infos": [{"id": "file1"}]}
            return resp

        monkeypatch.setattr(notifiers.requests, "post", fake_post)

        ch = SimpleNamespace(
            mm_base_url="https://chat.example.com",
            mm_token="tok",
            mm_channel_id="chan1",
            mm_message_prefix="",
        )
        notifiers.send_to_mattermost(ch, path, "")

    assert upload["name"] == "meme.jpg"
    assert upload["data"].startswith(b"\xff\xd8\xff")


def test_mattermost_upload_file_passes_through_non_webp() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = __import__("pathlib").Path(tmp) / "photo.png"
        PilImage.new("RGB", (4, 4)).save(path)

        with notifiers._mattermost_upload_file(path) as (upload_path, upload_name):
            assert upload_path == path
            assert upload_name == "photo.png"
