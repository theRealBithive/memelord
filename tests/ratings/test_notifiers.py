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

        cfg = SimpleNamespace(
            signal_api_url="http://signal.local:8080",
            signal_sender="+15550001",
            signal_recipients="+15550002",
            signal_message_prefix="",
        )
        notifiers.send_to_signal(cfg, path, "imgur/memes")

    assert captured["url"] == "http://signal.local:8080/v2/send"
    assert captured["json"]["message"] == "imgur/memes"
    assert captured["json"]["recipients"] == ["+15550002"]
    assert captured["json"]["base64_attachments"][0].startswith("data:image/png;")
    assert captured["timeout"] == 60
