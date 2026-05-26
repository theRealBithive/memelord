"""Protected media serving (works when DEBUG=False)."""

import os
import uuid
from pathlib import Path

import django
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, override_settings


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    (images / "test.jpg").write_bytes(b"\xff\xd8\xff fake jpeg")
    return tmp_path


@pytest.fixture
def auth_client() -> Client:
    client = Client()
    username = f"media_{uuid.uuid4().hex[:8]}"
    User.objects.create_user(username, password="secret")
    client.login(username=username, password="secret")
    return client


@override_settings(DEBUG=False)
def test_anonymous_media_request_redirects_to_login(media_root: Path) -> None:
    with override_settings(MEDIA_ROOT=media_root, DATA_DIR=media_root):
        response = Client().get("/media/images/test.jpg")
    assert response.status_code == 302
    assert response.url.startswith("/login/")


@override_settings(DEBUG=False)
def test_authenticated_media_request_returns_image(
    media_root: Path, auth_client: Client
) -> None:
    with override_settings(MEDIA_ROOT=media_root, DATA_DIR=media_root):
        response = auth_client.get("/media/images/test.jpg")
    assert response.status_code == 200
    assert response["Content-Type"] == "image/jpeg"
    assert b"".join(response.streaming_content) == b"\xff\xd8\xff fake jpeg"


@override_settings(DEBUG=False)
def test_media_path_outside_allowlist_returns_404(
    media_root: Path, auth_client: Client
) -> None:
    (media_root / "config.toml").write_text("secret", encoding="utf-8")
    with override_settings(MEDIA_ROOT=media_root, DATA_DIR=media_root):
        assert auth_client.get("/media/config.toml").status_code == 404
        assert auth_client.get("/media/memelord.db").status_code == 404


@override_settings(DEBUG=False)
def test_media_path_traversal_returns_404(
    media_root: Path, auth_client: Client
) -> None:
    (media_root / "memelord.db").write_text("db", encoding="utf-8")
    with override_settings(MEDIA_ROOT=media_root, DATA_DIR=media_root):
        assert auth_client.get("/media/images/../../memelord.db").status_code == 404
