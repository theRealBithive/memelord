"""CSRF and HTTPS reverse-proxy settings."""

import os

import django
import pytest
from django.test import Client, override_settings

from memelord.settings import _csv_env, _https_csrf_origins_for_hosts


@pytest.fixture(scope="module", autouse=True)
def django_setup() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
    django.setup()


def test_csv_env_splits_and_strips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_CSV", " a ,b, c ")
    assert _csv_env("TEST_CSV") == ["a", "b", "c"]


def test_https_csrf_origins_skips_local_hosts() -> None:
    hosts = ["localhost", "127.0.0.1", "meme.webway.link"]
    assert _https_csrf_origins_for_hosts(hosts) == ["https://meme.webway.link"]


@override_settings(
    DEBUG=False,
    ALLOWED_HOSTS=["meme.webway.link"],
    CSRF_TRUSTED_ORIGINS=["https://meme.webway.link"],
    SECURE_PROXY_SSL_HEADER=("HTTP_X_FORWARDED_PROTO", "https"),
    CSRF_COOKIE_SECURE=False,
    SESSION_COOKIE_SECURE=False,
)
def test_login_post_not_rejected_when_proxy_terminates_tls() -> None:
    """POST must not 403 when browser uses HTTPS but Gunicorn sees HTTP + X-Forwarded-Proto."""
    client = Client(enforce_csrf_checks=True)
    page = client.get("/login/", HTTP_HOST="meme.webway.link")
    assert page.status_code == 200
    csrf_token = page.cookies["csrftoken"].value

    response = client.post(
        "/login/",
        {
            "username": "nobody",
            "password": "wrong",
            "csrfmiddlewaretoken": csrf_token,
        },
        HTTP_HOST="meme.webway.link",
        HTTP_X_FORWARDED_PROTO="https",
        HTTP_REFERER="https://meme.webway.link/login/",
        HTTP_ORIGIN="https://meme.webway.link",
    )
    assert response.status_code != 403
