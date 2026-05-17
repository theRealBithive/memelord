"""Static file serving in production (DEBUG=False)."""

import os
from pathlib import Path

import django
import pytest
from django.conf import settings
from django.core.management import call_command
from django.test import Client, override_settings


@pytest.fixture(scope="module", autouse=True)
def django_setup() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
    django.setup()


def test_whitenoise_middleware_follows_security_middleware() -> None:
    security_idx = settings.MIDDLEWARE.index(
        "django.middleware.security.SecurityMiddleware"
    )
    whitenoise_idx = settings.MIDDLEWARE.index(
        "whitenoise.middleware.WhiteNoiseMiddleware"
    )
    assert whitenoise_idx == security_idx + 1


def test_app_css_served_with_css_content_type_when_debug_false(tmp_path: Path) -> None:
    static_root = tmp_path / "staticfiles"
    with override_settings(DEBUG=False, STATIC_ROOT=static_root):
        call_command("collectstatic", verbosity=0, interactive=False)
        response = Client().get("/static/app.css")
    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/css")
