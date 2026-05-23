"""Site-wide login gate middleware."""

import os
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, override_settings


@override_settings(DEBUG=False)
def test_anonymous_root_redirects_to_login() -> None:
    response = Client().get("/")
    assert response.status_code == 302
    assert response.url.startswith("/login/")


@override_settings(DEBUG=False)
def test_anonymous_review_redirects_to_login() -> None:
    response = Client().get("/review/")
    assert response.status_code == 302
    assert response.url.startswith("/login/")


@override_settings(DEBUG=False)
def test_login_page_is_public() -> None:
    response = Client().get("/login/")
    assert response.status_code == 200


@override_settings(DEBUG=False)
def test_authenticated_user_can_reach_review() -> None:
    client = Client()
    username = f"auth_{uuid.uuid4().hex[:8]}"
    User.objects.create_user(username, password="secret")
    client.login(username=username, password="secret")
    response = client.get("/review/")
    assert response.status_code == 200


@override_settings(DEBUG=False)
def test_legacy_rate_inbox_redirects_to_review() -> None:
    client = Client()
    username = f"auth_{uuid.uuid4().hex[:8]}"
    User.objects.create_user(username, password="secret")
    client.login(username=username, password="secret")
    response = client.get("/rate/inbox/", follow=False)
    assert response.status_code == 301
    assert response.url == "/review/"


@override_settings(DEBUG=False)
def test_legacy_rate_nsfw_paths_share_single_name() -> None:
    from django.urls import reverse

    assert reverse("rate_nsfw_inbox") == "/rate/nsfw/inbox/"
    client = Client()
    username = f"auth_{uuid.uuid4().hex[:8]}"
    User.objects.create_user(username, password="secret")
    client.login(username=username, password="secret")
    assert client.get("/rate/nsfw/", follow=False).status_code == 301
    assert client.get("/rate/nsfw/inbox/", follow=False).status_code == 301
