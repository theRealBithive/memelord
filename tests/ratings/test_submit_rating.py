"""HTMX rating responses update the nav bar without a full page reload."""

import os
import uuid
from unittest.mock import patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.http import HttpResponse
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from ratings.models import Image


@override_settings(DEBUG=True)
class SubmitRatingHtmxTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        username = f"rate_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    @staticmethod
    def _inbox_image() -> Image:
        content_hash = uuid.uuid4().hex
        return Image.objects.create(
            content_hash=content_hash,
            file_path=f"inbox/{content_hash}.jpg",
            source_label="test",
            location=Image.INBOX,
        )

    def test_submit_rating_htmx_renders_card_and_oob_nav(self) -> None:
        img = self._inbox_image()
        self._inbox_image()
        url = reverse("submit_rating", args=[img.content_hash, "bad"])

        response = self.client.post(
            url,
            {"mode": "inbox"},
            HTTP_HX_REQUEST="true",
        )

        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('id="card"', content)
        self.assertIn('id="main-nav"', content)
        self.assertIn('hx-swap-oob="true"', content)
        self.assertIn('id="badge-inbox"', content)

    def test_submit_rating_htmx_uses_combined_template(self) -> None:
        img = self._inbox_image()
        url = reverse("submit_rating", args=[img.content_hash, "skip"])

        with patch(
            "ratings.views.render",
            return_value=HttpResponse("ok"),
        ) as mock_render:
            self.client.post(
                url,
                {"mode": "inbox"},
                HTTP_HX_REQUEST="true",
            )

        template_name = mock_render.call_args[0][1]
        self.assertEqual(template_name, "ratings/_htmx_rating.html")

    def test_submit_rating_non_htmx_uses_full_page_template(self) -> None:
        img = self._inbox_image()
        url = reverse("submit_rating", args=[img.content_hash, "skip"])

        with patch(
            "ratings.views.render",
            return_value=HttpResponse("ok"),
        ) as mock_render:
            self.client.post(url, {"mode": "inbox"})

        template_name = mock_render.call_args[0][1]
        self.assertEqual(template_name, "ratings/rate.html")
