"""
Feedback as toasts (UI overhaul).

Contract: V7 Rückmeldungen erscheinen als Toast an einer festen Stelle, auf
             allen Seiten gleich, und verschwinden von selbst.

The server side of V7 is small: an htmx response carries the toast in its
HX-Trigger header, a full-page redirect flashes a Django message that the
next page renders into #server-toasts. Both paths end in toast.js, which has
no mutation tool in this stack, so the tests pin the data the script receives.
"""

from __future__ import annotations

import json
import os
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from unittest import mock

from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.test import TestCase, override_settings
from django.urls import reverse
from hypothesis import given
from hypothesis import strategies as st

from ratings.models import Image, NotificationChannel, Source, Tag
from ratings.toast import TOAST_HEADER, TOAST_KINDS, with_toast


@given(message=st.text(max_size=300), kind=st.sampled_from(TOAST_KINDS))
def test_toast_header_round_trips_any_message_as_data(message: str, kind: str) -> None:
    """Contract: V7

    Any text — quotes, angle brackets, emoji, line breaks — must arrive in the
    browser unchanged and as data (OWASP A03: the header is JSON, the sink is
    textContent). HTTP headers are ASCII, so the encoded header must be too.
    """
    response = with_toast(HttpResponse(status=204), message, kind)
    header = response[TOAST_HEADER]
    assert header.isascii()
    assert json.loads(header) == {"toast": {"message": message, "kind": kind}}


def test_toast_kind_defaults_to_ok_and_rejects_unknown_kinds() -> None:
    """Contract: V7"""
    response = with_toast(HttpResponse(), "done")
    assert json.loads(response[TOAST_HEADER])["toast"]["kind"] == "ok"
    try:
        with_toast(HttpResponse(), "x", "loud")
    except ValueError:
        return
    raise AssertionError("an unknown toast kind must be rejected")


@override_settings(DEBUG=True)
class FeedbackResponses(TestCase):
    def setUp(self) -> None:
        user = get_user_model().objects.create_user(f"toast-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def _toast(self, response) -> dict:
        return json.loads(response[TOAST_HEADER])["toast"]

    def test_purging_from_review_answers_with_a_toast(self) -> None:
        """Contract: V7"""
        h = uuid.uuid4().hex
        image = Image.objects.create(
            content_hash=h, file_path=f"images/{h}.png", source_label="t", predicted_score=0.9
        )
        response = self.client.post(reverse("purge_corpus", args=[h]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._toast(response), {"message": "Image purged", "kind": "ok"})
        image.refresh_from_db()
        self.assertTrue(image.is_purged)

    def test_deleting_a_source_channel_or_tag_answers_with_a_toast(self) -> None:
        """Contract: V7"""
        source = Source.objects.create(type="4chan", name=f"t{uuid.uuid4().hex[:6]}")
        channel = NotificationChannel.objects.create(
            name=f"c{uuid.uuid4().hex[:6]}", service=NotificationChannel.MATTERMOST
        )
        tag = Tag.objects.create(name=f"tag-{uuid.uuid4().hex[:6]}")
        expectations = [
            (reverse("source_delete", args=[source.pk]), "Source deleted"),
            (reverse("channel_delete", args=[channel.pk]), "Channel deleted"),
            (reverse("tag_delete", args=[tag.pk]), "Tag deleted"),
        ]
        for url, message in expectations:
            response = self.client.post(url)
            self.assertEqual(response.status_code, 200, url)
            self.assertEqual(self._toast(response), {"message": message, "kind": "ok"}, url)

    def test_clearing_the_log_flashes_a_toast_on_the_next_page(self) -> None:
        """Contract: V7 (full-page path: the redirect target renders the message)"""
        response = self.client.post(reverse("log_clear"), follow=True)
        html = response.content.decode()
        self.assertIn('id="server-toasts"', html)
        self.assertIn("Log cleared", html)
        again = self.client.get(reverse("logs")).content.decode()
        self.assertNotIn('id="server-toasts"', again)

    def test_source_import_reports_how_many_sources_arrived(self) -> None:
        """Contract: V7"""
        for count, text in ((2, "Imported 2 new sources from config.toml"), (1, "Imported 1 new source from config.toml")):
            with self.subTest(count=count):
                with mock.patch("ratings.scraper.import_from_config", return_value=count):
                    response = self.client.post(reverse("source_import"))
                self.assertEqual(response.status_code, 200)
                toast = json.loads(response[TOAST_HEADER])["toast"]
                self.assertEqual(toast, {"message": text, "kind": "info"})

    def test_purging_a_rated_image_from_re_review_reports_too(self) -> None:
        """Contract: V7 — the out-of-queue branch of the review purge carries the same toast."""
        h = uuid.uuid4().hex
        Image.objects.create(content_hash=h, file_path=f"images/{h}.png", source_label="t", score=4)
        response = self.client.post(reverse("purge_corpus", args=[h]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response[TOAST_HEADER])["toast"]["message"], "Image purged")
        self.assertTrue(Image.objects.get(content_hash=h).is_purged)
