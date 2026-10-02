"""
Gallery lightbox (UI overhaul).

Contract: V9 Review und Gallery-Lightbox benutzen dieselbe Score-Zeile und
             denselben Tag-Editor. Eine Änderung in der Lightbox ist sofort im
             Grid sichtbar, die Reihenfolge prev/next entspricht dem Grid.
Contract: V5 … Scores, NSFW-Markierung und Tags brauchen keine [Bestätigung].

The order of prev/next is computed in gallery.js from the grid's DOM order;
the server-side half of that promise is that the lightbox carries no server
navigation at all, which is what the first test pins.
"""

from __future__ import annotations

import json
import os
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.django import TestCase as HypothesisTestCase

from ratings.models import Image
from ratings.toast import TOAST_HEADER


def _image(**fields) -> Image:
    h = uuid.uuid4().hex
    defaults = {"content_hash": h, "file_path": f"images/{h}.png", "source_label": "t", "predicted_score": 0.9}
    defaults.update(fields)
    return Image.objects.create(**defaults)


def _login(client) -> None:
    user = get_user_model().objects.create_user(f"lb-{uuid.uuid4().hex[:8]}", password="pw")
    client.force_login(user)


@override_settings(DEBUG=True)
class LightboxTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        _login(self.client)
        self.image = _image(score=3)

    def test_panel_uses_the_shared_rows_and_steps_through_the_grid_on_the_client(self) -> None:
        """Contract: V9"""
        html = self.client.get(reverse("lightbox", args=[self.image.content_hash])).content.decode()
        self.assertIn('class="score-row"', html)
        self.assertIn('class="card-tags"', html)
        self.assertEqual(html.count("data-score="), 7)
        self.assertIn(f'hx-post="{reverse("lightbox_score", args=[self.image.content_hash])}"', html)
        self.assertIn('hx-target="#lightbox-content"', html)
        for action in ("prev", "next"):
            self.assertIn(f'data-action="{action}"', html)
        self.assertNotIn("hx-get=", html, "prev/next are client-side: the grid order is the only order")
        self.assertIn("hx-confirm=", html)  # purge asks (V5)

    def test_grid_cards_carry_the_lightbox_url_and_a_visible_badge(self) -> None:
        """Contract: V6, V9"""
        unrated = _image(score=None, predicted_score=0.1)
        for url_name in ("gallery", "below_cutoff"):
            html = self.client.get(reverse(url_name)).content.decode()
            shown = self.image if url_name == "gallery" else unrated
            self.assertIn(f'id="item-{shown.content_hash}"', html, url_name)
            self.assertIn(reverse("lightbox", args=[shown.content_hash]), html, url_name)
        gallery_html = self.client.get(reverse("gallery")).content.decode()
        self.assertIn('class="gallery-item-score gallery-item-score--3">3<', gallery_html)
        below_html = self.client.get(reverse("below_cutoff")).content.decode()
        self.assertIn("gallery-item-score--unrated", below_html)

    def test_scoring_updates_the_grid_card_and_badges_in_the_same_response(self) -> None:
        """Contract: V9"""
        response = self.client.post(reverse("lightbox_score", args=[self.image.content_hash]), {"score": "5"})
        self.assertEqual(response.status_code, 200)
        self.image.refresh_from_db()
        self.assertEqual(self.image.score, 5)
        html = response.content.decode()
        self.assertIn(f'id="item-{self.image.content_hash}"', html)
        self.assertIn('hx-swap-oob="true"', html)
        self.assertIn("gallery-item-score--5", html)
        self.assertIn('id="badge-queue" hx-swap-oob="true"', html)
        self.assertIn("score-btn--5 score-active", html)
        self.assertNotIn(TOAST_HEADER, response.headers, "a score is routine, not news")

    def test_nsfw_toggle_reports_the_pressed_state(self) -> None:
        """Contract: V3, V9"""
        response = self.client.post(reverse("lightbox_nsfw", args=[self.image.content_hash]))
        self.image.refresh_from_db()
        self.assertTrue(self.image.is_nsfw)
        self.assertIn('aria-pressed="true"', response.content.decode())

    def test_purge_removes_the_card_closes_the_panel_and_reports(self) -> None:
        """Contract: V5, V7, V9"""
        response = self.client.post(reverse("lightbox_purge", args=[self.image.content_hash]))
        self.image.refresh_from_db()
        self.assertTrue(self.image.is_purged)
        html = response.content.decode()
        self.assertIn(f'<a id="item-{self.image.content_hash}" hx-swap-oob="delete"></a>', html)
        self.assertNotIn("lb-inner", html)
        self.assertEqual(json.loads(response[TOAST_HEADER])["toast"]["message"], "Image purged")

    def test_purged_images_have_no_lightbox(self) -> None:
        Image.objects.filter(pk=self.image.pk).update(is_purged=True)
        self.assertEqual(self.client.get(reverse("lightbox", args=[self.image.content_hash])).status_code, 404)


class LightboxScoreProperty(HypothesisTestCase):
    # No setUp: Hypothesis' Django TestCase runs the first setUp before
    # _pre_setup creates self.client, so each example logs in itself.

    @settings(max_examples=30, deadline=None)
    @given(value=st.one_of(st.integers(-3, 9), st.sampled_from(["", "x", "3.5", "seven", " 4"])))
    def test_the_scale_is_exactly_zero_to_six(self, value) -> None:
        """Contract: V4 (0–6 is the scale) — anything else leaves the image as it was.

        The generator reaches every accepted value, both out-of-range sides,
        and non-numeric text; " 4" is the one string Python's int() accepts.
        """
        Image.objects.all().delete()
        _login(self.client)
        image = _image(score=3)
        self.client.post(reverse("lightbox_score", args=[image.content_hash]), {"score": value})
        image.refresh_from_db()
        if isinstance(value, int) and 0 <= value <= 6:
            self.assertEqual(image.score, value)
        elif value == " 4":
            self.assertEqual(image.score, 4)
        else:
            self.assertEqual(image.score, 3)
