"""
Review card structure (UI overhaul).

Contract: V4 Jede Fläche, die mit dem Finger getroffen werden muss, ist
             mindestens 44 px hoch und breit. Die Score-Knöpfe 0 bis 6 stehen
             in einer eigenen Zeile, Purge steht nicht neben einem Score.
Contract: V5 Jede irreversible Aktion (Purge, Source löschen, Channel löschen,
             Tag löschen, Log leeren) verlangt eine Bestätigung im selben Sheet.
             Scores, NSFW-Markierung und Tags brauchen keine.
Contract: V3 Der Schalter „NSFW einblenden“ und der Knopf „dieses Bild als NSFW
             markieren“ sind beschriftet und optisch unterscheidbar.

Pixel sizes cannot be measured without a browser; the row heights live in the
--score-h and --action-h tokens, which test_ui_guards pins. What can be
checked here is the structure the sizes hang on: which buttons share a row.
"""

from __future__ import annotations

import os
import uuid
from html.parser import HTMLParser

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from ratings.models import Image

VOID_TAGS = {"img", "input", "br", "meta", "link", "hr"}


class _Elements(HTMLParser):
    """Collects every element with its own classes and its ancestors' classes."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[list[str]] = []
        self.elements: list[dict] = []

    def handle_starttag(self, tag, attrs) -> None:
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        ancestors = [c for frame in self.stack for c in frame]
        self.elements.append({"tag": tag, "attrs": attrs, "classes": classes, "ancestors": ancestors})
        if tag not in VOID_TAGS:
            self.stack.append(classes)

    def handle_endtag(self, tag) -> None:
        if self.stack:
            self.stack.pop()


def _elements(html: str) -> list[dict]:
    parser = _Elements()
    parser.feed(html)
    return parser.elements


@override_settings(DEBUG=True)
class ReviewCardTests(TestCase):
    def setUp(self) -> None:
        Image.objects.all().delete()
        user = get_user_model().objects.create_user(f"card-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)
        h = uuid.uuid4().hex
        # predicted_score is set so the card does not try to compute one lazily.
        self.image = Image.objects.create(
            content_hash=h, file_path=f"images/{h}.png", source_label="t", predicted_score=0.9
        )
        self.html = self.client.get(reverse("review_corpus")).content.decode()
        self.elements = _elements(self.html)

    def _in_row(self, row_class: str) -> list[dict]:
        return [e for e in self.elements if row_class in e["ancestors"] and e["tag"] in ("button", "span", "form")]

    def test_score_row_holds_exactly_the_scale_zero_to_six(self) -> None:
        """Contract: V4"""
        buttons = [e for e in self._in_row("score-row") if e["tag"] == "button"]
        self.assertEqual([e["attrs"].get("data-score") for e in buttons], [str(n) for n in range(7)])
        for button in buttons:
            self.assertIn("score-btn", button["classes"])
            self.assertNotIn("hx-confirm", button["attrs"], "scores need no confirmation (V5)")

    def test_purge_lives_in_the_action_row_far_from_the_scores_and_asks_first(self) -> None:
        """Contract: V4, V5"""
        purge = [e for e in self.elements if e["attrs"].get("data-action") == "purge"]
        self.assertEqual(len(purge), 1)
        self.assertIn("action-row", purge[0]["ancestors"])
        self.assertNotIn("score-row", purge[0]["ancestors"])
        self.assertIn("hx-confirm", purge[0]["attrs"])
        self.assertEqual(purge[0]["attrs"].get("data-confirm-label"), "Purge")
        action_row_buttons = [e for e in self._in_row("action-row") if e["tag"] == "button"]
        self.assertEqual(action_row_buttons[-1]["attrs"].get("data-action"), "purge", "Purge is the last action")
        self.assertFalse(any("score-btn" in e["classes"] for e in self._in_row("action-row")))

    def test_nsfw_mark_button_is_labelled_and_needs_no_confirmation(self) -> None:
        """Contract: V3, V5"""
        nsfw = [e for e in self.elements if e["attrs"].get("data-action") == "nsfw"]
        self.assertEqual(len(nsfw), 1)
        self.assertNotIn("hx-confirm", nsfw[0]["attrs"])
        self.assertEqual(nsfw[0]["attrs"].get("aria-pressed"), "false")
        self.assertIn('<span class="action-label">NSFW</span>', self.html)
        self.assertIn("Show NSFW", self.html, "the visibility switch in the nav keeps its own wording")

    def test_marking_nsfw_shows_the_pressed_state(self) -> None:
        """Contract: V3"""
        Image.objects.filter(pk=self.image.pk).update(is_nsfw=True)
        self.client.post(reverse("nsfw_toggle"))  # merge NSFW into Review so the image stays visible
        html = self.client.get(reverse("review_corpus_image", args=[self.image.content_hash])).content.decode()
        nsfw = next(e for e in _elements(html) if e["attrs"].get("data-action") == "nsfw")
        self.assertEqual(nsfw["attrs"].get("aria-pressed"), "true")
        self.assertIn("action-btn--on", nsfw["classes"])

    def _other_unrated(self, **fields) -> Image:
        h = uuid.uuid4().hex
        return Image.objects.create(
            content_hash=h, file_path=f"images/{h}.png", source_label="t", predicted_score=0.9, **fields
        )

    def test_marking_an_unrated_image_nsfw_advances_when_nsfw_is_hidden(self) -> None:
        """Contract: V3 — the mark leaves the queue it was taken from, so the card moves on."""
        other = self._other_unrated()
        response = self.client.post(
            reverse("toggle_nsfw", args=[self.image.content_hash]), {"mode": "corpus"}
        )
        self.image.refresh_from_db()
        self.assertTrue(self.image.is_nsfw)
        html = response.content.decode()
        self.assertIn(other.content_hash, html)
        self.assertNotIn(self.image.content_hash, html)

    def test_marking_a_rated_image_nsfw_keeps_it_in_view(self) -> None:
        """Contract: V3 — re-review from the gallery stays put and shows the pressed state."""
        Image.objects.filter(pk=self.image.pk).update(score=4)
        response = self.client.post(
            reverse("toggle_nsfw", args=[self.image.content_hash]), {"mode": "corpus"}
        )
        html = response.content.decode()
        self.assertIn(self.image.content_hash, html)
        nsfw = next(e for e in _elements(html) if e["attrs"].get("data-action") == "nsfw")
        self.assertEqual(nsfw["attrs"].get("aria-pressed"), "true")

    def test_clearing_the_mark_in_the_nsfw_queue_advances(self) -> None:
        """Contract: V3"""
        Image.objects.filter(pk=self.image.pk).update(is_nsfw=True)
        other = self._other_unrated(is_nsfw=True)
        response = self.client.post(
            reverse("toggle_nsfw", args=[self.image.content_hash]), {"mode": "nsfw_corpus"}
        )
        self.image.refresh_from_db()
        self.assertFalse(self.image.is_nsfw)
        html = response.content.decode()
        self.assertIn(other.content_hash, html)
        self.assertNotIn(self.image.content_hash, html)


@override_settings(DEBUG=True)
class ShowNsfwSwitchTests(TestCase):
    """The nav's Show NSFW switch flips the session and returns to the page it was pressed on."""

    def setUp(self) -> None:
        user = get_user_model().objects.create_user(f"switch-{uuid.uuid4().hex[:8]}", password="pw")
        self.client.force_login(user)

    def test_returns_to_the_local_page_it_was_pressed_on(self) -> None:
        """Contract: V3"""
        response = self.client.post(reverse("nsfw_toggle"), HTTP_REFERER="http://testserver/gallery/?sort=random")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], "http://testserver/gallery/?sort=random")
        self.assertTrue(self.client.session["show_nsfw"])

    def test_an_offsite_referer_falls_back_to_the_index(self) -> None:
        """OWASP A01 (unvalidated redirect): a forged Referer never leaves this host."""
        for referer in ("https://evil.example/phish", "//evil.example", "javascript:alert(1)", ""):
            with self.subTest(referer=referer):
                response = self.client.post(reverse("nsfw_toggle"), HTTP_REFERER=referer)
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response["Location"], reverse("index"))
