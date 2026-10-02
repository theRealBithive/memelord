"""
Rendered-page checks for the UI overhaul.

Contract: V1  Jede Seite teilt Kopf, Navigation und Script-Grundgerüst. Keine
              Seite bringt ein eigenes Stylesheet oder Inline-Styles mit.
Contract: V2  Die Navigation zeigt jeden Bereich genau einmal. Der aktuelle
              Bereich ist auf jeder Seite markiert, auch Stats, Logs, Config
              und Tags. Jeder Bereich ist auf jeder Bildschirmbreite erreichbar.
Contract: V10 Die Oberfläche lädt nichts von Drittservern und ist als
              Home-Screen-App installierbar.

Pages are rendered through the test client against the dev database (the
project's test convention) so the assertions see the real template inheritance
and context, not a unit-level approximation.
"""

from __future__ import annotations

import os
import re

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

PAGES = {
    "review_corpus": "corpus",
    "rate_nsfw_corpus": "nsfw_corpus",
    "gallery": "gallery",
    "below_cutoff": "below_cutoff",
    "stats": "stats",
    "logs": "logs",
    "config": "config",
    "tag_list": "tags",
}


class UiBaseTests(TestCase):
    def setUp(self) -> None:
        user = get_user_model().objects.create_user("ui-tester", password="pw")
        self.client.force_login(user)

    def _html(self, url_name: str) -> str:
        response = self.client.get(reverse(url_name))
        self.assertEqual(response.status_code, 200, url_name)
        return response.content.decode()

    def test_every_page_shares_the_base_skeleton(self) -> None:
        """Contract: V1, V10"""
        for url_name in PAGES:
            html = self._html(url_name)
            self.assertEqual(html.count('id="main-nav"'), 1, url_name)
            self.assertEqual(html.count("app.css"), 1, url_name)
            self.assertIn('rel="manifest"', html, url_name)
            self.assertIn("X-CSRFToken", html, url_name)
            self.assertIn('src="/static/vendor/htmx.min.js"', html, url_name)
            self.assertNotIn("<style", html, url_name)
            self.assertNotIn("unpkg.com", html, url_name)
            stray_styles = [m for m in re.findall(r'style="([^"]*)"', html) if not m.startswith("width:")]
            self.assertEqual(stray_styles, [], url_name)

    def test_login_page_has_no_nav_but_the_same_skeleton(self) -> None:
        """Contract: V1"""
        self.client.logout()
        html = self.client.get(reverse("login")).content.decode()
        self.assertNotIn('id="main-nav"', html)
        self.assertIn('rel="manifest"', html)
        self.assertEqual(html.count("app.css"), 1)
