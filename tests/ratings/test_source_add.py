"""Tests for the source_add view's handle/name validation.

Pixelfed and Mastodon both target an account via @user@instance (both speak the
Mastodon-compatible API), so both reject a bare name without an instance.
"""

import os
import uuid

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from ratings.models import Source


@override_settings(DEBUG=True)
class SourceAddTests(TestCase):
    def setUp(self) -> None:
        Source.objects.all().delete()
        self.client = Client()
        username = f"src_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    def _add(self, stype: str, name: str):
        return self.client.post(
            reverse("source_add"), {"type": stype, "name": name}
        )

    def test_pixelfed_account_handle_accepted(self) -> None:
        response = self._add(Source.PIXELFED, "@art@pixelfed.social")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            Source.objects.filter(
                type=Source.PIXELFED, name="@art@pixelfed.social"
            ).exists()
        )

    def test_pixelfed_bare_name_rejected(self) -> None:
        response = self._add(Source.PIXELFED, "pixelfed.social")
        self.assertIn("must include an instance", response.content.decode())
        self.assertFalse(Source.objects.filter(type=Source.PIXELFED).exists())

    def test_pixelfed_url_rejected(self) -> None:
        """A bare instance URL is no longer valid — Pixelfed targets an account now."""
        response = self._add(Source.PIXELFED, "https://pixelfed.social")
        self.assertIn("must include an instance", response.content.decode())
        self.assertFalse(Source.objects.filter(type=Source.PIXELFED).exists())

    def test_mastodon_handle_accepted(self) -> None:
        response = self._add(Source.MASTODON, "@user@mastodon.social")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(Source.objects.filter(type=Source.MASTODON).exists())

    def test_mastodon_bare_name_rejected(self) -> None:
        response = self._add(Source.MASTODON, "justaname")
        self.assertIn("must include an instance", response.content.decode())

    def test_fourchan_board_has_no_handle_requirement(self) -> None:
        response = self._add(Source.FOURCHAN, "wg")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            Source.objects.filter(type=Source.FOURCHAN, name="wg").exists()
        )

    def test_invalid_type_rejected(self) -> None:
        response = self._add("myspace", "@user@myspace.social")
        self.assertIn("Invalid source type", response.content.decode())

    def test_requires_login(self) -> None:
        response = Client().post(
            reverse("source_add"), {"type": Source.FOURCHAN, "name": "wg"}
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)


@override_settings(DEBUG=True)
class FlickrSourceAddTests(TestCase):
    """Contract V1/V2 of tests/retina/test_flickr.py over HTTP."""

    def setUp(self) -> None:
        Source.objects.all().delete()
        self.client = Client()
        username = f"src_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")

    def _add(self, name: str):
        return self.client.post(reverse("source_add"), {"type": Source.FLICKR, "name": name})

    def test_pasted_url_is_stored_under_the_canonical_name(self) -> None:
        """Contract: V1"""
        response = self._add("https://www.flickr.com/groups/419512@N22/pool/with/54645046861")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            list(Source.objects.filter(type=Source.FLICKR).values_list("name", flat=True)),
            ["group/419512@N22"],
        )
        self.assertIn("group/419512@N22", response.content.decode())

    def test_two_spellings_of_one_group_give_one_source(self) -> None:
        """Contract: V1"""
        self._add("https://www.flickr.com/groups/419512@N22/pool/")
        self._add("group/419512@N22")
        self.assertEqual(Source.objects.filter(type=Source.FLICKR).count(), 1)

    def test_user_url_is_accepted(self) -> None:
        """Contract: V1"""
        self._add("flickr.com/photos/alexgee/55573868969/")
        self.assertTrue(Source.objects.filter(type=Source.FLICKR, name="user/alexgee").exists())

    def test_non_flickr_input_is_rejected_without_a_source(self) -> None:
        """Contract: V2"""
        for name in (
            "https://example.com/groups/419512@N22/",
            "419512@N22",
            "https://www.flickr.com/photos/tags/cats",
            "group/<script>",
        ):
            response = self._add(name)
            self.assertIn("Flickr source must be a group or user URL", response.content.decode())
        self.assertFalse(Source.objects.filter(type=Source.FLICKR).exists())
