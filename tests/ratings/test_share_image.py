"""Tests for share_image view — channel dispatch, disabled-channel filtering, toast output."""

import os
import uuid
from unittest.mock import patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from ratings.models import Image, NotificationChannel


def _image():
    """Create and return a throwaway Image."""
    h = uuid.uuid4().hex
    return Image.objects.create(
        content_hash=h,
        file_path=f"images/{h}.jpg",
        source_label="test-src",
    )


def _mattermost_channel(name="TownSquare", enabled=True):
    return NotificationChannel.objects.create(
        name=name,
        service=NotificationChannel.MATTERMOST,
        enabled=enabled,
        mm_base_url="https://mm.example.com",
        mm_token="tok",
        mm_channel_id="chan1",
    )


def _signal_channel(name="Aurea", enabled=True):
    return NotificationChannel.objects.create(
        name=name,
        service=NotificationChannel.SIGNAL,
        enabled=enabled,
        signal_api_url="http://localhost:8080",
        signal_sender="+490001",
        signal_recipients="+490002",
    )


@override_settings(DEBUG=True)
class ShareImageDispatchTests(TestCase):
    def setUp(self) -> None:
        NotificationChannel.objects.all().delete()
        self.client = Client()
        username = f"share_{uuid.uuid4().hex[:8]}"
        User.objects.create_user(username, password="secret")
        self.client.login(username=username, password="secret")
        self.image = _image()

    @patch("ratings.notifiers.send_to_mattermost")
    def test_dispatches_to_selected_mattermost_channel(self, mock_mm) -> None:
        ch = _mattermost_channel()
        response = self.client.post(
            reverse("share_image", args=[self.image.content_hash]),
            {"channels": [ch.pk]},
        )
        self.assertEqual(response.status_code, 200)
        mock_mm.assert_called_once()
        self.assertIn(ch.name, response.content.decode())

    @patch("ratings.notifiers.send_to_signal")
    def test_dispatches_to_selected_signal_channel(self, mock_sig) -> None:
        ch = _signal_channel()
        response = self.client.post(
            reverse("share_image", args=[self.image.content_hash]),
            {"channels": [ch.pk]},
        )
        self.assertEqual(response.status_code, 200)
        mock_sig.assert_called_once()
        self.assertIn(ch.name, response.content.decode())

    @patch("ratings.notifiers.send_to_mattermost")
    @patch("ratings.notifiers.send_to_signal")
    def test_dispatches_to_multiple_selected_channels(self, mock_sig, mock_mm) -> None:
        mm = _mattermost_channel("TownSquare")
        sig = _signal_channel("Aurea")
        response = self.client.post(
            reverse("share_image", args=[self.image.content_hash]),
            {"channels": [mm.pk, sig.pk]},
        )
        self.assertEqual(response.status_code, 200)
        mock_mm.assert_called_once()
        mock_sig.assert_called_once()
        body = response.content.decode()
        self.assertIn("TownSquare", body)
        self.assertIn("Aurea", body)

    @patch("ratings.notifiers.send_to_mattermost")
    def test_ignores_disabled_channel_even_if_pk_passed(self, mock_mm) -> None:
        disabled = _mattermost_channel("Disabled", enabled=False)
        response = self.client.post(
            reverse("share_image", args=[self.image.content_hash]),
            {"channels": [disabled.pk]},
        )
        self.assertEqual(response.status_code, 200)
        mock_mm.assert_not_called()
        # No channel sent → "No channels selected" or empty sent list
        body = response.content.decode()
        self.assertNotIn("Disabled", body)

    @patch("ratings.notifiers.send_to_mattermost")
    def test_no_channels_posted_returns_no_channels_toast(self, mock_mm) -> None:
        response = self.client.post(
            reverse("share_image", args=[self.image.content_hash]),
            {},
        )
        self.assertEqual(response.status_code, 200)
        mock_mm.assert_not_called()
        self.assertIn("No channels", response.content.decode())

    @patch("ratings.notifiers.send_to_mattermost", side_effect=RuntimeError("timeout"))
    def test_notifier_error_appears_in_toast(self, mock_mm) -> None:
        ch = _mattermost_channel()
        response = self.client.post(
            reverse("share_image", args=[self.image.content_hash]),
            {"channels": [ch.pk]},
        )
        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn("timeout", body)
