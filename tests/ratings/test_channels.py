"""Tests for NotificationChannel model and share dispatch."""

import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.test import TestCase

from ratings.models import NotificationChannel


class NotificationChannelConfiguredTests(TestCase):
    def test_mattermost_is_configured_when_credentials_present(self) -> None:
        ch = NotificationChannel(
            name="TownSquare",
            service=NotificationChannel.MATTERMOST,
            mm_base_url="https://chat.example.com",
            mm_token="tok",
            mm_channel_id="chan1",
        )
        self.assertTrue(ch.is_configured)

    def test_mattermost_not_configured_when_token_missing(self) -> None:
        ch = NotificationChannel(
            name="TownSquare",
            service=NotificationChannel.MATTERMOST,
            mm_base_url="https://chat.example.com",
            mm_token="",
            mm_channel_id="chan1",
        )
        self.assertFalse(ch.is_configured)

    def test_signal_is_configured_when_credentials_present(self) -> None:
        ch = NotificationChannel(
            name="Simon",
            service=NotificationChannel.SIGNAL,
            signal_api_url="http://localhost:8080",
            signal_sender="+49000",
            signal_recipients="+49111",
        )
        self.assertTrue(ch.is_configured)

    def test_signal_not_configured_when_recipients_missing(self) -> None:
        ch = NotificationChannel(
            name="Simon",
            service=NotificationChannel.SIGNAL,
            signal_api_url="http://localhost:8080",
            signal_sender="+49000",
            signal_recipients="",
        )
        self.assertFalse(ch.is_configured)

    def test_channel_str_includes_name_and_service(self) -> None:
        ch = NotificationChannel(name="Aurea", service=NotificationChannel.SIGNAL)
        self.assertEqual(str(ch), "Aurea (signal)")
