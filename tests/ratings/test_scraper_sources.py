"""Tests for ratings.scraper source loading."""

import os
import tempfile
from pathlib import Path

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.test import TestCase

from ratings import scraper
from ratings.models import Source


class LoadSourcesPixelfedTests(TestCase):
    def setUp(self) -> None:
        Source.objects.all().delete()

    def test_loads_all_enabled_pixelfed_accounts_from_db(self) -> None:
        Source.objects.create(type=Source.PIXELFED, name="@a@pix.a", enabled=True)
        Source.objects.create(type=Source.PIXELFED, name="@b@pix.b", enabled=True)
        Source.objects.create(type=Source.PIXELFED, name="@off@pix.off", enabled=False)
        # Any enabled non-pixelfed source prevents config.toml fallback.
        Source.objects.create(type=Source.IMGUR, name="cats", enabled=True)

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as fh:
            config_path = Path(fh.name)

        try:
            sources = scraper._load_sources(config_path)
        finally:
            config_path.unlink(missing_ok=True)

        self.assertEqual(
            sorted(sources["pixelfed_accounts"]),
            ["@a@pix.a", "@b@pix.b"],
        )

    def test_config_toml_pixelfed_accounts_list(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as fh:
            fh.write(b'[pixelfed]\naccounts = ["@art@pixelfed.social"]\n')
            fh.flush()
            config_path = Path(fh.name)

        try:
            sources = scraper._load_sources(config_path)
        finally:
            config_path.unlink(missing_ok=True)

        self.assertEqual(sources["pixelfed_accounts"], ["@art@pixelfed.social"])

    def test_config_toml_empty_pixelfed_accounts(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as fh:
            fh.write(b'[pixelfed]\naccounts = []\n')
            fh.flush()
            config_path = Path(fh.name)

        try:
            sources = scraper._load_sources(config_path)
        finally:
            config_path.unlink(missing_ok=True)

        self.assertEqual(sources["pixelfed_accounts"], [])
