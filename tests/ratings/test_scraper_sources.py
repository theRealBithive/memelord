"""Tests for ratings.scraper source loading."""

import os
import tempfile
import uuid
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

    def test_loads_all_enabled_pixelfed_instances_from_db(self) -> None:
        Source.objects.create(type=Source.PIXELFED, name="https://pix.a", enabled=True)
        Source.objects.create(type=Source.PIXELFED, name="https://pix.b", enabled=True)
        Source.objects.create(
            type=Source.PIXELFED, name="https://pix.off", enabled=False
        )
        # Any enabled non-pixelfed source prevents config.toml fallback.
        Source.objects.create(type=Source.IMGUR, name="cats", enabled=True)

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as fh:
            config_path = Path(fh.name)

        try:
            sources = scraper._load_sources(config_path)
        finally:
            config_path.unlink(missing_ok=True)

        self.assertEqual(
            sorted(sources["pixelfed_instances"]),
            ["https://pix.a", "https://pix.b"],
        )

    def test_config_toml_pixelfed_becomes_single_item_list(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as fh:
            fh.write(b'[pixelfed]\ninstance_base = "https://pix.example"\n')
            fh.flush()
            config_path = Path(fh.name)

        try:
            sources = scraper._load_sources(config_path)
        finally:
            config_path.unlink(missing_ok=True)

        self.assertEqual(sources["pixelfed_instances"], ["https://pix.example"])
