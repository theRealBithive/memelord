from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from ratings import scraper


class Command(BaseCommand):
    help = "Scrape images from all configured sources into the inbox."

    def add_arguments(self, parser):
        parser.add_argument(
            "--config",
            default=str(settings.CONFIG_PATH),
            help="Path to config.toml (default: settings.CONFIG_PATH)",
        )

    def handle(self, *args, **options):
        counts = scraper.run(
            config_path=Path(options["config"]),
            data_dir=Path(settings.DATA_DIR),
            vision=scraper.vision_config_from_settings(),
        )
        total = sum(counts.values())
        self.stdout.write(f"Done. {total} new images added to inbox.")
        for source, n in counts.items():
            self.stdout.write(f"  {source}: {n}")
