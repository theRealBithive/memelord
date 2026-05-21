from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from ratings.scraper import suggest_tags_for_pending


class Command(BaseCommand):
    help = "Generate Florence-2 keyword suggestions for images that have none yet."

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Max images to process (default: all without suggestions).",
        )
        parser.add_argument(
            "--refill",
            action="store_true",
            help="Re-generate suggestions even for images that already have them.",
        )

    def handle(self, *args, **options):
        result = suggest_tags_for_pending(
            Path(settings.DATA_DIR),
            refill=options["refill"],
            limit=options["limit"],
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. {result['updated']} updated, {result['skipped']} skipped (missing file)."
            )
        )
