from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from core import brain
from ratings.embeddings import reencode_stale_embeddings, stale_images


class Command(BaseCommand):
    help = (
        "Re-encode images whose embedding is missing or was produced by an older "
        "encoder. Use after an encoder upgrade; a Train run does the same as a side effect."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit", type=int, default=None, help="Max images to re-encode this run."
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=256,
            help="Images saved to the DB per chunk (resume granularity).",
        )
        parser.add_argument(
            "--batch-size", type=int, default=32, help="Images per encoder forward pass."
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Only report how many rows are stale."
        )

    def handle(self, *args, **options):
        stale = stale_images().count()
        if options["dry_run"]:
            self.stdout.write(f"{stale} image(s) would be re-encoded with {brain.ENCODER_ID}.")
            return
        self.stdout.write(f"{stale} stale image(s); encoding with {brain.ENCODER_ID}…")
        result = reencode_stale_embeddings(
            Path(settings.DATA_DIR),
            chunk_size=options["chunk_size"],
            batch_size=options["batch_size"],
            limit=options["limit"],
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Re-encoded {result['encoded']} image(s); "
                f"{result['missing_file']} file(s) missing, {result['unreadable']} unreadable."
            )
        )
