from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from core import siglip
from ratings.search import encode_stale_search_embeddings, stale_search_images


class Command(BaseCommand):
    help = (
        "Encode images that have no current search vector (SigLIP2) in the foreground. "
        "The in-app index job does the same in background slices; use this for one-off "
        "runs, e.g. `docker compose run --rm memelord index_search`."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit", type=int, default=None, help="Max images to encode this run."
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
        stale = stale_search_images().count()
        if options["dry_run"]:
            self.stdout.write(
                f"{stale} image(s) would be encoded with {siglip.SEARCH_ENCODER_ID}."
            )
            return
        self.stdout.write(f"{stale} stale image(s); encoding with {siglip.SEARCH_ENCODER_ID}…")
        result = encode_stale_search_embeddings(
            Path(settings.DATA_DIR),
            chunk_size=options["chunk_size"],
            batch_size=options["batch_size"],
            limit=options["limit"],
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Encoded {result['encoded']} image(s); "
                f"{result['missing_file']} file(s) missing, {result['unreadable']} unreadable."
            )
        )
