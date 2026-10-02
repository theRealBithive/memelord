from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from ratings.reset import CONFIRM_WORD, fresh_start


class Command(BaseCommand):
    help = (
        "Delete every image (rows, files, scores, embeddings, predictions, image tags), "
        "both classifier weight files and the scrape cursors. Sources, channels, "
        "thresholds, schedule, users, the tag vocabulary and the log stay. "
        "Stop any running scrape or train job first; the CLI does not check."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--yes", action="store_true", help=f"Skip the interactive '{CONFIRM_WORD}' prompt."
        )

    def handle(self, *args, **options):
        if not options["yes"]:
            typed = input(f"This deletes the whole image library. Type {CONFIRM_WORD} to continue: ")
            if typed.strip() != CONFIRM_WORD:
                raise CommandError("Aborted; nothing was deleted.")
        result = fresh_start(
            Path(settings.DATA_DIR),
            [Path(settings.WEIGHTS_PATH), Path(settings.NSFW_WEIGHTS_PATH)],
        )
        self.stdout.write(
            f"Fresh start done: {result['images']} image rows, {result['files']} files and "
            f"{result['weights']} classifier file(s) removed; {result['cursors']} scrape cursor(s) reset."
        )
