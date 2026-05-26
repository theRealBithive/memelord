from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from ratings.models import Image


class Command(BaseCommand):
    help = (
        "Mark is_purged=True on Image records whose file no longer exists on disk. "
        "Use after manually deleting images or recovering from a storage failure."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would be marked without writing.",
        )

    def handle(self, *args, **options):
        data_dir = Path(settings.DATA_DIR)
        dry = options["dry_run"]

        qs = Image.objects.filter(is_purged=False)
        total = qs.count()
        self.stdout.write(f"Scanning {total} non-purged records under {data_dir}…")

        marked = 0
        missing = 0
        for img in qs.iterator(chunk_size=500):
            p = data_dir / img.file_path
            if p.exists():
                continue
            missing += 1
            if not dry:
                img.is_purged = True
                img.save(update_fields=["is_purged"])
                marked += 1

        if dry:
            self.stdout.write(self.style.WARNING(f"Dry-run: {missing} would be marked is_purged=True"))
        else:
            self.stdout.write(self.style.SUCCESS(f"Marked {marked} records is_purged=True"))
