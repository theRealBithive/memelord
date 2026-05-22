from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from ratings.models import Image


class Command(BaseCommand):
    help = (
        "Mark file_deleted=True on Image records whose file no longer exists on disk. "
        "Cleans up phantoms left by the concurrent-scrape unlink bug + move_image "
        "silent FileNotFoundError catch, which together produced records pointing at "
        "non-existent files in corpus/, void/, and inbox/."
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

        qs = Image.objects.filter(file_deleted=False)
        total = qs.count()
        self.stdout.write(f"Scanning {total} non-deleted records under {data_dir}…")

        missing_by_loc: dict[str, int] = {}
        marked = 0
        for img in qs.iterator(chunk_size=500):
            p = data_dir / img.file_path
            if p.exists():
                continue
            missing_by_loc[img.location] = missing_by_loc.get(img.location, 0) + 1
            if not dry:
                img.file_deleted = True
                img.save(update_fields=["file_deleted"])
                marked += 1

        for loc, n in sorted(missing_by_loc.items()):
            self.stdout.write(f"  {loc}: {n} missing")
        if dry:
            self.stdout.write(self.style.WARNING(f"Dry-run: {sum(missing_by_loc.values())} would be marked file_deleted=True"))
        else:
            self.stdout.write(self.style.SUCCESS(f"Marked {marked} records file_deleted=True"))
