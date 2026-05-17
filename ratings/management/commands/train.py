from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from core import trainer


class Command(BaseCommand):
    help = "Train the DINOv2+LogisticRegression classifier on rated images."

    def handle(self, *args, **options):
        trainer.run(
            data_dir=Path(settings.DATA_DIR),
            weights_path=Path(settings.WEIGHTS_PATH),
            nsfw_weights_path=Path(settings.NSFW_WEIGHTS_PATH),
            nsfw_threshold=settings.NSFW_THRESHOLD,
        )
