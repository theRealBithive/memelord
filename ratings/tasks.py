from pathlib import Path

from django.conf import settings

from ratings import scraper


def run_scrape():
    scraper.run(
        config_path=Path(settings.CONFIG_PATH),
        data_dir=Path(settings.DATA_DIR),
        weights_path=Path(settings.WEIGHTS_PATH),
    )
