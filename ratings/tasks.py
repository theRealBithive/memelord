from pathlib import Path

from django.conf import settings

from ratings import scraper


def run_scrape():
    scraper.run(
        config_path=Path(settings.CONFIG_PATH),
        data_dir=Path(settings.DATA_DIR),
        weights_path=Path(settings.WEIGHTS_PATH),
    )


def run_train():
    from core import trainer
    try:
        trainer.run(
            data_dir=Path(settings.DATA_DIR),
            weights_path=Path(settings.WEIGHTS_PATH),
        )
        return {"ok": True}
    except SystemExit:
        return {"ok": False, "error": "Need both corpus and void images to train."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
