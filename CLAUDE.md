# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
uv sync --extra dev          # install all deps including dev
make run                     # django dev server on :8000
make migrate                 # makemigrations + migrate
make superuser               # create admin user (interactive)
make test                    # full test suite
make test-fast               # skip @pytest.mark.integration tests
```

### Data tasks (CLI equivalents of the in-app buttons)
```bash
make scrape                  # scrape all sources from config.toml → data/inbox/
make train                   # encode corpus+void with DINOv2, fit classifier, save weights
uv run python manage.py scrape --config /path/to/config.toml
uv run python manage.py train
```

### Single test
```bash
uv run pytest tests/core/test_brain.py::test_get_transform_returns_compose
uv run pytest -v -m "not integration"
```

### Docker (production)
```bash
make build                         # docker compose build
docker compose up -d               # start webapp on :8000
docker compose run --rm memelord scrape   # one-off scrape
docker compose run --rm memelord train    # one-off train
docker compose run --rm memelord createsuperuser  # create admin user
```

## Architecture

### What it does
Memelord is a mobile-first Django webapp for curating images with a personal taste classifier. The operator scrapes images from the web, rates them (good / fav / bad), and uses those ratings to train a DINOv2+LogisticRegression classifier that learns to predict taste.

### Three-phase loop
1. **Scrape** (`ratings/scraper.py` + `retina/`) — pull images from configured sources into `data/inbox/`
2. **Rate** (Django webapp) — manually classify each inbox image as good / fav / bad
3. **Train** (`core/trainer.py`) — encode corpus+void images with DINOv2, fit LogisticRegression with fav-weighted samples

### Module map

**`ratings/`** — Django app (the new core)
- `models.py` — single `Image` model: `content_hash` (PK), `file_path` (relative to `DATA_DIR`), `location` (inbox/corpus/void), `is_favourite`, `rated_at`
- `views.py` — rating views (`rate_inbox/corpus/void`), `submit_rating`, `stats`, `trigger_scrape`, `trigger_train`
- `scraper.py` — orchestrates all `retina/` scrapers, deduplicates by SHA256, inserts into DB
- `admin.py` — ImageAdmin with inline thumbnails
- `management/commands/scrape.py` — CLI wrapper over `ratings/scraper.py`
- `management/commands/train.py` — CLI wrapper over `core/trainer.py`

**`core/`** — ML logic (largely unchanged from original)
- `brain.py` — DINOv2 encoder (768-d embeddings), classifier load/save
- `trainer.py` — reads corpus/void from Django ORM, computes `is_favourite` weights (fav=3.0, good=1.0), fits LogisticRegression

**`retina/`** — image scrapers (unchanged)
- `fourchan.py`, `imgur.py`, `tumblr.py`, `pixelfed.py`
- Each exposes `iter_image_urls()`/`iter_image_items()` + `download_images()` → `list[tuple[Path, str, str]]` (path, source_url, source_label)

**`memelord/`** — Django project config (`settings.py`, `urls.py`, `wsgi.py`)

### Rating modes and actions

| Mode | Queue | Bad | Fav | Good | Skip |
|---|---|---|---|---|---|
| Mixed (inbox) | `location='inbox'` | → void | → corpus ★ | → corpus | — |
| Corpus review | `location='corpus'` | → void | toggle ★ | no-op | no-op |
| Trash rescue | `location='void'` | no-op | → corpus ★ | → corpus | no-op |

Swipe left/right/up/down maps to bad/good/fav/skip. Arrow keys work on desktop.

### Key settings (`memelord/settings.py`)
- `DATA_DIR` — root for all image files and the SQLite DB (env: `DATA_DIR`, default: `BASE_DIR/data`)
- `CONFIG_PATH` — path to `config.toml` for scraper sources (env: `CONFIG_PATH`)
- `WEIGHTS_PATH` — `DATA_DIR/Janulon_weights.pkl`
- `MEDIA_ROOT = DATA_DIR`, `MEDIA_URL = "/media/"` — images served by `serve_media` (`@login_required`, inbox/corpus/void only); not via `static()` (DEBUG-gated)

### Data layout under `DATA_DIR`
```
data/
├── memelord.db          # SQLite (Django ORM)
├── Janulon_weights.pkl  # trained classifier
├── config.toml          # scraper config (mounted in Docker)
├── inbox/               # scraped, awaiting rating
├── corpus/              # rated good/fav (positive training samples)
└── void/                # rated bad (negative training samples)
```

### File move flow
`submit_rating` in `views.py` calls `_move_image()` which uses `shutil.move` to relocate the file and updates `image.file_path` + `image.location` in the DB atomically. Content hashes are the dedup key — `scraper.py` skips any download whose SHA256 already exists in the DB.

### Training weight formula
`is_favourite=True` → weight 3.0, otherwise 1.0. Passed as `sample_weight` to `LogisticRegression.fit()` in `core/trainer.py`.

### Docker
Single volume `/data` holds everything (DB, weights, images, config). Entrypoint runs `migrate` + `collectstatic` on every start then launches gunicorn. One-off commands map as `docker compose run --rm memelord <manage.py subcommand>`.

### Tests
- Mirror source structure: `tests/core/`, `tests/retina/`
- Integration tests (real DINOv2, real network): `@pytest.mark.integration`, skipped by default
- No Django test runner yet — `pytest` with `pytest-django` would need adding if Django view tests are written
