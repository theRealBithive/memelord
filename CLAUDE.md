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
make scrape                  # scrape all sources from config.toml → data/images/
make train                   # encode corpus+void with DINOv2, fit classifier, save weights
uv run python manage.py scrape --config /path/to/config.toml
uv run python manage.py train
```

### Running manage.py commands

Always prefix with `DJANGO_DEBUG=true` (required because the settings guard rejects the insecure default key in non-debug mode):

```bash
DJANGO_DEBUG=true uv run python manage.py migrate
DJANGO_DEBUG=true uv run python manage.py shell
DJANGO_DEBUG=true uv run python manage.py makemigrations
DJANGO_DEBUG=true uv run python manage.py createsuperuser
```

The `make` targets (run, migrate, superuser) already set this for you via the dev server entrypoint. Use the `DJANGO_DEBUG=true` prefix only when calling `manage.py` directly.

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
Memelord is a mobile-first Django webapp for curating images with a personal taste classifier. The operator scrapes images from the web, scores each one 1–6, and uses those scores to train a DINOv2+LogisticRegression classifier that learns to predict taste.

### Three-phase loop
1. **Scrape** (`ratings/scraper.py` + `retina/`) — pull images from configured sources into `data/images/`
2. **Rate** (Django webapp) — give each unscored image a 1–6 score (a pure DB write; no file moves)
3. **Train** (`core/trainer.py`) — encode all scored images with DINOv2, fit LogisticRegression with score-weighted samples

### Module map

**`ratings/`** — Django app (the new core)
- `models.py` — `Image` model: `content_hash` (PK), `file_path` (relative to `DATA_DIR`), `score` (0–6, NULL = unrated; 0 = trash), `is_nsfw`, `is_purged`, `rated_at`, `predicted_score`, `embedding`, `phash`. Plus `Source` (scrape config with pagination `cursor`), `Tag`, `NotificationChannel`, `ReviewThresholds`, `LogEntry`.
- `views.py` — review/score views (`review_corpus`, `score_corpus`, `purge_corpus`), `below_cutoff`, `gallery`, `gallery_action`, `stats`, `trigger_scrape`, `trigger_train`, channel + source CRUD
- `scraper.py` — orchestrates all `retina/` scrapers, deduplicates by SHA256/pHash/embedding, inserts into DB, auto-classifies
- `utils.py` — `purge_image()`: hard-delete file from disk + mark `is_purged=True`
- `admin.py` — ImageAdmin with inline thumbnails
- `management/commands/scrape.py` — CLI wrapper over `ratings/scraper.py`
- `management/commands/train.py` — CLI wrapper over `core/trainer.py`

**`core/`** — ML logic
- `brain.py` — DINOv2 encoder (768-d embeddings), classifier load/save
- `trainer.py` — reads scored images from Django ORM (good = score ≥ 3, bad = score ≤ 2), applies the per-score weight table, fits LogisticRegression

**`retina/`** — image scrapers
- `fourchan.py`, `imgur.py`, `tumblr.py`, `reddit.py` — URL/timeline scrapers, each exposes `iter_image_urls()`/`iter_image_items()` + `download_images()`
- `_mastoapi.py` — shared Mastodon-compatible API helpers (account lookup, cursor pagination, download)
- `mastodon.py`, `pixelfed.py` — thin wrappers over `_mastoapi.py`; both target an account via `@user@instance` handle
- All `download_images()` return `list[tuple[Path, str, str]]` (path, source_url, source_label)

**`memelord/`** — Django project config (`settings.py`, `urls.py`, `wsgi.py`, `media.py`)

### Rating model and queues
A score is a pure DB write — no file ever moves. `score IS NULL` means unrated, `score IS NOT NULL` means rated. Scores run **0–6**: 1–6 is the taste scale, and **0 = "trash"** — garbage below the scale, kept on disk as the strongest negative training example (distinct from purge, which deletes the file).

| Queue | View | Filter | Action |
|---|---|---|---|
| Review (SFW/NSFW) | `review_corpus` | `score IS NULL`, visibility dial | trash (0) or score 1–6 → advance to next; purge → hard-delete |
| Below Cutoff | `below_cutoff` | `score ≤ 2` | re-score upward or purge from the gallery lightbox |
| Gallery | `gallery` | `score ≥ min_score` | re-score / tag / purge / share |

Keys `0`–`6` rate and advance in review (`0` = trash); `s` opens the share picker, `n` toggles NSFW, arrows navigate. (The fav star was removed with the 1–6 model; trash returned as score 0.)

### Key settings (`memelord/settings.py`)
- `DATA_DIR` — root for all image files and the SQLite DB (env: `DATA_DIR`, default: `BASE_DIR/data`)
- `CONFIG_PATH` — path to `config.toml` for scraper sources (env: `CONFIG_PATH`)
- `WEIGHTS_PATH` — `DATA_DIR/Janulon_weights.pkl`
- `MEDIA_ROOT = DATA_DIR`, `MEDIA_URL = "/media/"` — images served by `serve_media` (`@login_required`, `images/` prefix only); not via `static()` (DEBUG-gated)

### Data layout under `DATA_DIR`
```
data/
├── memelord.db          # SQLite (Django ORM)
├── Janulon_weights.pkl  # trained classifier
├── config.toml          # scraper config (mounted in Docker)
└── images/              # all scraped images, one flat directory (path = file_path)
```

### Rating flow (no file moves)
`score_corpus` in `views.py` sets `image.score` + `image.rated_at` with a single DB write — files never move. `file_path` is set once at scrape time and never touched. Purging (`purge_image` in `utils.py`) hard-deletes the file and sets `is_purged=True`; the `content_hash` stays in the dedup index so it can't be re-downloaded. Content hashes are the dedup key — `scraper.py` skips any download whose SHA256 already exists in the DB.

### Training weight formula
Positive class is score ≥ 3, negative is score ≤ 2 (score-NULL images are excluded). Per-score weights are symmetric: 5–6 → 3.0, 3–4 → 1.0, 1–2 → 1.0, and 0 (trash) → 3.0 (`_POSITIVE_WEIGHTS` / `_NEGATIVE_WEIGHTS` in `core/trainer.py`). Passed as `sample_weight` to `LogisticRegression.fit()`.

### Docker
Single volume `/data` holds everything (DB, weights, images, config). Entrypoint runs `migrate` + `collectstatic` on every start then launches gunicorn. One-off commands map as `docker compose run --rm memelord <manage.py subcommand>`.

### Tests
- Mirror source structure: `tests/core/`, `tests/retina/`, `tests/ratings/`, `tests/memelord/`
- Integration tests (real DINOv2, real network): `@pytest.mark.integration`, skipped by default
- Django view/model tests use `django.test.TestCase` (each module calls `django.setup()` at import); run with plain `pytest` (no `pytest-django` plugin)

## Code style

### Comments
Every non-trivial function should have a docstring explaining **why** it works the way it does — hidden constraints, design trade-offs, non-obvious decisions, workarounds for specific behaviour. Not *what* the code does (the names do that).

`core/brain.py` is the reference: it explains why ViT-B/14 was chosen, why `eval()` is required, why pickle is used, why float32 blobs instead of a native array type.

One-liners are fine for simple helpers. Multi-paragraph is fine when there is genuinely important context. Never strip existing docstrings.
