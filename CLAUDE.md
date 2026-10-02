# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
uv sync                      # install all deps (the dev group is included by default)
make run                     # django dev server on :8000
make migrate                 # makemigrations + migrate
make superuser               # create admin user (interactive)
make test                    # full test suite
make test-fast               # skip @pytest.mark.integration tests
make lint                    # ruff check (config in pyproject.toml)
```

### Data tasks (CLI equivalents of the in-app buttons)
```bash
make scrape                  # scrape all sources from config.toml → data/images/
make train                   # encode rated images with DINOv3, fit classifier, save weights
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

The Makefile exports `DJANGO_DEBUG=true` for every target (an explicit value in your shell wins), so plain `make run` / `make migrate` work. Use the prefix only when calling `manage.py` directly.

### Single test
```bash
uv run pytest tests/core/test_brain.py::test_is_image_path_true_for_image_extensions
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
Memelord is a mobile-first Django webapp for curating images with a personal taste classifier. The operator scrapes images from the web, scores each one 1–6, and uses those scores to train a DINOv3+LogisticRegression classifier that learns to predict taste.

### Three-phase loop
1. **Scrape** (`ratings/scraper.py` + `retina/`) — pull images from configured sources into `data/images/`
2. **Rate** (Django webapp) — give each unscored image a 1–6 score (a pure DB write; no file moves)
3. **Train** (`core/trainer.py`) — encode all scored images with DINOv3, fit LogisticRegression with score-weighted samples

### Module map

**`ratings/`** — Django app (the new core)
- `models.py` — `Image` model: `content_hash` (PK), `file_path` (relative to `DATA_DIR`), `score` (0–6, NULL = unrated; 0 = trash), `is_nsfw`, `is_purged`, `rated_at`, `predicted_score`, `embedding`, `embedding_model` (encoder stamp), `phash`. Plus `Source` (scrape config with pagination `cursor`), `Tag`, `NotificationChannel`, `ReviewThresholds`, `LogEntry`.
- `views.py` — review/score views (`review_corpus`, `score_corpus`, `purge_corpus`), `below_cutoff`, `gallery`, lightbox endpoints (`lightbox`, `lightbox_score`, `lightbox_nsfw`, `lightbox_purge`), `stats`, `trigger_scrape`, `trigger_train`, `job_indicator`, `fresh_start_view`, channel + source CRUD
- `toast.py` — `with_toast(response, message, kind)`: puts a toast into the `HX-Trigger` header as JSON; `toast.js` renders it as text (never HTML)
- `reset.py` — `fresh_start()`: wipe every image row + file, both classifiers and the source cursors; keeps sources, channels, tags, thresholds, users (`CONFIRM_WORD = "RESET"`)
- `context_processors.py` — `app_version`, `notification_config`, `server_toasts` (Django messages → toast JSON), `active_jobs` (training/scrape state for the nav indicator on every page)
- `scraper.py` — orchestrates all `retina/` scrapers, deduplicates by SHA256/pHash/embedding, inserts into DB, auto-classifies
- `utils.py` — `purge_image()`: hard-delete file from disk + mark `is_purged=True`
- `embeddings.py` — `has_current_embedding()`, `stale_images()`, `reencode_stale_embeddings()`: embedding-generation handling (see below)
- `admin.py` — ImageAdmin with inline thumbnails
- `management/commands/scrape.py` — CLI wrapper over `ratings/scraper.py`
- `management/commands/train.py` — CLI wrapper over `core/trainer.py`
- `management/commands/reencode_embeddings.py` — bulk re-encode of rows whose vector is missing or from an older encoder
- `management/commands/fresh_start.py` — CLI twin of the Config → Danger zone button (`--yes` skips the typed `RESET` prompt)

**`core/`** — ML logic
- `brain.py` — DINOv3 ViT-B/16 encoder via Hugging Face Transformers (768-d embeddings, gated repo → `HF_TOKEN`), classifier load/save. `ENCODER_ID` names the current embedding generation.
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

Keys `0`–`6` rate and advance in review (`0` = trash); `s` opens the share picker, `n` toggles NSFW, arrows navigate. The same keys work inside the gallery lightbox, where a score keeps the image in view (no advance) and arrows follow the grid order. (The fav star was removed with the 1–6 model; trash returned as score 0.)

### UI conventions (mobile first)
- Every page extends `templates/base.html`; htmx is vendored (`static/vendor/htmx.min.js`), nothing loads from third-party servers. Scripts: `nav.js`, `sheet.js`, `toast.js` on every page; `review.js`, `gallery.js`, `tags.js`, `share-sheet.js` per page.
- The review card and the gallery lightbox share two partials: `_score_actions.html` (score row 0–6 above an action row prev · NSFW · share · purge · next) and `_tag_editor.html`. The review context passes `score_url`/`purge_url`/`nsfw_url`/`hx_target`/`hx_swap`; the lightbox passes `lightbox=True` and steps through the grid in `gallery.js`.
- A response that changes counts also sends the nav badges out of band (`_nav_badges.html`); job-triggering responses send `_nav_job.html` OOB. The nav indicator polls `job_indicator` every 5 s only while a job runs.
- Feedback: htmx views return `with_toast(...)` (`HX-Trigger` header); full-page views use Django `messages`, which `server_toasts` turns into the same toast. `toast.js` sets text via `textContent` (OWASP A03).
- Confirmation: exactly one sheet in `base.html`. htmx elements use `hx-confirm` (+ `data-confirm-label`), plain forms use `data-confirm`; `data-confirm-word="RESET"` + `data-confirm-field` demand a typed word, which the view checks again (OWASP A04). Scores, NSFW and tags never ask; purge, delete, clear and fresh start always do.
- Design tokens live in `:root` of `static/app.css`: `--tap: 48px` (minimum touch target), `--score-h`, `--action-h`, `--nav-h`. Hover may recolour but never show or hide anything.
- Guards in `tests/memelord/test_ui_guards.py` enforce all of this (one base template, no `confirm()`/`alert()`, destructive URL refs carry a confirm, destructive views have `login_required` + `require_POST`, no unused CSS class, no hover-only visibility). `tests/memelord/test_nav_contract.py` renders the nav under Hypothesis (exactly one active entry, every area linked once).

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

### Embedding generations
Every stored vector is stamped with the encoder that produced it (`Image.embedding_model` = `core.brain.ENCODER_ID` at write time). All similarity code (dedup index, similar images, kNN tag suggestions, taste/NSFW prediction) filters on the current stamp; a row with a missing or foreign stamp is "stale" and is re-encoded by the next encoder pass over it (trainer backfill for rated rows, `classify_images` for unrated rows, or `manage.py reencode_embeddings`). Classifier pickles are dicts `{"encoder", "classifier"}`; `load_classifier` returns `None` for a foreign or legacy (bare estimator) file. To switch encoders: change `get_encoder`/`get_transform`, bump `ENCODER_ID`, and add a migration if the old rows need a label (0021 labelled all pre-existing vectors `dinov2_vitb14`).

### Training weight formula
Positive class is score ≥ 3, negative is score ≤ 2 (score-NULL images are excluded). Per-score weights are symmetric: 5–6 → 3.0, 3–4 → 1.0, 1–2 → 1.0, and 0 (trash) → 3.0 (`_POSITIVE_WEIGHTS` / `_NEGATIVE_WEIGHTS` in `core/trainer.py`). Passed as `sample_weight` to `LogisticRegression.fit()`.

### Docker
Single volume `/data` holds everything (DB, weights, images, config). The image installs dependencies with `uv sync --locked` from `uv.lock`, so run `uv lock` after editing `pyproject.toml` or the build fails. Entrypoint runs `migrate` + `collectstatic` on every start then launches gunicorn. One-off commands map as `docker compose run --rm memelord <manage.py subcommand>`.

### Tests
- Mirror source structure: `tests/core/`, `tests/retina/`, `tests/ratings/`, `tests/memelord/`
- Integration tests (real DINOv3 weights, real network): `@pytest.mark.integration`, skipped by default. `tests/core/test_brain_integration.py` loads the real encoder; run it after a torch/transformers bump with `uv run pytest --run-integration tests/core/test_brain_integration.py` (needs `HF_TOKEN` for an account with access to the gated repo).
- Django view/model tests use `django.test.TestCase` (each module calls `django.setup()` at import); run with plain `pytest` (no `pytest-django` plugin). There is no separate test database: tests open the dev DB under `DATA_DIR`, delete rows inside a transaction and roll back, so **apply new migrations locally (`make migrate`) before running the suite** or every DB test fails with `no such column`.
- Mutation testing: `[tool.mutmut]` in `pyproject.toml` is scoped to `core/brain.py`, `core/dedup.py`, `ratings/embeddings.py`; run command and env vars are in the comment above that section. Run it only when one of those three modules changes, not routinely: a full run takes about 35 minutes (555 mutants, one child because the tests share one SQLite file, each child pays the torch/transformers import). Run it against a copy of the DB (`DATA_DIR=/path/to/copy`) so the ordinary suite can run meanwhile, always `rm -rf mutants` before a run whose sources or tests changed, and delete `mutants/` afterwards. `MUTANTS.md` records the last run and the disposition of every surviving mutant.

## Code style

### Comments
Every non-trivial function should have a docstring explaining **why** it works the way it does — hidden constraints, design trade-offs, non-obvious decisions, workarounds for specific behaviour. Not *what* the code does (the names do that).

`core/brain.py` is the reference: it explains why ViT-B/16 was chosen, why `eval()` is required, why pickle is used, why float32 blobs instead of a native array type.

One-liners are fine for simple helpers. Multi-paragraph is fine when there is genuinely important context. Never strip existing docstrings.
