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
DJANGO_DEBUG=true uv run python manage.py index_search   # SigLIP2 vectors for the text search, foreground (the in-app job does it in background slices)
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
- `models.py` — `Image` model: `content_hash` (PK), `file_path` (relative to `DATA_DIR`), `score` (0–6, NULL = unrated; 0 = trash), `is_nsfw`, `is_purged`, `rated_at`, `predicted_score`, `embedding`, `embedding_model` (DINOv3 encoder stamp), `search_embedding`, `search_embedding_model` (SigLIP2 stamp, text search only), `phash`, `queue_seen_at`. Plus `Source` (scrape config with pagination `cursor`), `Tag`, `NotificationChannel`, `ReviewThresholds` (singleton: SFW/NSFW hide thresholds + `queue_order`), `LogEntry`.
- `views/` — one module per area; `__init__.py` re-exports every URL-bound view so `urls.py` keeps addressing them as `views.<name>`
  - `common.py` — the taste-model cache (`get_taste_model`, `WEIGHTS_PATH`; the tests patch these here), `taste_prediction` (lazy, persists the result), `flip_nsfw_and_repredict`, `nav_counts` (one aggregate query for the badges), `score_from_post`/`apply_score`, `int_in_range` (form clamping), `index`, `nsfw_toggle`
  - `review.py` — the queues (`_review_qs`, `_review_nsfw_qs`), `_browse_ctx` (card + windowed prev/next), `review_corpus`, `rate_nsfw_corpus`, `score_corpus`/`purge_corpus` (+ NSFW twins via `_score_impl`/`_purge_impl`), `toggle_nsfw`
  - `gallery.py` — `gallery` (plain grid, `?q=` text search, `?similar=<hash>`), `below_cutoff`, lightbox endpoints (`lightbox`, `lightbox_score`, `lightbox_nsfw`, `lightbox_purge`), tag endpoints
  - `jobs.py` — scrape/train session jobs (`trigger_scrape`/`scrape_status`, `trigger_train`/`train_status`, one `_poll_session_job` decision tree keyed by `kind`), `training_ctx`/`scrape_ctx` (nav indicator, used by the context processor), `task_outcome`, the index and re-encode chains (`trigger_search_index` + `search_index_status`, `trigger_taste_reencode` + `taste_reencode_status`, `index_status_ctx`/`reencode_status_ctx`), `job_indicator`, `fresh_start_view`, the log viewer
  - `config.py` — `config_view`, `set_vision_thresholds`, `set_scrape_schedule`, source + channel CRUD, `share_image`
  - `stats.py` — `stats` and its helpers (`_last_train_info`, `_taste_group_breakdown`, `_avg_inbox_hours`)
- `toast.py` — `with_toast(response, message, kind)`: puts a toast into the `HX-Trigger` header as JSON; `toast.js` renders it as text (never HTML)
- `reset.py` — `fresh_start()`: wipe every image row + file, both classifiers and the source cursors; keeps sources, channels, tags, thresholds, users (`CONFIRM_WORD = "RESET"`)
- `context_processors.py` — `app_version`, `notification_config`, `server_toasts` (Django messages → toast JSON), `active_jobs` (training/scrape state from the session plus the index and re-encode chains from OrmQ, for the nav indicator on every page)
- `scraper.py` — orchestrates all `retina/` scrapers (`run`: one `_ScrapeRun` bundle per run, one helper per source kind: `_scrape_simple_source`, `_scrape_board`, `_scrape_account`), deduplicates by SHA256/pHash/embedding, inserts into DB, auto-classifies (`classify_images` re-encodes at most `CLASSIFY_BACKFILL_LIMIT` = 200 stale library rows inline, oldest first; the rest is the re-encode chain's)
- `queue_rules.py` — single source of truth for the review/below-cutoff filters and for the review-queue order: `QueueOrder` (order_by fields, their reverse, and the prev/next window filters, all derived from one `(field, descending)` pair plus an optional SQL expression and its Python twin for a derived key; `content_hash` is the tiebreak in every order), `QUEUE_ORDERS` (`oldest`/`newest` by `downloaded_at`, `shuffle` by `content_hash`, `uncertain` by `|predicted_score − 0.5|` with unpredicted rows last), `normalize_queue_order()` (whitelist), `get_queue_order()`
- `utils.py` — `purge_image()`: hard-delete file from disk + mark `is_purged=True`
- `embeddings.py` — `has_current_embedding()`, `stale_images()`, `taste_vector_counts()`, `reencode_stale_embeddings()` (most recently rated first, then download order; saves per chunk): embedding-generation handling (see below). Plus the re-encode chain, the twin of the index chain in `search.py`: `REENCODE_TASK`, `REENCODE_SLICE_SIZE` (500), `reencode_job_queued()`, `enqueue_reencode_job_if_needed()`, `last_reencode_report()`
- `features.py` — `has_taste_features(image)` (both vectors current) and `taste_features(image)` (the 1536-d DINOv3+SigLIP2 feature via `core.taste.combine_features`); the one gate the trainer, `classify_images` and the lazy view prediction share
- `vector_bank.py` — `VectorBank`: one unit-normalised matrix per embedding generation held in the web process, refreshed when `(rows with current stamp, purged rows)` changes; `rank()` + `first_allowed()` are the whole similarity search
- `search.py` — text search: stale/count helpers for the SigLIP2 generation, `encode_stale_search_embeddings()` (rated first, `only()`, per-row save; `only_rated=True` is the trainer's inline backfill), the index chain (`INDEX_TASK`, `index_job_queued()`, `enqueue_index_job_if_needed()`, `last_index_report()`), `candidate_images(scope, …)` (the one filter both the plain gallery and the ranked modes use), `rank_by_text()`
- `similar.py` — `rank_similar()`: nearest neighbours over the DINOv3 vectors via the taste `VectorBank`
- `tasks.py` — django-q entry points `run_scrape` (enqueues the index chain and the re-encode chain at the end), `run_train`, `run_search_index` (one slice of `INDEX_SLICE_SIZE`, re-enqueues itself while rows remain and the slice encoded something), `run_taste_reencode` (same shape for the DINOv3 generation, slices of `REENCODE_SLICE_SIZE`, log source `reencode`)
- `admin.py` — ImageAdmin with inline thumbnails
- `management/commands/scrape.py` — CLI wrapper over `ratings/scraper.py`
- `management/commands/train.py` — CLI wrapper over `core/trainer.py`
- `management/commands/reencode_embeddings.py` — foreground twin of the re-encode chain (`--limit`, `--chunk-size`, `--batch-size`, `--dry-run`)
- `management/commands/index_search.py` — foreground twin of the search index job (`--limit`, `--dry-run`)
- `management/commands/fresh_start.py` — CLI twin of the Config → Danger zone button (`--yes` skips the typed `RESET` prompt)

**`core/`** — ML logic
- `brain.py` — DINOv3 ViT-B/16 encoder via Hugging Face Transformers at `INPUT_SIZE` = 448 px (768-d embeddings, gated repo → `HF_TOKEN`), classifier load/save. `ENCODER_ID` (`dinov3_vitb16_448`) names the current embedding generation; `encode()` runs batches of 16 (four times the activations of 224 px per image).
- `taste.py` — the taste model (pure, no ORM): `taste_group(source_label, is_nsfw)` + `NSFW_GROUP` (`"(nsfw)"`; the one place that maps a row to its rating category: its source, or the NSFW group when flagged), `TasteModel` (shared classifier + one per taste group, `classifier_for(group)` falls back to the shared one), `fit_classifier()` (the one place with the LogisticRegression settings; balances the liked/disliked side by weight via `balance_class_weights()`), `group_by_source()` + `has_enough_examples()` (10 good + 10 bad = own model), `save_taste_model()`/`load_taste_model()` (stamped pickle, same rule as `brain.load_classifier`).
- `siglip.py` — SigLIP2 ViT-B/16 (`google/siglip2-base-patch16-224`, not gated) for the text search: `get_image_encoder()` (vision tower only, runs through `brain.encode`), `get_text_encoder()` (text tower only, lazily cached in the web process), `encode_text()` (64-token max_length padding, unit norm). `SEARCH_ENCODER_ID` stamps `Image.search_embedding_model`.
- `trainer.py` — reads scored images from Django ORM (good = score ≥ 3, bad = score ≤ 2), applies the per-score weight table, fits the shared classifier on everything and one classifier per taste group (a source, or the NSFW group) that has enough ratings (`_fit_per_source_classifiers`, one log line per group)

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
| Text search | `gallery?q=` | candidate set (`scope=rated`: the gallery filter; `scope=all`: plus every unrated image), ranked by SigLIP2 cosine, best 100 | same lightbox, so backlog hits can be rated on the spot |
| Similar images | `gallery?similar=<hash>` | same candidate set, ranked by DINOv3 cosine to the anchor, anchor excluded | linked from the review card and the lightbox action row |

**Review queue order** (Config → Review queue → Order, stored in `ReviewThresholds.queue_order`): `oldest` (default, download order), `newest`, `shuffle`, or `uncertain` (the images the taste model is least sure about first, i.e. smallest `|predicted_score − 0.5|`, unpredicted ones last: uncertainty sampling, the plain form of active learning). Unseen images (`queue_seen_at IS NULL`) always come first; the option only sorts inside that block. An image is stamped seen when the user moves on from it via prev/next without rating it (the links carry `?left=<hash>`, handled by `_mark_left_image_seen`), not when its card renders, so the card's prev/next and rate-and-advance always agree. `shuffle` orders by `content_hash`, which is unrelated to source and download time, so the sequence is as good as random but stable across requests, and prev/next, the position counter and rate-and-advance keep working (a real `order_by("?")` would re-roll on every request). The queryset order, the reversed order for predecessor lookups and the prev/next window filters all come from one `QueueOrder` object, so they cannot drift apart. Every order ends on `content_hash` as the tiebreak (two downloads in one second, a block of unpredicted rows), and the `uncertain` order's key is an annotation the queue builders add (`QueueOrder.annotations()`), computed identically in Python for the current card (`value_of()`). Contract V1–V12 in `tests/ratings/test_queue_order.py`.

**Review latency** (contract R1–R5 in `tests/ratings/test_review_latency.py`, measured with `tools/measure_review_latency.py`): the image table is wide (two 3 KB vectors per row), so every queue query must stay off the rows. `image_review_queue_idx` (migration 0025: `is_purged, score, is_nsfw, queue_seen_at, downloaded_at, content_hash, predicted_score`) covers the head/neighbour lookups, the position counts and the nav badge aggregate; `file_path` is indexed for the purged-path check in `memelord/media.py`. The two chain counts (`taste_vector_counts`, `index_counts`), which the nav indicator reads on every page while a chain is queued, walk the partial indexes `image_taste_vector_idx` / `image_search_vector_idx` (migration 0026, `WHERE <blob> IS NOT NULL`); SQLite only uses a partial index when the query spells its condition the same way, so these counts filter with `<blob>__isnull=False`, never `exclude(<blob>=None)`. The card preloads the next `PRELOAD_AHEAD` (2) queue images (`preload_paths` from `_browse_ctx`, `<link rel="preload">` in `_review_card.html`), and media/thumbnail responses carry `Cache-Control: private, max-age=31536000, immutable`, so rate-and-advance paints the next picture from the cache. On a 25k-row copy the score POST went from ~266 ms to ~60 ms and "key press → picture" from ~330 ms to ~66 ms. Live (28.5k images, 20 GB host): 4.5 s → 1.3 s with v2.6.0, the remaining second was the two chain counts. If a queue query is added, check its plan with `.explain()`: it must name the index, never `SCAN ratings_image`.

**Auto-scrape timer** (Config → Auto-scrape, `ScrapeSchedule` singleton mirrored into one django-q `Schedule` row by `ratings/schedule_sync.py`): Enable starts the interval from now (`restart_interval=True`), the `post_migrate` bootstrap in `ratings/apps.py` only reconciles the row (`restart_interval=False`) because the Docker entrypoint runs `migrate` on every container start, and `Q_CLUSTER["catch_up"] = False` makes a run missed during downtime fire once, not once per missed interval. Never delete and recreate the Schedule row: a new row is due immediately. Contract V1–V6 in `tests/ratings/test_schedule_sync.py`.

**Text search and similar images** (contract V1–V17 in `tests/ratings/test_search.py`): the query goes through `normalize_query` (whitespace, 200 chars) to the tokenizer only and is echoed via autoescape; `scope` is a whitelist; a ranked mode replaces the sort, hides the sort buttons and falls back to the plain gallery with a toast when it cannot run (missing weights, anchor without a current vector). Each web process holds one `VectorBank` per generation (about 77 MB per 25k images) and the text tower (about 1.1 GB) after the first search. The **search index** is a django-q chain: `run_search_index` encodes one slice of 1000 (rated first), re-enqueues itself while rows remain, and stops when nothing is left or a whole slice encoded nothing (missing/unreadable files → `repair_orphans`); `run_scrape` enqueues the chain at its end, the Config page has the button and the counts, and all status comes from the DB/OrmQ, never the session (the chain has a new task id per slice). Never encode SigLIP inline in the scrape: a 25k-image scrape would hit the 4 h cluster timeout.

Keys `0`–`6` rate and advance in review (`0` = trash); `s` opens the share picker, `n` toggles NSFW, arrows navigate. The same keys work inside the gallery lightbox, where a score keeps the image in view (no advance) and arrows follow the grid order. (The fav star was removed with the 1–6 model; trash returned as score 0.)

### UI conventions (mobile first)
- Every page extends `templates/base.html`; htmx is vendored (`static/vendor/htmx.min.js`), nothing loads from third-party servers. Scripts: `nav.js`, `sheet.js`, `toast.js` on every page; `review.js`, `gallery.js`, `tags.js`, `share-sheet.js` per page.
- The review card and the gallery lightbox share two partials: `_score_actions.html` (score row 0–6 above an action row prev · NSFW · share · purge · next) and `_tag_editor.html`. The review context passes `score_url`/`purge_url`/`nsfw_url`/`hx_target`/`hx_swap`; the lightbox passes `lightbox=True` and steps through the grid in `gallery.js`.
- A response that changes counts also sends the nav badges out of band (`_nav_badges.html`); job-triggering responses send `_nav_job.html` OOB. The nav indicator polls `job_indicator` every 5 s only while a job runs (scrape/train from the session, the search index and the taste re-encode from OrmQ via `active_index` / `active_reencode`).
- Feedback: htmx views return `with_toast(...)` (`HX-Trigger` header); full-page views use Django `messages`, which `server_toasts` turns into the same toast. `toast.js` sets text via `textContent` (OWASP A03).
- Confirmation: exactly one sheet in `base.html`. htmx elements use `hx-confirm` (+ `data-confirm-label`), plain forms use `data-confirm`; `data-confirm-word="RESET"` + `data-confirm-field` demand a typed word, which the view checks again (OWASP A04). Scores, NSFW and tags never ask; purge, delete, clear and fresh start always do.
- Design tokens live in `:root` of `static/app.css`: `--tap: 48px` (minimum touch target), `--score-h`, `--action-h`, `--nav-h`. Hover may recolour but never show or hide anything.
- Guards in `tests/memelord/test_ui_guards.py` enforce all of this (one base template, no `confirm()`/`alert()`, destructive URL refs carry a confirm, destructive views have `login_required` + `require_POST`, no unused CSS class, no hover-only visibility). `tests/memelord/test_nav_contract.py` renders the nav under Hypothesis (exactly one active entry, every area linked once).

### Key settings (`memelord/settings.py`)
- `DATA_DIR` — root for all image files and the SQLite DB (env: `DATA_DIR`, default: `BASE_DIR/data`)
- `CONFIG_PATH` — path to `config.toml` for scraper sources (env: `CONFIG_PATH`)
- `WEIGHTS_PATH` — `DATA_DIR/Janulon_weights.pkl`
- `MEDIA_ROOT` resolves symlinks and refuses files outside itself, so a `DATA_DIR` copy for measurements must copy `images/` (a symlink 404s every picture)
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
`score_corpus` in `views/review.py` sets `image.score` + `image.rated_at` with a single DB write (`common.apply_score`) — files never move. `file_path` is set once at scrape time and never touched. Purging (`purge_image` in `utils.py`) hard-deletes the file and sets `is_purged=True`; the `content_hash` stays in the dedup index so it can't be re-downloaded. Content hashes are the dedup key — `scraper.py` skips any download whose SHA256 already exists in the DB.

### Embedding generations
Two independent generations live on every row: the DINOv3 taste vector (`embedding` / `embedding_model`, stamp `core.brain.ENCODER_ID`) and the SigLIP2 search vector (`search_embedding` / `search_embedding_model`, stamp `core.siglip.SEARCH_ENCODER_ID`). They are never compared with each other; the only place both meet is the taste feature (`core.taste.combine_features`: each block at unit length, DINOv3 first, 1536 values), which the trainer, `classify_images` and the lazy view prediction build through `ratings/features.py`. A row without a current vector of either generation has no taste feature: it is neither trained on nor predicted and keeps `predicted_score` NULL ("show anyway") until the index chain fills the search half. The trainer fetches missing search vectors for its *rated* rows inline (`encode_stale_search_embeddings(only_rated=True)`), never for unrated ones. The NSFW head stays on the DINOv3 vector alone. The rest of this section describes the taste generation; the search generation follows the same stamp discipline with its own stale set (`search.stale_search_images()`) and its own backfill (the index chain / `index_search`).

Every stored vector is stamped with the encoder that produced it (`Image.embedding_model` = `core.brain.ENCODER_ID` at write time). All similarity code (dedup index, similar images, kNN tag suggestions, taste/NSFW prediction) filters on the current stamp; a row with a missing or foreign stamp is "stale" and is re-encoded by the next encoder pass over it (trainer backfill for rated rows, `classify_images` for up to 200 unrated rows per scrape, `manage.py reencode_embeddings`, or the **re-encode chain**: `run_taste_reencode` encodes one slice of 500, most recently rated first, re-enqueues itself while rows remain and stops when nothing is left or a whole slice encoded nothing; `run_scrape` enqueues it at the end, the Config page has the button and the counts, and all status comes from the DB/OrmQ like the index chain). Never re-encode the whole library inline in a scrape or a request: at 448 px a 25k-image library is about twelve hours on the CPU host. Classifier pickles are dicts `{"encoder", "classifier"}`; the taste file additionally carries `"search_encoder"` (the SigLIP2 stamp of its second feature block) and `"per_source": {group: classifier}` keyed by taste group, i.e. source labels and `(nsfw)` (`core/taste.py`; a file without `per_source` loads with no group models, a file from before the NSFW group has no `(nsfw)` key and sends flagged images to the shared model until the next training, a file without `search_encoder` is foreign). `load_classifier` / `load_taste_model` return `None` for a foreign or legacy (bare estimator) file. To switch encoders: change `get_encoder`/`get_transform`, bump `ENCODER_ID`, and add a migration if the old rows need a label (0021 labelled all pre-existing vectors `dinov2_vitb14`; the 2026-10-03 move from 224 to 448 px bumped the stamp to `dinov3_vitb16_448` without a migration, the old stamp simply reads as stale).

### Training weight formula
Positive class is score ≥ 3, negative is score ≤ 2 (score-NULL images are excluded). Per-score weights are symmetric: 5–6 → 3.0, 3–4 → 1.0, 1–2 → 1.0, and 0 (trash) → 3.0 (`_POSITIVE_WEIGHTS` / `_NEGATIVE_WEIGHTS` in `core/trainer.py`). Before the fit, `taste.balance_class_weights()` scales each side to half of the total weight, so the liked and the disliked side always weigh the same and the per-score ratios survive inside a side (sklearn's `class_weight="balanced"` balances by row count, which would let ten 3×-weighted sixes outweigh a hundred ones). Passed as `sample_weight` to `LogisticRegression.fit()`.

### One taste model per source
The library mixes categories with different criteria (jokes, painted miniatures, wallpapers); one hyperplane over all of them learns the category as a shortcut as soon as the positive rate differs per category. So every `source_label` (the bare board / topic / blog / handle, fixed at download) is its own rating category for safe images, and **every NSFW image belongs to the one `(nsfw)` category whatever its source** (`taste.taste_group(source_label, is_nsfw)`, `taste.NSFW_GROUP`; the operator judges NSFW by a criterion of its own in which the origin plays no part, and the parentheses keep the name from colliding with a real source): `trainer.run` fits the shared classifier on all ratings as before and, for every group with at least 10 good and 10 bad rated rows (`taste.MIN_GOOD_PER_SOURCE` / `MIN_BAD_PER_SOURCE`), one classifier on that group's rows only. `TasteModel.classifier_for(group)` picks the group's own classifier or the shared one; `classify_images` and the lazy `_taste_prediction` in the views both go through it, so the dial cutoff applies within each group. Flipping the NSFW mark (`toggle_nsfw`, `lightbox_nsfw`) goes through `_flip_nsfw_and_repredict`, which drops the stored `predicted_score` and recomputes it with the new category's model, so a flagged image never keeps a score its source's model produced. The names `per_source` / `group_by_source` / `SourceGroup` / `MIN_*_PER_SOURCE` stay (the pickle key must stay readable) and mean "category" throughout. The train log says per source which model it got (`taste: own model for 'tg' (12 good, 15 bad)` / `taste: 'wsg' uses the shared model (3 good, 2 bad, needs 10/10)`), and the stats page shows the good/bad counts with an "own model" / "shared" pill. The classifiers work on the 1536-d taste feature (DINOv3 at 448 px + SigLIP2, see "Embedding generations"); the resolution is for the fine structure the criteria depend on (brush work on a miniature, wallpaper texture). Contract V1–V27 in `tests/core/test_taste.py` (V23–V27 are the NSFW category; their ORM side is in `test_trainer.py`, `test_embedding_generation.py`, `test_nsfw_toggle_prediction.py` and `test_stats.py`); the chain tests are `tests/ratings/test_taste_reencode.py`.

### Docker
Single volume `/data` holds everything (DB, weights, images, config). The image installs dependencies with `uv sync --locked` from `uv.lock`, so run `uv lock` after editing `pyproject.toml` or the build fails. Entrypoint runs `migrate` + `collectstatic` on every start then launches gunicorn. One-off commands map as `docker compose run --rm memelord <manage.py subcommand>`.

### Tests
- Mirror source structure: `tests/core/`, `tests/retina/`, `tests/ratings/`, `tests/memelord/`
- Integration tests (real DINOv3 weights, real network): `@pytest.mark.integration`, skipped by default. `tests/core/test_brain_integration.py` loads the real encoder; run it after a torch/transformers bump with `uv run pytest --run-integration tests/core/test_brain_integration.py` (needs `HF_TOKEN` for an account with access to the gated repo).
- Django view/model tests use `django.test.TestCase` (each module calls `django.setup()` at import); run with plain `pytest` (no `pytest-django` plugin). There is no separate test database: tests open the dev DB under `DATA_DIR`, delete rows inside a transaction and roll back, so **apply new migrations locally (`make migrate`) before running the suite** or every DB test fails with `no such column`.
- Mutation testing: `[tool.mutmut]` in `pyproject.toml` is scoped to `core/brain.py`, `core/dedup.py`, `core/taste.py`, `ratings/embeddings.py`, `ratings/features.py`, `core/siglip.py`, `ratings/vector_bank.py`, `ratings/search.py`, `ratings/similar.py`; run command and env vars are in the comment above that section. Run it only when one of those modules changes, not routinely: a full run takes about 35 minutes (555 mutants, one child because the tests share one SQLite file, each child pays the torch/transformers import). Narrow `only_mutate` to the modules that changed and `pytest_add_cli_args_test_selection` to their test modules for the run (restore both afterwards): the full scope with both test dirs crawls at minutes per mutant. Run it against a copy of the DB (`DATA_DIR=/path/to/copy`) so the ordinary suite can run meanwhile, always `rm -rf mutants` before a run whose sources or tests changed, and delete `mutants/` afterwards. To re-check one mutant (a timeout, or a survivor after a new test) without a full run, copy the changed test module into `mutants/tests/...` and run `cd mutants && DJANGO_DEBUG=true DATA_DIR=/path/to/copy HYPOTHESIS_PROFILE=mutmut MUTANT_UNDER_TEST=<id> uv run python -m pytest tests/<module>.py` (a failing test means killed). `MUTANTS.md` records the last run and the disposition of every surviving mutant.

## Code style

### Comments
Every non-trivial function should have a docstring explaining **why** it works the way it does — hidden constraints, design trade-offs, non-obvious decisions, workarounds for specific behaviour. Not *what* the code does (the names do that).

`core/brain.py` is the reference: it explains why ViT-B/16 was chosen, why `eval()` is required, why pickle is used, why float32 blobs instead of a native array type.

One-liners are fine for simple helpers. Multi-paragraph is fine when there is genuinely important context. Never strip existing docstrings.
