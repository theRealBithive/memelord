# Memelord 2.0 — Overhaul Plan

## Vision

Replace the binary corpus/trash mental model with a pure 1–6 numeric taste score. Replace the single-channel share button with named, multi-channel destinations. Remove accumulated dead code, sanity-check the test suite, and validate the Pixelfed scraper. The tag system and gallery are good — keep them, polish the edges.

---

## 1. Rating Model Overhaul

### 1a. Flat filesystem — no more inbox/corpus/void directories

2.0 is a clean break. The filesystem separation was only meaningful when `location` encoded the rating — `corpus/` = good, `void/` = bad. With scores in the DB, the file path just needs to point to the file; it doesn't need to encode any meaning.

- All images scrape to `data/images/<hash>.<ext>`. One flat directory.
- `Image.location` field is **deleted entirely**. `score IS NULL` means unrated; `score IS NOT NULL` means rated. No other distinction is needed.
- `_move_image()` and `ratings/utils.py`'s move logic are **deleted entirely**. Rating an image is a pure DB write — no filesystem operation, no `file_deleted` edge cases.
- `file_path` is set once at scrape time and never touched again.
- Trainer queries change from `location__in=[CORPUS]` to `score__gte=3` (positive) and `score__lte=2` (negative).
- **No migration needed** — 2.0 is a fresh start. Existing data is wiped and re-scraped.

**Remove:** `inbox/`, `corpus/`, `void/` directories, `Image.location` field, `_move_image()`, `ratings/utils.py` (if move logic is its only content), `file_deleted` field, `is_purged` field, `void_seen_at`, `inbox_seen_at`, `corpus_seen_at` timestamp fields, all `location=` queryset filters throughout views.

### 1b. Remove `is_favourite` — score=6 is the favourite

`is_favourite` is redundant with 1–6 scoring. A score of 6 IS a favourite. The only reason the field existed was the old good/fav binary didn't have enough resolution.

- No migration needed — fresh start in 2.0.
- Remove `is_favourite` from model, views, gallery `fav_only` filter, share logic, and training.
- Update training weights: `score 5–6 → weight 3.0`, `score 3–4 → weight 1.0`, `score 1–2 → weight 0.0` (excluded from positive training set, treated as negative examples). This makes the score directly meaningful to the ML model instead of the binary fav signal.

### 1c. User-defined visibility cutoff (already 80% there)

The existing `ReviewThresholds` sfw/nsfw mechanism is solid. In 2.0:

- Rename/reframe in the UI as "visibility cutoff" — "show me images scored ≥ X".
- Gallery `min_score` filter already does this — expose it prominently in the UI.
- Add a **Below Cutoff** view (replaces void grid) that shows all scored images with `score < cutoff`. From there the user can re-score upward or purge. This is the mental replacement for "rescue from trash" — just re-score the image.

---

## 2. Multi-Channel Share UX

### 2a. New `NotificationChannel` model (replaces `NotificationConfig` singleton)

```python
class NotificationChannel(models.Model):
    name = models.CharField(max_length=100, unique=True)   # "Aurea", "TownSquare", "Simon"
    service = models.CharField(max_length=20, choices=[("mattermost", "Mattermost"), ("signal", "Signal")])
    enabled = models.BooleanField(default=True)
    # Mattermost
    mm_base_url = models.CharField(max_length=255, blank=True)
    mm_token = models.CharField(max_length=255, blank=True)
    mm_channel_id = models.CharField(max_length=64, blank=True)
    mm_message_prefix = models.CharField(max_length=255, blank=True)
    # Signal
    signal_api_url = models.CharField(max_length=255, blank=True)
    signal_sender = models.CharField(max_length=32, blank=True)
    signal_recipients = models.TextField(blank=True)      # comma-separated
    signal_message_prefix = models.CharField(max_length=255, blank=True)
```

**Migration from `NotificationConfig`:** if Mattermost was configured, create a channel named "Mattermost"; if Signal was configured, create a channel named "Signal". User can rename them immediately after upgrading.

### 2b. Share UI redesign

Replace the auto-fire share button with a named channel picker:

- Share button opens a compact popover (or inline drawer) listing all `enabled=True` channels as labelled checkboxes.
- Default: all channels checked (or remember last session selection per image).
- One submit fires sends to all selected channels.
- Toast: "Sent to Aurea, TownSquare" on success, per-channel errors if any fail.
- Mobile-friendly: the picker must be reachable with a thumb in single-hand use.

### 2c. `notifiers.py` refactor

`send_to_mattermost(cfg, ...)` and `send_to_signal(cfg, ...)` now accept a `NotificationChannel` instead of `NotificationConfig`. Function bodies are unchanged — only the field names on `cfg` adjust. No logic changes.

### 2d. Config page channel list

Replace the single Mattermost + Signal block with a channel list: add / edit / delete named channels. Each channel card has its own enable toggle, service selector, and service-specific fields (collapsed by default, expandable inline).

---

## 3. Pixelfed Scraper Audit

`retina/pixelfed.py` currently fetches a public timeline from an instance URL. This pulls boosts and posts from any account on the instance — including book covers and unrelated content.

- Audit what the current scraper actually fetches (instance public timeline vs. local timeline vs. specific account).
- Add support for targeting a specific account handle (e.g. `@user@pixelfed.social`) within the `Source.name` field — already used for Mastodon handles, same pattern.
- Add optional hashtag filter within the instance.
- Test against at least one real Pixelfed account and document expected output.
- If the current URL-only mode is fundamentally unreliable, deprecate it and require account-level config.

---

## 4. Dead Code Removal

Audit and remove after the model changes land:

- All void/trash views, templates, URL patterns.
- `is_favourite` field and all references once the migration backfill is done.
- `void_seen_at`, `inbox_seen_at`, `corpus_seen_at` timestamp fields — only `rated_at` and `queue_seen_at` are needed.
- `NotificationConfig` model + singleton logic once `NotificationChannel` is live.
- `ratings/tests.py` (empty stub — all real tests live in `tests/`).
- `knn_suggestions.py` management command — review if still needed or superseded by in-view kNN.
- `queue_rules.py`: `AUTO_PROMOTE_THRESHOLD` / `AUTO_TRASH_THRESHOLD` logic routes inbox → void, which no longer exists. The auto-promote-to-corpus path stays; auto-trash must become "score 1 and move to corpus".
- Old `config.toml` source-import button — sources are fully DB-managed; the import path is a one-time bootstrap that confuses returning users. Document it as a CLI-only first-run tool, remove from the UI.

---

## 5. Test Suite Sanity Check

Current coverage is solid for core ML and some feature slices but has real gaps:

**Update existing tests:**
- `test_trash.py` → replace with `test_below_cutoff.py` covering score-based filtering and the Below Cutoff view.
- `test_notifiers.py` → update for `NotificationChannel` instead of `NotificationConfig`.
- `test_vision_thresholds.py` → keep, update descriptions to match new naming ("visibility cutoff").

**Add new tests:**
- View tests for score submission (`POST /score/<hash>`) + navigation advancement.
- View tests for the Below Cutoff view (filtering, re-scoring from there).
- `test_channels.py` — `NotificationChannel` CRUD, share dispatch to multiple channels, partial failure handling (one channel fails, others succeed).
- Test for updated training weight formula (score→weight mapping in `core/trainer.py`).
- `test_pixelfed.py` — update for account-level scraping once that's fixed.

**Run target:** `pytest -v -m "not integration"` must be green after each phase before merging.

---

## 6. UX Polish

These are alongside the relevant feature phase, not separate work:

- **Keyboard shortcuts:** keys `1`–`6` score and advance in review (verify works cleanly with score-only model). `s` opens the share picker. `n` toggles NSFW. `d` = purge (hard delete). Remove shortcuts tied to the old good/fav/bad model.
- **Gallery pagination:** the current 500-item hard DOM cap is a bottleneck at scale — add cursor-based pages or infinite scroll so large corpora don't degrade page load.
- **Stats page:** update to make score distribution the primary chart; remove void-count stat once void is gone. Add a "below cutoff" count instead.
- **Config page copy:** rename all instances of "Trash" → "Below Cutoff", "Void" → "Below Cutoff" in labels and placeholder text.
- **Review card:** remove the fav-star button once `is_favourite` is gone — pressing `6` is the equivalent.
- **NSFW/SFW dual queue:** working well, keep it. Make sure SFW and NSFW threshold dials survive the model rename cleanly.
- **CLAUDE.md update:** reflect new architecture once 2.0 ships (remove void references, add NotificationChannel, update rating model docs).

---

## Implementation Order

Execute in phases, each mergeable independently:

Each phase ends with: **run `pytest -v -m "not integration"`**, delete or update any tests that no longer apply, add tests for new behaviour, and refactor anything the phase exposed as awkward. No phase is done until the suite is green and the new code is clean.

| Phase | Scope | Risk |
|---|---|---|
| **P1** | `NotificationChannel` model + config UI + notifiers refactor | Low — additive |
| **P2** | Share UI redesign (channel picker popover) | Low — UI only, depends on P1 |
| **P3** | Rating model overhaul: drop `location`, `is_favourite`, all move logic; flat `data/images/`; Below Cutoff view; fresh DB | High — do on a branch, wipe data first |
| **P4** | Training weight formula update (score → weight) | Low — isolated in `core/trainer.py` |
| **P5** | Pixelfed scraper audit + account-handle targeting | Medium — needs real-world testing |
| **P6** | Dead code sweep | Low — cleanup only |
| **P7** | Final test pass + full refactor review | Low |

---

## What Stays Unchanged

- Tag system (working well).
- Gallery (working well) — only add pagination and remove `fav_only` filter once `is_favourite` is gone.
- Core ML pipeline (`core/brain.py`, `core/trainer.py` structure) — only the weight formula changes.
- NSFW/SFW split queue — keep.
- Django-Q background training, scrape scheduling — keep as-is.
- `core/dedup.py`, `core/phash.py` — keep as-is.
- All retina scrapers except Pixelfed audit.
- kNN similar-image panel in review card — keep.
