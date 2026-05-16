# Memelord Webapp — Plan

## Goal

Replace the CLI + Mastodon posting workflow with a mobile-first Django webapp where the operator manually rates images (good / bad / fav) to build training data for the DINOv2 classifier. Scraping and training stay; Mastodon posting goes away.

**Fresh start**: existing `janulon.db` and `data/corpus/`, `data/void/` are discarded. Everything gets re-scraped and re-rated from zero.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Backend | Django 5.x | Batteries: ORM, auth, admin, static/media |
| Frontend interaction | HTMX | No page reloads without a JS framework; pairs with Django templates |
| Gestures | ~30 lines vanilla JS | Swipe left/right/up/down on mobile |
| Database | SQLite (fresh) | No infra change; Django ORM replaces Peewee |
| Auth | Django `django.contrib.auth` | Single superuser, login required on all views |

Keep entirely: `core/brain.py`, `core/trainer.py`, `core/caption.py`, `retina/`

Replace: `core/db.py` (Peewee → Django ORM), `main.py` (CLI → Django views + management commands)

Remove: `core/mastodon.py`, Mastodon config, `peewee`, `mastodon-py`, `schedule` dependencies

---

## Data Model

Single Django model replaces the Peewee `Image` model. Drop all Mastodon fields.

```python
# ratings/models.py
class Image(models.Model):
    INBOX   = "inbox"
    CORPUS  = "corpus"
    VOID    = "void"
    LOCATION_CHOICES = [(INBOX, "inbox"), (CORPUS, "corpus"), (VOID, "void")]

    content_hash  = models.CharField(primary_key=True, max_length=64)
    file_path     = models.CharField(max_length=2048)       # relative to DATA_DIR
    source_url    = models.CharField(max_length=2048, null=True, blank=True)
    source_label  = models.CharField(max_length=255)
    location      = models.CharField(max_length=32, default=INBOX, choices=LOCATION_CHOICES)
    downloaded_at = models.DateTimeField(auto_now_add=True)
    rated_at      = models.DateTimeField(null=True, blank=True)
    is_favourite  = models.BooleanField(default=False)      # extra weight in training
    file_deleted  = models.BooleanField(default=False)
```

### Rating → location + weight mapping

| User action | `location` | `is_favourite` | Training weight |
|---|---|---|---|
| Good | corpus | False | 1.0 |
| Fav | corpus | True | 3.0 |
| Bad | void | False | 1.0 |

`core/trainer.py` needs a small update: instead of pulling Mastodon engagement weights it reads `is_favourite` from the DB to compute `sample_weight`.

---

## Three Rating Modes

### 1. Mixed — new inbox images
Queue: `location='inbox'`, ordered by `downloaded_at` ascending (oldest first, clears the backlog)

Buttons: **Bad** (left) / **Fav** (up) / **Good** (right)

After rating: image moves to corpus or void, next inbox image loads.

Done state: "Inbox empty — scrape more or train the model."

### 2. Corpus review — re-rate existing good images
Queue: `location='corpus'`, random order

Buttons: **Remove** (left, → void) / **Upgrade Fav** (up, toggle is_favourite) / **Keep** (right, no-op) / **Skip** (down, no-op)

Use case: quality control pass, catching false positives the classifier let through before training was good.

### 3. Trash rescue — recover void images
Queue: `location='void'`, random order

Buttons: **Keep trashed** (left, no-op) / **Rescue Fav** (up, → corpus + fav) / **Rescue Good** (right, → corpus) / **Skip** (down, no-op)

Use case: the classifier (or an earlier bad rating) wrongly rejected something worth keeping.

### Mode switching
A persistent nav strip at the top (always visible):

```
[ Mixed (42) ] [ Corpus (318) ] [ Trash (891) ]
```

Counts update live via HTMX polling or after each rating swap.

---

## UI Layout

```
┌─────────────────────────────┐
│  Mixed (42) Corpus  Trash   │  ← mode nav
├─────────────────────────────┤
│                             │
│                             │
│          [image]            │
│                             │
│                             │
├───────┬─────────┬───────────┤
│ ✕ BAD │  ★ FAV  │  ✓ GOOD  │
└───────┴─────────┴───────────┘
```

- Full-screen image, object-fit: cover
- Three large tap zones (44px min touch target)
- Swipe gestures mirror buttons: left=bad, right=good, up=fav, down=skip (corpus/trash modes)
- After each rating: HTMX swaps `#card` with next image partial — no page reload
- Keyboard shortcuts for desktop: ← bad, → good, ↑ fav, ↓ skip

---

## HTMX Flow

```
GET /rate/inbox/          → renders rate.html with first inbox image
GET /rate/corpus/         → renders rate.html with random corpus image
GET /rate/void/           → renders rate.html with random void image

POST /rate/<hash>/good/   → corpus, rated_at=now, return _next_image.html partial
POST /rate/<hash>/bad/    → void,   rated_at=now, return _next_image.html partial
POST /rate/<hash>/fav/    → corpus, is_favourite=True, rated_at=now, return partial
POST /rate/<hash>/skip/   → no DB change, return next partial

All POSTs:  hx-post, hx-target="#card", hx-swap="outerHTML"
Done state: partial returns done.html fragment with scrape/train buttons
```

---

## Django Project Structure

```
memelord/
├── manage.py
├── memelord/                    # Django project package
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── ratings/                     # Main app
│   ├── models.py
│   ├── views.py                 # mode views + submit_rating + train + scrape
│   ├── urls.py
│   ├── templates/ratings/
│   │   ├── base.html            # mobile viewport, HTMX script, nav
│   │   ├── rate.html            # full-screen card + buttons
│   │   ├── _next_image.html     # HTMX partial (the swappable card)
│   │   └── _done.html           # empty queue state
│   └── management/commands/
│       ├── scrape.py            # thin wrapper around retina/ scrapers
│       └── train.py             # thin wrapper around core/trainer.py
├── core/                        # unchanged except trainer.py ORM update
├── retina/                      # unchanged
├── static/
│   └── swipe.js                 # touch + keyboard → HTMX requests
└── media → data/                # MEDIA_ROOT points at data/
```

---

## Settings

```python
# memelord/settings.py (key additions)
DATA_DIR     = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
MEDIA_ROOT   = DATA_DIR
MEDIA_URL    = "/media/"
WEIGHTS_PATH = DATA_DIR / "Janulon_weights.pkl"
LOGIN_URL    = "/login/"
```

---

## Implementation Phases

### Phase 1 — Django scaffold + auth
- `django-admin startproject memelord .`
- `python manage.py startapp ratings`
- Settings: DATA_DIR, MEDIA_ROOT, `login_required` decorator on all views
- Login/logout views (Django built-in), create superuser
- Smoke-test: `/login/` → redirect → `/rate/inbox/`

### Phase 2 — Image model
- Write `ratings/models.py` and run `makemigrations` / `migrate` (fresh DB)
- Update `core/trainer.py`: replace `db.get_posted_engagement_weights()` with Django ORM query on `Image.objects.filter(location='corpus')`
- Remove `peewee`, `mastodon-py`, `schedule` from `pyproject.toml`; add `django`, `django-htmx`

### Phase 3 — Mixed mode rating UI
- `rate.html`, `_next_image.html`, `_done.html` templates
- Views: `rate_inbox` (GET), `submit_rating` (POST `/rate/<hash>/<action>/`)
- File move logic (inbox → corpus/void dir + DB update)
- `swipe.js` for touch and keyboard events
- Test end-to-end on mobile browser

### Phase 4 — Corpus review + trash rescue modes
- `rate_corpus` and `rate_void` views (same template, different queryset + button config)
- Pass `mode` context var to template to render correct button labels and actions
- Skip action (no DB write, just returns next partial)
- Mode nav strip with live counts

### Phase 5 — Trainer + scraper integration
- `management/commands/train.py` wrapping `core/trainer.py`
- `management/commands/scrape.py` wrapping `retina/` scrapers + DB insert
- `/train/` and `/scrape/` HTMX endpoints (run synchronously, return status snippet)
- "Train model" and "Scrape more" buttons on done screen + stats page

### Phase 6 — Polish
- Stats page: corpus / void / inbox counts, last trained timestamp, top favs grid
- Django admin for bulk image management
- Docker: update `docker-compose.yml` to run `gunicorn memelord.wsgi`
- Update `CLAUDE.md` with new commands (`manage.py runserver`, `manage.py scrape`, etc.)
