# MEMELORD

A private torment nexus for the discerning image hoarder.

Memelord scrapes the cursed corners of the internet, makes you swipe through the results like a deranged sommelier, and uses your judgement to train a neural network that learns your specific, unjustifiable taste. It then starts pre-sorting new scrapes automatically so you only have to touch the ambiguous ones.

It is a machine that watches you suffer, learns from it, and tries to suffer more efficiently on your behalf.

---

## The loop

1. **Scrape** — pull images from 4chan, Tumblr, Imgur, Pixelfed into an inbox
2. **Review** — work through the queue: score 1–6, star favourites, trash the bad ones
3. **Train** — DINOv2 encodes your rated images; a logistic regression learns your damage

After enough ratings, the classifier starts auto-sorting new scrapes before they reach your queue. High-confidence good matches go straight to corpus. High-confidence bad matches go straight to void. You only see the confusing middle ground.

---

## Docker setup

### 1. Configure

```bash
cp .env.example .env
# set DJANGO_SECRET_KEY (generation hint is inside the file)
# set ALLOWED_HOSTS to your server hostname if not running locally
```

Drop a `config.toml` in the `data/` directory with your sources (see below).

### 2. Run

```bash
docker compose up -d
```

Two containers start from the same image:
- **memelord** — web UI on port 8000, runs migrations on first start
- **qcluster** — background worker for scrape/train jobs; waits for the web service to be healthy before starting

### 3. Create a user

```bash
docker compose run --rm memelord createsuperuser
```

Open `http://localhost:8000` and begin your torment.

Images are served at `/media/` behind Django login — no unauthenticated access to your collection.

### One-off commands

```bash
docker compose run --rm memelord scrape   # scrape now
docker compose run --rm memelord train    # train now
```

---

## Sources (`data/config.toml`)

```toml
[4chan]
boards = ["wg", "a"]

[tumblr]
blogs = ["someblog"]

[imgur]
topics = ["pics"]

[pixelfed]
instance_base = "https://pixelfed.social"
```

Sources can also be managed directly from the **Config** page in the UI — add, toggle, delete, and import from config.toml without restarting. Mark a source NSFW and its images are quarantined to a separate review queue.

Auto-scrape scheduling is also configured from the Config page: set an interval (1–168 hours) and the background worker picks it up immediately.

---

## Review

The review queue shows unscored inbox and corpus images, unseen ones first.

**Score 1–6** to rate an image (moves it to corpus). **Trash** sends it to void. **Star** marks a favourite (higher training weight). **Purge** hard-deletes it from disk and blocks re-download.

| Queue | What's in it | Trash | Rescue |
|---|---|---|---|
| **Review** | inbox + unscored corpus | → void | — |
| **NSFW Review** | is_nsfw=True, unscored | → void | — |
| **Void** | trashed images | purge (permanent) | → corpus |

Keyboard shortcuts: `1`–`6` to score, `f` to favourite, `t` to trash, `p` to purge, `n` to toggle NSFW. Arrow keys navigate prev/next.

---

## Gallery

Scored corpus images (score ≥ filter) browsable with sort (newest / oldest / random), fav-only filter, and `#tag` filter. Tags are assigned per-image from the review queue or gallery.

---

## Releases

Tagged releases are automatically built and pushed to the GitHub Container Registry:

```bash
docker pull ghcr.io/therealbiwhive/memelord:latest
```

---

## Development

```bash
uv sync --extra dev
make run        # Django dev server on :8000
make qcluster   # background worker (separate terminal — required for Train button)
make migrate
make test
```

### Stack

- Django 6 + django-htmx (mobile-first UI, swipe gestures)
- django-q2 (background jobs, SQLite broker — no Redis)
- PyTorch + DINOv2 ViT-B/14 (768-d image embeddings)
- scikit-learn LogisticRegression (the taste oracle)
- Playwright optional: used as fallback for Imgur topic pages and Pixelfed instances that don't serve the API without auth

---

## License

MIT. Not responsible for what you teach it.
