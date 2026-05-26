# MEMELORD

A private torment nexus for the discerning image hoarder.

Memelord scrapes the cursed corners of the internet, makes you swipe through the results like a deranged sommelier, and uses your judgement to train a neural network that learns your specific, unjustifiable taste. It then starts pre-sorting new scrapes automatically so you only have to touch the ambiguous ones.

It is a machine that watches you suffer, learns from it, and tries to suffer more efficiently on your behalf.

---

## The loop

1. **Scrape** — pull images from 4chan, Tumblr, Imgur, Pixelfed, Mastodon into an inbox
2. **Review** — swipe inbox images good/bad/fav, then score the keepers 1–6
3. **Train** — DINOv2 encodes your rated images; a logistic regression learns your damage

After enough ratings the classifier starts auto-sorting new scrapes before they reach your queue. High-confidence good matches go straight to corpus. High-confidence bad matches go straight to void. You only see the confusing middle ground — and a 🎯 toggle on the inbox card lets you actively *target* that middle for fastest model improvement.

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

## Upgrading to 2.0 — read before you deploy

> ⚠️ **The 2.0 migration wipes every image record.** This is intentional. 2.0
> replaces the old `data/inbox|corpus|void/` directory tree with a single flat
> `data/images/` directory and a 0–6 score on each row, so the legacy file paths
> no longer resolve. Migration `0020_p3_flatten_image` therefore `DELETE`s all
> `ratings_image` rows (the delete is **not reversible** — rolling the migration
> back restores the dropped columns but not the data), and the Docker entrypoint
> runs `migrate` on **every container start**.
>
> What this means in practice:
> - Your scores, tags, and NSFW labels for already-rated images are discarded.
> - The old image files under `data/inbox/`, `data/corpus/`, and `data/void/` are
>   left on disk, orphaned — delete them once you're satisfied, or re-scrape.
>
> **Before deploying 2.0 onto a populated instance, back up the DB**, e.g.
> `docker compose run --rm memelord python manage.py dumpdata ratings > backup.json`
> (or copy `data/memelord.db`). A fresh install has nothing to lose and needs no action.

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
accounts = ["@user@pixelfed.social"]

[mastodon]
accounts = ["@user@instance.social"]
```

> **Pixelfed config changed in 2.0.** The old `instance_base = "…"` key is gone —
> Pixelfed now targets specific account handles, just like Mastodon. A config that
> still uses `instance_base` is ignored (a warning is logged on scrape) until you
> switch it to `accounts = [...]`.

Sources can also be managed directly from the **Config** page in the UI — add, toggle, delete, and import from `config.toml` without restarting. Mark a source NSFW and its images are quarantined to a separate review queue.

Auto-scrape scheduling lives on the same Config page: set an interval (1–168 hours) and the background worker picks it up immediately.

New downloads are deduplicated in three layers — SHA-256, perceptual hash, and DINO cosine similarity — so resizes, recompressions, and watermarked clones of images you've already seen never reach the queue.

---

## Curating images

Two flows, depending on whether you're triaging a fresh inbox or polishing the corpus.

### Swipe flow (inbox & NSFW inbox)

Full-screen card, mobile-first, gesture-driven.

| Gesture | Key | Action |
|---|---|---|
| Swipe right | `→` | Good — moves to corpus |
| Swipe left | `←` | Bad — moves to void |
| Swipe up | `↑` | Fav — moves to corpus with high training weight |
| Swipe down | `↓` | Skip |
| — | `N` | Toggle NSFW flag |

Each card shows the classifier's confidence (`P(corpus)` as a percentage), a tag editor with kNN tag suggestions (full-screen modal on mobile, inline on desktop), and a horizontal strip of the six visually-most-similar already-rated images (with score / fav / trash badges) so you can rate consistently and see whether the model's neighborhood actually matches your taste.

Tap the 🎯 button in the meta row to switch from random ordering to **active-learning mode** — the queue surfaces the images the classifier is least sure about, concentrating your ratings where they teach the model the most per click.

The card preloads the next image while you're looking at the current one, so the rate→next swap is instant on mobile.

### Corpus review

A grid of scored corpus images for re-grading, re-tagging, or trashing. Keyboard: `1`–`6` to score, `f` to favourite, `t` to trash, `n` to toggle NSFW, arrows to navigate.

### Void grid

Bulk-review trashed images — rescue back to corpus or permanently **purge** (deletes from disk and blocks re-download by content hash).

---

## Tags

Each image can carry free-form tags. Suggestions auto-populate per image via **kNN** — your own taste tags ("warhammer40k", "cursed") inherited from the visually-nearest already-tagged images via cosine similarity over DINO embeddings. Every new tag you add immediately improves what neighbours can inherit; no model retraining required.

Tap a suggestion pill to apply it, or type a new tag with autocomplete on existing ones. A `#tag` filter on the gallery surfaces everything you've labelled the same way.

The **Tags** page (under the `⋯` menu) lists all tags by image count. Rename inline; renaming to an existing name merges the two. Delete to strip a tag from every image that carries it.

---

## Gallery & stats

The **Gallery** shows scored corpus images with controls for minimum score, sort order (newest / oldest / random), fav-only, and tag filter.

The **Stats** page summarises queue / collection counts, score distribution, source breakdown, tagging coverage (tagged vs untagged corpus), top tags by image count, average inbox dwell time, the last training run's success/failure with error trace, and recent 7-day scrape & rate velocity.

The **Logs** page surfaces background scrape and train output for in-app debugging.

---

## Sharing

Configure Mattermost (personal access token, posts from your own account) or Signal (via signal-cli-rest-api) under Config → Notifications. Then any image gets a share button that posts the image URL to the configured channel / recipients with an optional prefix.

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

- Django 6 + django-htmx (mobile-first UI, swipe gestures, image preload, haptic feedback)
- django-q2 (background jobs, SQLite broker — no Redis)
- PyTorch + DINOv2 ViT-B/14 (768-d image embeddings for taste classifier + dedup + kNN)
- scikit-learn LogisticRegression (the taste oracle + a separate NSFW classifier)
- Playwright optional: used as fallback for Imgur topic pages and Pixelfed instances that don't serve the API without auth

---

## License

MIT. Not responsible for what you teach it.
