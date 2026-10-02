# MEMELORD

A private torment nexus for the discerning image hoarder.

Memelord scrapes the cursed corners of the internet, makes you swipe through the results like a deranged sommelier, and uses your judgement to train a neural network that learns your specific, unjustifiable taste. It then starts pre-sorting new scrapes automatically so you only have to touch the ambiguous ones.

It is a machine that watches you suffer, learns from it, and tries to suffer more efficiently on your behalf.

---

## The loop

1. **Scrape** — pull images from 4chan, Tumblr, Imgur, Pixelfed and Mastodon into one flat `data/images/` directory
2. **Review** — give every new image a score from 0 (trash) to 6 (favourite)
3. **Train** — DINOv3 encodes your rated images; a logistic regression learns your damage

After enough ratings the classifier predicts a score for every new download. Set a visibility cutoff (Config → Review thresholds, separately for the SFW and NSFW queues) and anything the model rates below it stays out of your review queue until you lower the cutoff again. Nothing is moved or deleted by a prediction, so a wrong guess never costs you an image.

---

## Docker setup

### 1. Configure

```bash
cp .env.example .env
# set DJANGO_SECRET_KEY (generation hint is inside the file)
# set ALLOWED_HOSTS to your server hostname if not running locally
# set HF_TOKEN: DINOv3 is a gated model — accept the licence once on
#   https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m and
#   create a read token in your Hugging Face settings
```

Drop a `config.toml` in the `data/` directory with your sources (see below).

### 2. Run

```bash
docker compose up -d
```

Two containers start from the same image:
- **memelord** — web UI on port 8000, runs migrations on every start
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

## Upgrading to DINOv3 — one Train run migrates the library

The encoder moved from DINOv2 ViT-B/14 to DINOv3 ViT-B/16. Every stored
embedding carries the name of the encoder that produced it; vectors from the
old encoder are never compared with new ones, so nothing is mixed silently.
Until re-encoded, old vectors drop out of the cosine dedup layer and of the
similar-image / kNN features. SHA-256 and pHash dedup are unaffected.

After deploying: set `HF_TOKEN`, then press **Train** once (or run
`docker compose run --rm memelord reencode_embeddings` on a GPU host for a
progress bar). Training re-encodes every rated image, the classification pass
that follows re-encodes the unrated ones, and both classifiers are retrained.
Classifier files from the old encoder are ignored until then; existing
predictions stay in place and are overwritten by the first training run.

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

The score is the whole model: unrated, `0` = trash, `1`–`6` = your taste scale. A score is a single database write — files never move. Trash keeps the file on disk as the strongest negative training example; **purge** is the only action that deletes a file, and its content hash stays blacklisted so it can't be re-downloaded.

### Review

One full-screen card at a time, mobile-first, with separate SFW and NSFW queues.

| Key | Action |
|---|---|
| `0` | Trash — kept as a negative example, advance |
| `1`–`6` | Score, advance |
| `n` | Toggle the NSFW flag |
| `s` | Open the share sheet |
| `←` / `→` | Previous / next image |

Each card carries a tag editor with kNN tag suggestions (full-screen modal on mobile, inline on desktop) and a purge button.

### Below Cutoff

Everything you scored `0`–`2`, as a grid. Changed your mind? Re-score upward from the lightbox. Sure about it? Purge. Rescuing an image is just giving it a better score.

### Gallery

Everything at or above the minimum score you pick, sortable newest / oldest / random and filterable by `#tag`. The lightbox supports swipe navigation and lets you re-score, re-tag, purge or share.

---

## Tags

Each image can carry free-form tags. Suggestions auto-populate per image via **kNN** — your own taste tags ("warhammer40k", "cursed") inherited from the visually-nearest already-tagged images via cosine similarity over DINO embeddings. Every new tag you add immediately improves what neighbours can inherit; no model retraining required.

Tap a suggestion pill to apply it, or type a new tag with autocomplete on existing ones. A `#tag` filter on the gallery surfaces everything you've labelled the same way.

The **Tags** page (under the `⋯` menu) lists all tags by image count. Rename inline; renaming to an existing name merges the two. Delete to strip a tag from every image that carries it.

---

## Training

Positive class is score ≥ 3, negative is score ≤ 2; unrated images are excluded. Scores 5–6 and trash (0) carry weight 3.0, everything else 1.0, so your strongest opinions pull hardest on the decision boundary. Trigger it from the **Stats** page or with `make train`. A separate NSFW classifier on the same embeddings feeds the NSFW queue.

## Stats & logs

The **Stats** page summarises queue sizes (to rate / gallery / below cutoff), score distribution, source breakdown, tagging coverage, top tags, average time from scrape to rating, the last training run (with error trace on failure), and 7-day scrape & rate velocity.

The **Logs** page surfaces background scrape and train output for in-app debugging.

---

## Sharing

Configure one or more channels — Mattermost (personal access token, posts from your own account) or Signal (via signal-cli-rest-api) — under Config → Notifications. The share sheet (`s` in review, or the share button in the gallery lightbox) posts the image to the channel you pick, with an optional message prefix.

---

## Releases

Tagged releases are automatically built and pushed to the GitHub Container Registry:

```bash
docker pull ghcr.io/therealbiwhive/memelord:latest
```

---

## Development

```bash
uv sync         # runtime + dev dependencies, exactly as pinned in uv.lock
make run        # Django dev server on :8000
make qcluster   # background worker (separate terminal — required for the Scrape/Train buttons)
make migrate
make test
make lint       # ruff
```

The Makefile exports `DJANGO_DEBUG=true`, so the insecure default secret key is accepted for local runs. After changing `pyproject.toml` run `uv lock` — the Docker build installs with `uv sync --locked` and fails on a stale lock. To upgrade everything, run `uv lock --upgrade`, re-run the tests, and run the opt-in model test: `uv run pytest --run-integration tests/core/test_brain_integration.py` (needs `HF_TOKEN` with access to the gated repo).

### Stack

- Python 3.14, Django 6.1 + django-htmx (mobile-first, HTMX-driven UI)
- django-q2 (background jobs, SQLite broker — no Redis)
- PyTorch 2.14 + DINOv3 ViT-B/16 via Hugging Face Transformers (768-d image embeddings for the taste classifier, dedup and kNN tag suggestions)
- scikit-learn LogisticRegression (the taste oracle + a separate NSFW classifier)
- Playwright optional: fallback for Imgur topic pages that refuse the plain HTTP scraper

---

## License

MIT. Not responsible for what you teach it.
