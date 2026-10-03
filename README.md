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
docker compose run --rm memelord scrape        # scrape now
docker compose run --rm memelord train         # train now
docker compose run --rm memelord index_search  # build the text-search index in the foreground
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

## Upgrading to 2.3 — one taste model per source, 448 px vectors

2.3 changes how taste is learned and what it is learned from:

- **One rating category per source.** Every scrape source (board, topic, blog,
  handle) is its own category. Training fits the shared classifier as before
  and, for every source with at least 10 liked (score ≥ 3) and 10 disliked
  (score ≤ 2, trash included) images, one classifier on that source's ratings
  only. Predictions come from the source's own model when it has one, else from
  the shared one, so "a good /b/ image" and "a good miniature" stop competing on
  one scale. The Stats page shows liked/disliked counts per source and whether
  it has its own model; the train log says so per source.
- **Two vectors per image.** The taste feature is the DINOv3 vector plus the
  SigLIP2 search vector. An image is trained on and predicted only when it has
  both; until the search index has reached it, it shows in review without a
  prediction. Keep the search index running (Config → Search index).
- **DINOv3 at 448 px.** The encoder now sees every image at 448×448 instead of
  224×224, for the fine structure the criteria depend on (brush work on a
  miniature, the texture of a wallpaper). Every stored DINOv3 vector is from the
  old resolution and counts as stale: dedup by vector, similar images and kNN
  tag suggestions work only over re-encoded images until the library is
  through. SHA-256 and pHash dedup are unaffected.

After deploying:

1. Press **Re-encode now** under Config → Taste vectors (the chain also starts
   by itself after the next scrape). It runs in the background worker in slices
   of 500, most recently rated images first, about 15 minutes per slice on a
   CPU (roughly 1.7 s per image, so a 25,000-image library takes about twelve
   hours); scrapes get their turn between slices. The nav indicator shows the
   progress.
2. Press **Train** once the rated images are through (the first slice or two).
   Training re-encodes any rated image the chain has not reached yet, fetches
   missing search vectors for rated images, and writes the taste and NSFW
   models with the new stamp. Classifier files from the old resolution are
   ignored until then; existing predictions stay in place and are replaced by
   the classification pass after training.

Expect a worker running at full CPU for the duration of the chain and about
four times the per-image RAM of 224 px (the batch size dropped from 32 to 16).

---

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

### Text search

Type what you are looking for into the search box above the gallery grid ("cat on a skateboard", "Katze auf Skateboard": the model is multilingual). Images are ranked by how well they match the text, the best 100 are shown, and the min-score, tag and NSFW filters still apply. Switch the **scope** from `rated` to `all` to search the unrated backlog too; a hit opens in the same lightbox, so you can rate it on the spot.

Behind it is a second embedding model, SigLIP2 (`google/siglip2-base-patch16-224`, no token needed): it maps images and text into one vector space, so nothing has to be classified or labelled first. Every image needs its SigLIP2 vector once. The **Search index** block on the Config page shows how many images have one and starts the index job; the job also starts by itself after every scrape. It runs in the background worker in slices of 1000 images (about 10 minutes each on a CPU, roughly 2 images per second), rated images first, so the gallery is searchable after the first slice while a large backlog fills in behind it. A 25,000-image library takes about 3.5 hours once. The first run downloads the 1.5 GB weights into the shared `hf-cache` volume. Expect about 1.3 GB of RAM per web worker after the first search (the text model plus the vector matrices) and about 1 GB for the worker.

### Similar images

Every image in the review card and in the gallery lightbox has a **similar** action. It opens the gallery with the images closest to it by DINOv3 cosine similarity, across the whole library by default (switch the scope to `rated` to stay in the gallery). This needs no extra model: it reuses the taste embeddings that the classifier, dedup and kNN tag suggestions already share.

---

## Tags

Each image can carry free-form tags. Suggestions auto-populate per image via **kNN** — your own taste tags ("warhammer40k", "cursed") inherited from the visually-nearest already-tagged images via cosine similarity over DINO embeddings. Every new tag you add immediately improves what neighbours can inherit; no model retraining required.

Tap a suggestion pill to apply it, or type a new tag with autocomplete on existing ones. A `#tag` filter on the gallery surfaces everything you've labelled the same way.

The **Tags** page (under the `⋯` menu) lists all tags by image count. Rename inline; renaming to an existing name merges the two. Delete to strip a tag from every image that carries it.

---

## Training

Positive class is score ≥ 3, negative is score ≤ 2; unrated images are excluded. Scores 5–6 and trash (0) carry weight 3.0, everything else 1.0, so your strongest opinions pull hardest on the decision boundary; the liked and the disliked side are then balanced to equal total weight. Training fits one shared model on all ratings and one model per source with at least 10 liked and 10 disliked images (see "Upgrading to 2.3"). Trigger it from the **Stats** page or with `make train`. A separate NSFW classifier on the DINOv3 vectors alone feeds the NSFW queue.

## Stats & logs

The **Stats** page summarises queue sizes (to rate / gallery / below cutoff), score distribution, source breakdown (with liked/disliked counts and whether the source has its own taste model), tagging coverage, top tags, average time from scrape to rating, the last training run (with error trace on failure), and 7-day scrape & rate velocity.

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

The Makefile exports `DJANGO_DEBUG=true`, so the insecure default secret key is accepted for local runs. After changing `pyproject.toml` run `uv lock` — the Docker build installs with `uv sync --locked` and fails on a stale lock. To upgrade everything, run `uv lock --upgrade`, re-run the tests, and run the opt-in model tests: `uv run pytest --run-integration tests/core/test_brain_integration.py` (needs `HF_TOKEN` with access to the gated repo) and `uv run pytest --run-integration tests/core/test_siglip_integration.py` (downloads the 1.5 GB SigLIP2 weights, no token).

### Stack

- Python 3.14, Django 6.1 + django-htmx (mobile-first, HTMX-driven UI)
- django-q2 (background jobs, SQLite broker — no Redis)
- PyTorch 2.14 + DINOv3 ViT-B/16 via Hugging Face Transformers (768-d image embeddings for the taste classifier, dedup, kNN tag suggestions and similar images)
- SigLIP2 ViT-B/16 (text-aligned 768-d embeddings for the gallery text search; indexed by a background job)
- scikit-learn LogisticRegression (the taste oracle + a separate NSFW classifier)
- Playwright optional: fallback for Imgur topic pages that refuse the plain HTTP scraper

---

## License

MIT. Not responsible for what you teach it.
