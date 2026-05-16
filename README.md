# JANULON

> "The machine does not need to see. It only needs to feel."

Janulon is a subjective aesthetic engine. Unlike generative AI (which creates new noise), Janulon is a curatorial AI designed to filter the digital ocean for specific visual frequencies.

It uses Meta's DINOv2 (self-supervised vision transformer) to map images into high-dimensional vector space, then applies a custom-trained linear probe to determine if a new image aligns with the operator's specific taste. Unlike language-supervised models, DINOv2 learns pure visual structure — texture, composition, light — which is exactly what aesthetic judgement demands.

It is a mirror. You teach it what you love; it finds more of it.

---

## How it works

Three phases, running in a loop:

1. **Scrape** — pull images from configured sources (4chan, Tumblr, Imgur, Pixelfed) into an inbox
2. **Rate** — swipe through the inbox in the mobile-first web UI: good / fav / bad
3. **Train** — DINOv2 encodes your rated images; a logistic regression learns your taste

After training, the classifier auto-sorts new scrapes before they even reach the rating queue.

---

## Docker setup (recommended)

### 1. Copy and fill in the env file

```bash
cp .env.example .env
```

Edit `.env` — at minimum set a real `DJANGO_SECRET_KEY`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(50))"
```

Also set `ALLOWED_HOSTS` to your server's hostname or IP if running remotely.

### 2. Create the data directory and drop in a config

```bash
mkdir -p data
```

Create `data/config.toml` with your sources (see [Sources](#sources) below). This file is required for scraping; the app starts fine without it.

### 3. Start everything

```bash
docker compose up -d
```

This starts two containers from the same image:
- **memelord** — gunicorn web server on port 8000; runs migrations on first start
- **qcluster** — django-q worker for background scrape/train jobs; starts only after the web service is healthy

### 4. Create an admin user

```bash
docker compose run --rm memelord createsuperuser
```

Then open `http://localhost:8000` and log in.

### One-off commands

```bash
docker compose run --rm memelord scrape   # scrape now (outside the UI)
docker compose run --rm memelord train    # train now (outside the UI)
```

---

## Sources

Create `data/config.toml` (or manage sources from the Config page in the UI):

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

Sources can be marked NSFW individually in the UI. NSFW images are kept in a separate rating mode and hidden from the main queue unless toggled.

---

## Rating modes

| Mode | Queue | Left swipe / Bad | Up swipe / Fav | Right swipe / Good |
|---|---|---|---|---|
| **Inbox** | newly scraped | → void | → corpus ★ | → corpus |
| **Corpus** | rated good/fav | → void | toggle ★ | — |
| **Fav** | favourites only | → void | toggle ★ | — |
| **Trash** | void | — | → corpus ★ | → corpus |

Arrow keys and keyboard shortcuts work on desktop. Press `?` for the shortcut reference.

---

## Development setup

```bash
uv sync --extra dev      # install all deps
make run                 # Django dev server on :8000
make qcluster            # background worker (separate terminal — required for Train)
make migrate             # makemigrations + migrate
make superuser           # create admin user
make test                # full test suite
make test-fast           # skip @pytest.mark.integration tests
```

### Stack

- **Framework:** Django 6 + django-htmx
- **Background jobs:** django-q2 (ORM broker — no Redis needed)
- **Vision:** PyTorch + DINOv2 ViT-B/14
- **Classifier:** scikit-learn LogisticRegression
- **Scrapers:** 4chan, Tumblr, Imgur, Pixelfed (Playwright for JS-rendered pages)
- **Tooling:** [uv](https://docs.astral.sh/uv/)

### Data layout

```
data/
├── memelord.db           # SQLite (Django ORM + django-q broker)
├── Janulon_weights.pkl   # trained classifier
├── config.toml           # scraper sources (mount or place here)
├── inbox/                # scraped, awaiting rating
├── corpus/               # rated good/fav — positive training samples
└── void/                 # rated bad — negative training samples
```

---

## License

MIT. This tool is a prism; use it to refract the light you want to see.
