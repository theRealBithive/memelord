# JANULON

> "The machine does not need to see. It only needs to feel."

Janulon is a subjective aesthetic engine. Unlike generative AI (which creates new noise), Janulon is a curatorial AI designed to filter the digital ocean for specific visual frequencies.

It uses Meta's DINOv2 (self-supervised vision transformer) to map images into high-dimensional vector space, then applies a custom-trained linear probe to determine if a new image aligns with the operator's specific taste. Unlike language-supervised models, DINOv2 learns pure visual structure — texture, composition, light — which is exactly what aesthetic judgement demands.

It is a mirror. You teach it what you love; it finds more of it.

## Architecture

Janulon operates in a continuous loop of three phases: Acquisition, Evaluation, and Curating.

```mermaid
graph LR
    A[The Web] -->|Scrapers| B(Input Buffer)
    B -->|Pre-process| C{DINOv2 Encoder}
    C -->|Vector 768d| D[The Taste Matrix]
    D -->|Score > 0.90| E[Archive / Post]
    D -->|Score < 0.90| F[The Void]
```

### The Stack

- **Core:** Python 3.9+
- **Tooling:** [uv](https://docs.astral.sh/uv/) (install & run)
- **Vision:** PyTorch + DINOv2 (ViT-B/14)
- **Logic:** Scikit-Learn (Logistic Regression / MLP)
- **Retina (Scrapers):** 4chan, Tumblr (v1 API), Imgur (scraping; optional Playwright for JS-rendered topic pages)

### Directory Structure

```
Janulon/
├── core/
│   ├── brain.py       # The neural logic (DINOv2 + Classifier)
│   └── trainer.py     # The script that learns your taste
├── retina/
│   ├── fourchan.py    # Scraper for 4chan /wg/ (wallpaper)
│   ├── tumblr.py      # Scraper for Tumblr
│   ├── imgur.py       # Scraper for Imgur topics (e.g. /t/funny)
│   └── reddit.py      # Scraper for Reddit
├── data/
│   ├── corpus/        # POSITIVE samples (Images you love)
│   └── void/          # NEGATIVE samples (Random noise/memes)
├── main.py            # The execution loop
├── pyproject.toml     # Dependencies and project metadata
└── tests/             # Tests mirroring source structure
```

## Quick Start

### 1. Installation

Clone the repository and install the project (dependency management via `pyproject.toml`). Using [uv](https://docs.astral.sh/uv/) is recommended.

Or with pip:

```bash
pip install -e .
```

### 2. Induction (Training Phase)

Janulon creates a decision boundary based on your curation history.

1. **Fill the Corpus:** Place 50–100 images that represent your target aesthetic into `data/corpus/`.
2. **Fill the Void:** Place 50–100 random images (screenshots, text, bad photos) into `data/void/`.
3. **Run the calibration:**

```bash
python3 core/trainer.py
```

Output: `Janulon_weights.pkl` (the mathematical representation of your taste).

### 3. Observation (Inference Phase)

Once calibrated, run the main loop. Janulon will scrape configured sources, judge images, and save the matches.

**Simple (all sources from config):**

```bash
# Edit config.toml with 4chan boards, tumblr blogs, imgur topics, then:
python3 main.py --source all
```

**Single source (CLI):**

```bash
# 4chan /wg/ (wallpaper general): no API key
python3 main.py --source 4chan --board wg --output_folder ./pics --index_pages 2

# Imgur topic (e.g. /t/funny): scraping only; for JS-rendered pages install optional browser support
uv pip install 'janulon[imgur-browser]' && playwright install chromium
python3 main.py --source imgur --topic funny --output_folder ./pics

# Reddit (when implemented): requires API key
python3 main.py --source reddit --subreddit architecture --threshold 0.85
```

### 4. Testing

- **Framework:** pytest
- **Layout:** Tests live in `tests/`, mirroring the source layout (e.g. `tests/core/test_brain.py`).
- **Naming:** Files `test_<module>.py`; functions `test_<behavior_being_tested>`.

Install dev dependencies and run tests:

```bash
uv sync --extra dev
uv run pytest
```

Every new function, class, or feature must have corresponding tests. Use one behavior per test, Arrange–Act–Assert structure, and minimal fixtures.

## Configuration

### The Threshold (`--threshold`)

The confidence required for Janulon to accept an image.

| Value | Description |
|-------|-------------|
| 0.50 | Permissive. Will let in anything remotely similar. |
| 0.85 | Strict. High quality, distinct style match. **(Recommended)** |
| 0.98 | The God Tier. Only images mathematically nearly identical to your corpus. |

### Config (for `--source all`)

With `--source all`, Janulon reads **`config.toml`** (or `--config <path>`) and scrapes every listed 4chan board, Tumblr blog, and Imgur topic. No need to pass `--board`, `--blog`, or `--topic` on the CLI.

Example **`config.toml`**:

```toml
[4chan]
boards = ["wg", "a"]

[tumblr]
blogs = ["staff", "someblog"]

[imgur]
topics = ["funny", "pics"]
```

Run: `python main.py --source all` (optionally `--output_folder ./out`, `--weights Janulon_weights.pkl`).

### Sources (API usage)

- **4chan /wg/ (wallpaper general):** Public JSON API, no API key. Use the `retina.fourchan` scraper:

  ```python
  from retina.fourchan import iter_image_urls, get_index, get_thread, image_url_from_post

  # Collect image URLs from the first 2 index pages (rate-limited to 1 req/s)
  urls = iter_image_urls(board="wg", index_pages=2)

  # Or fetch a single index page or full thread
  page = get_index("wg", page=1)
  thread = get_thread("wg", thread_no=12345)
  for post in thread.get("posts", []):
      url = image_url_from_post(post, "wg")
      if url:
          ...
  ```

- **Reddit:** Subreddits like /r/Brutalism, /r/LiminalSpace, /r/Cyberpunk
- **Tumblr:** Specific aesthetic blogs to traverse reblog trees
- **Local:** A folder of unsorted images to filter

## Development

- **Tooling:** [uv](https://docs.astral.sh/uv/) for installs and running scripts (`uv run pytest`, `uv run python main.py`, etc.).
- **Style:** PEP 8 and PEP 257 (docstrings); type hints on all function signatures.
- **Paths:** Prefer `pathlib` over `os.path`; use f-strings for formatting.
- **Dependencies:** Managed in `pyproject.toml` only; do not add new dependencies unless explicitly required.

## Roadmap

- [x] Phase I: Binary Classification (Like/Dislike)
- [ ] Phase II: Multi-Modal Feedback (re-training based on social engagement metrics)
- [ ] Phase III: The "Eye" (integration with Pinterest API for infinite scrolling)
- [ ] Phase IV: Video/GIF support (extracting keyframes for aesthetic evaluation)

## License

MIT. This tool is a prism; use it to refract the light you want to see.
