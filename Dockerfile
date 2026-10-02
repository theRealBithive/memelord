# Layers ordered for cache efficiency: slow/rare changes first.
FROM python:3.14-slim

# uv is copied from its own image so its version is pinned independently of
# pip. The dependency layer is built from uv.lock (--locked fails the build if
# the lock is stale), which makes the image reproducible; the previous
# `pip install -e .` resolved fresh versions on every build and ignored the lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

# UV_PROJECT_ENVIRONMENT puts the venv at a fixed path that we prepend to PATH,
# so the entrypoint can call `python`, `gunicorn` and `playwright` directly.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Python deps only — cached until pyproject.toml / uv.lock change.
# --no-install-project keeps app code out of this layer; the app is run from
# /app as a plain source tree (manage.py and gunicorn put cwd on sys.path),
# so the project itself never needs to be installed. --no-dev drops pytest,
# ruff and semantic-release. --no-cache keeps uv's download cache out of the
# layer (kaniko does not support BuildKit cache mounts, so a mount is not an
# option here).
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Chromium for the Playwright fallback scraper (Imgur). Installing it through
# the locked playwright package guarantees the browser build matches the
# driver version; --with-deps pulls the required apt libraries.
RUN playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# App code (separate COPYs so unrelated changes don't bust each other's layer)
COPY manage.py ./
COPY memelord/ ./memelord/
COPY ratings/ ./ratings/
COPY core/ ./core/
COPY retina/ ./retina/
COPY templates/ ./templates/
COPY static/ ./static/

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV DATA_DIR=/data
ENV CONFIG_PATH=/data/config.toml
ENV DJANGO_DEBUG=false

# Stamped at build time so the running app can display its version.
# CI passes the git tag (e.g. v1.10.3); local builds default to "dev".
ARG APP_VERSION=dev
ENV APP_VERSION=$APP_VERSION

VOLUME ["/data"]
EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["web"]
