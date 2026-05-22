# Layers ordered for cache efficiency: slow/rare changes first.
FROM python:3.14-slim

WORKDIR /app

# System deps for Playwright/Chromium (scraper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Playwright + Chromium
RUN pip install --no-cache-dir playwright \
    && playwright install chromium \
    && playwright install-deps chromium

# Python deps — cached until pyproject.toml changes
COPY pyproject.toml README.md ./
RUN mkdir -p core retina ratings memelord \
    && touch core/__init__.py retina/__init__.py ratings/__init__.py memelord/__init__.py \
    && pip install --no-cache-dir -e .

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
