FROM python:3.14-slim

WORKDIR /app

# System deps for Playwright/Chromium (Imgur browser scraping)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in a separate layer so code changes don't
# invalidate the slow pip + Playwright install.  Stub packages
# satisfy the editable-install metadata; real source is copied below.
COPY pyproject.toml README.md ./
RUN mkdir -p core retina \
    && touch core/__init__.py retina/__init__.py main.py \
    && pip install --no-cache-dir -e .

# Chromium for Playwright (Imgur JS-rendered pages)
RUN playwright install chromium && playwright install-deps chromium

# Now copy the actual source (editable install resolves via symlinks)
COPY main.py ./
COPY core/ ./core/
COPY retina/ ./retina/

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV JANULON_DATA=/data
VOLUME ["/data"]

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["--help"]
