# Order of layers is tuned for Kaniko/CI cache: rarely changing first.
# Pinning the base image by digest (e.g. python:3.14-slim@sha256:...) improves cache stability.
FROM python:3.14-slim

WORKDIR /app

# Layer 1: system deps (invalidated only when this RUN changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Playwright CLI + Chromium (install playwright first so the CLI exists)
RUN pip install --no-cache-dir playwright \
    && playwright install chromium \
    && playwright install-deps chromium

# Layer 3: Python deps only — copy just metadata so code changes don't bust this layer
COPY pyproject.toml README.md ./
RUN mkdir -p core retina \
    && touch core/__init__.py retina/__init__.py main.py \
    && pip install --no-cache-dir -e .

# Layer 4+: app code (separate COPYs so changes in one dir don't invalidate others)
COPY main.py ./
COPY core/ ./core/
COPY retina/ ./retina/

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV JANULON_DATA=/data
VOLUME ["/data"]

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["--help"]
