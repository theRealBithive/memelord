.PHONY: run migrate superuser sync lint test test-fast scrape train qcluster build push release release-push

REGISTRY_HOST       ?= registry.exkulpa.de
REGISTRY_IMAGE_BASE ?= registry.exkulpa.de/maximilian/janulon-registry
REGISTRY_IMAGE      ?= $(REGISTRY_IMAGE_BASE):latest

# Dev targets talk to the local SQLite DB with the insecure default SECRET_KEY,
# which the settings guard only permits in debug mode. Exported here so a plain
# `make run` works; an explicit DJANGO_DEBUG in the caller's shell wins (?=).
# Containers are unaffected: docker compose takes its env from .env, not from
# the shell.
export DJANGO_DEBUG ?= true

# === development
sync:
	uv sync

run:
	uv run python manage.py runserver

migrate:
	uv run python manage.py makemigrations
	uv run python manage.py migrate

superuser:
	uv run python manage.py createsuperuser

# === quality
lint:
	uv run ruff check .

test:
	uv run pytest -v

test-fast:
	uv run pytest -v -m "not integration"

# === data tasks (run manually or via the webapp buttons)
scrape:
	uv run python manage.py scrape

train:
	uv run python manage.py train

qcluster:
	uv run python manage.py qcluster

# === docker
# APP_VERSION is read by the Dockerfile and stamped into the image so the UI
# can display it. Falls back to "dev" if not in a git checkout.
APP_VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

build:
	APP_VERSION=$(APP_VERSION) docker compose build

push: build
	@if [ -n "$${PUBLIC_REGISTRY_PASSWORD}" ]; then \
		echo "$${PUBLIC_REGISTRY_PASSWORD}" | docker login "$(REGISTRY_HOST)" -u "PUBLIC_REGISTRY_USER" --password-stdin; \
	fi
	docker push "$(REGISTRY_IMAGE)"

# === release
release:
	uv run semantic-release version --no-push

release-push:
	@v=$$(git describe --tags --abbrev=0 2>/dev/null) || { echo "No tag found. Run 'make release' first."; exit 1; }; \
	git push && git push --tags; \
	if [ -n "$${PUBLIC_REGISTRY_PASSWORD}" ]; then \
		echo "$${PUBLIC_REGISTRY_PASSWORD}" | docker login "$(REGISTRY_HOST)" -u "PUBLIC_REGISTRY_USER" --password-stdin; \
	fi; \
	docker build --build-arg APP_VERSION=$$v -t "$(REGISTRY_IMAGE_BASE):$$v" -t "$(REGISTRY_IMAGE_BASE):latest" .; \
	docker push "$(REGISTRY_IMAGE_BASE):$$v"; \
	docker push "$(REGISTRY_IMAGE_BASE):latest"
