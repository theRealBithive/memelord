# Local mirror of .gitlab-ci.yml: test stage + docker build (no Kaniko, no push).
# Run "make test" and "make build" to replicate the pipeline locally.

.PHONY: test build login push release release-push

# === test stage (same as CI: pip install -e ".[dev]" then pytest)
test:
	pip install -e ".[dev]"
	pytest -v

# === build stage: build image locally (docker build instead of Kaniko)
# Default tag includes registry so "docker push" goes to exkulpa, not docker.io.
REGISTRY_HOST ?= registry.exkulpa.de
REGISTRY_IMAGE_BASE ?= registry.exkulpa.de/maximilian/janulon-registry
REGISTRY_IMAGE ?= $(REGISTRY_IMAGE_BASE):latest

build:
	docker build -t "$(REGISTRY_IMAGE)" .

# Login mit CI-Token: PUBLIC_REGISTRY_USER und PUBLIC_REGISTRY_PASSWORD setzen (wie in GitLab CI).
# Beispiel: PUBLIC_REGISTRY_USER=... PUBLIC_REGISTRY_PASSWORD=... make login
login:
	@echo "$${PUBLIC_REGISTRY_PASSWORD}" | docker login "$(REGISTRY_HOST)" -u "PUBLIC_REGISTRY_USER" --password-stdin

# Push (loggt ein, wenn PUBLIC_REGISTRY_USER und PUBLIC_REGISTRY_PASSWORD gesetzt sind)
push: build
	@if [ -n "$${PUBLIC_REGISTRY_PASSWORD}" ]; then $(MAKE) login; fi
	docker push "$(REGISTRY_IMAGE)"

# === release: bump version from conventional commits, update changelog, commit and tag (local only).
release:
	pip install -e ".[dev]"
	semantic-release version --no-push

# === release-push: push git + tags, then build and push Docker image tagged with release version and :latest.
# Requires a release first (make release). Uses latest git tag (e.g. v0.1.1) for the container tag.
release-push:
	@v=$$(git describe --tags --abbrev=0 2>/dev/null) || { echo "No tag found. Run 'make release' first."; exit 1; }; \
	git push && git push --tags; \
	if [ -n "$${PUBLIC_REGISTRY_PASSWORD}" ]; then $(MAKE) login; fi; \
	docker build -t "$(REGISTRY_IMAGE_BASE):$$v" -t "$(REGISTRY_IMAGE_BASE):latest" .; \
	docker push "$(REGISTRY_IMAGE_BASE):$$v"; \
	docker push "$(REGISTRY_IMAGE_BASE):latest"
