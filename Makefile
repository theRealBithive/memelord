# Local mirror of .gitlab-ci.yml: test stage + docker build (no Kaniko, no push).
# Run "make test" and "make build" to replicate the pipeline locally.

.PHONY: test build login push

# === test stage (same as CI: pip install -e ".[dev]" then pytest)
test:
	pip install -e ".[dev]"
	pytest -v

# === build stage: build image locally (docker build instead of Kaniko)
# Default tag includes registry so "docker push" goes to exkulpa, not docker.io.
REGISTRY_IMAGE ?= registry.exkulpa.de/maximilian/janulon-registry:latest

build:
	docker build -t "$(REGISTRY_IMAGE)" .

# Login mit CI-Token: PUBLIC_REGISTRY_USER und PUBLIC_REGISTRY_PASSWORD setzen (wie in GitLab CI).
# Beispiel: PUBLIC_REGISTRY_USER=... PUBLIC_REGISTRY_PASSWORD=... make login
REGISTRY_HOST ?= registry.exkulpa.de

login:
	@echo "$${PUBLIC_REGISTRY_PASSWORD}" | docker login "$(REGISTRY_HOST)" -u "PUBLIC_REGISTRY_USER" --password-stdin

# Push (loggt ein, wenn PUBLIC_REGISTRY_USER und PUBLIC_REGISTRY_PASSWORD gesetzt sind)
push: build
	@if [ -n "$${PUBLIC_REGISTRY_PASSWORD}" ]; then $(MAKE) login; fi
	docker push "$(REGISTRY_IMAGE)"
