"""Pytest configuration: loguru routing and integration-test gating."""

import io
import os

import pytest

# Provide a dummy key so the settings guard doesn't fire during tests.
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-not-for-production")
os.environ.setdefault("DJANGO_DEBUG", "true")
from loguru import logger
from PIL import Image


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --run-integration CLI flag."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run slow integration tests (model download, network).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip tests marked ``integration`` unless --run-integration is given."""
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="needs --run-integration flag")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def minimal_png_bytes() -> bytes:
    """Return bytes of a minimal valid PNG (1x1 pixel) for download tests."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Route loguru to caplog so caplog.text captures loguru output."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(autouse=True)
def _reset_views_module_caches():
    """Null the module-level caches in ratings.views before each test.

    _taste_clf_cache, _similar_index_cache and their mtime / built_at companions
    survive across tests in the same process; without a reset, the first test
    to populate them leaks state into every subsequent test (e.g. a stale kNN
    matrix from one TestCase's fixture images being seen by another). Cheap to
    null — the cache rebuilds lazily on first use.
    """
    yield
    try:
        from ratings import views
    except Exception:
        return
    views._taste_clf_cache = None
    views._taste_clf_mtime = None
    views._similar_index_cache = None
    views._similar_index_built_at = 0.0
