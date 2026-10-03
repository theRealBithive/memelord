"""Pytest configuration: loguru routing and integration-test gating."""

import io
import os

import pytest

# Provide a dummy key so the settings guard doesn't fire during tests.
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-not-for-production")
os.environ.setdefault("DJANGO_DEBUG", "true")
from hypothesis import HealthCheck
from hypothesis import settings as hypothesis_settings
from loguru import logger
from PIL import Image

# Hypothesis profiles. The DB-backed property tests are slow by nature (each
# example inserts rows into SQLite), so too_slow is a tooling signal here, not a
# property. Under mutmut the same test method runs twice in one process, which
# trips differing_executors; that profile is selected with
# HYPOTHESIS_PROFILE=mutmut and exists only for that runner.
hypothesis_settings.register_profile(
    "default", suppress_health_check=[HealthCheck.too_slow]
)
hypothesis_settings.register_profile(
    "mutmut",
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.differing_executors],
)
hypothesis_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


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
    """Null the taste-model cache in ratings.views.common before each test.

    _taste_model_cache and its mtime companion survive across tests in the same
    process; without a reset, the first test to populate them leaks state into
    every subsequent test. Cheap to null — the cache rebuilds lazily on first
    use.
    """
    yield
    try:
        from ratings.views import common
    except Exception:
        return
    common._taste_model_cache = None
    common._taste_model_mtime = None
