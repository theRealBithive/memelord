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
