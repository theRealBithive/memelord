"""Pytest configuration: make loguru output visible to caplog."""

import io

import pytest
from loguru import logger
from PIL import Image


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
