"""Pytest configuration: make loguru output visible to caplog."""

import pytest
from loguru import logger


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
