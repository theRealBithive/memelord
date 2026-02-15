"""Tests for retina.reddit."""

import pytest


def test_reddit_module_imports() -> None:
    """Reddit scraper module can be imported."""
    import retina.reddit  # noqa: F401

    assert retina.reddit is not None
