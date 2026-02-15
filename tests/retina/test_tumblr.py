"""Tests for retina.tumblr."""

import pytest


def test_tumblr_module_imports() -> None:
    """Tumblr scraper module can be imported."""
    import retina.tumblr  # noqa: F401

    assert retina.tumblr is not None
