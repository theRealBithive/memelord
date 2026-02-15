"""Tests for core.brain (DINOv2 + classifier)."""

import pytest


def test_brain_module_imports() -> None:
    """Brain module can be imported."""
    import core.brain  # noqa: F401

    assert core.brain is not None
