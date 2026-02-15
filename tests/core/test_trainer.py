"""Tests for core.trainer."""

import pytest


def test_trainer_module_imports() -> None:
    """Trainer module can be imported."""
    import core.trainer  # noqa: F401

    assert core.trainer is not None
