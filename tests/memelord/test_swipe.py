"""Swipe gesture script guards."""

from pathlib import Path


def test_swipe_only_handles_touches_starting_on_image() -> None:
    swipe_js = (Path(__file__).resolve().parents[2] / "static" / "swipe.js").read_text()
    assert '.closest(".image-wrap")' in swipe_js
    assert "swipeActive" in swipe_js
