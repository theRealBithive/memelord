"""Tests for core.phash."""

from pathlib import Path

from PIL import Image

from core import phash


def test_compute_phash_stable_for_same_image(tmp_path: Path) -> None:
    """Same file yields identical phash."""
    path = tmp_path / "a.png"
    Image.new("RGB", (64, 64), color=(100, 50, 200)).save(path)
    assert phash.compute_phash(path) == phash.compute_phash(path)


def test_hamming_distance_identical_is_zero() -> None:
    """Identical hex strings have Hamming distance 0."""
    h = "a" * 16
    assert phash.hamming_distance(h, h) == 0


def test_is_phash_duplicate_within_threshold(tmp_path: Path) -> None:
    """Near-identical images match within small Hamming distance."""
    base = tmp_path / "base.png"
    near = tmp_path / "near.png"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(base)
    img = Image.open(base)
    img.putpixel((0, 0), (11, 21, 31))
    img.save(near)
    h1 = phash.compute_phash(base)
    h2 = phash.compute_phash(near)
    assert phash.is_phash_duplicate(h2, [int(h1, 16)], max_distance=12)


def test_is_phash_duplicate_far_apart(tmp_path: Path) -> None:
    """Very different images are not duplicates at tight threshold."""
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(a)
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(b)
    h1 = phash.compute_phash(a)
    h2 = phash.compute_phash(b)
    assert not phash.is_phash_duplicate(h2, [int(h1, 16)], max_distance=3)


def test_all_zero_phash_is_a_valid_comparison_target() -> None:
    """A solid/near-uniform image hashes to int 0, which is falsy. The int list
    must not be truthiness-filtered, or such fingerprints would silently stop
    matching (PERF-8 regression guard)."""
    assert phash.is_phash_duplicate("0000000000000000", [0], max_distance=0)
