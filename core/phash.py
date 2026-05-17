"""Perceptual hashing for cheap near-duplicate pre-filtering."""

from pathlib import Path

import numpy as np
from PIL import Image


def _dct1d(signal: np.ndarray) -> np.ndarray:
    """Type-II DCT along one axis (orthonormal)."""
    n = signal.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        scale = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
        out[k] = scale * np.sum(signal * np.cos(np.pi * k * (np.arange(n) + 0.5) / n))
    return out


def _dct2(block: np.ndarray) -> np.ndarray:
    temp = np.apply_along_axis(_dct1d, 1, block)
    return np.apply_along_axis(_dct1d, 1, temp.T).T


def compute_phash(path: Path) -> str:
    """
    Compute a 64-bit perceptual hash as 16-char hex.

    Standard pHash: resize 32x32 grayscale, DCT, top-left 8x8 AC, median threshold.
    """
    img = Image.open(path).convert("L")
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    dct = _dct2(pixels)
    low = dct[:8, :8].flatten()
    median = np.median(low)
    bits = low > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("phash strings must have equal length")
    ai = int(a, 16)
    bi = int(b, 16)
    return (ai ^ bi).bit_count()


def is_phash_duplicate(phash: str, phashes: list[str], max_distance: int) -> bool:
    if not phash or not phashes:
        return False
    return any(
        hamming_distance(phash, other) <= max_distance for other in phashes if other
    )
