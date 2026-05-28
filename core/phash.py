"""Perceptual hashing for cheap near-duplicate pre-filtering."""

from pathlib import Path

import numpy as np
from PIL import Image


def _dct1d(signal: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy Type-II DCT (orthonormal) to avoid a scipy dependency. pHash
    runs on every downloaded image before the GPU encoder is loaded, so the
    import cost of scipy would slow down scrape startup even when the GPU path
    isn't used. Orthonormal scaling keeps energy distribution stable across
    different block sizes.
    """
    n = signal.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        scale = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
        out[k] = scale * np.sum(signal * np.cos(np.pi * k * (np.arange(n) + 0.5) / n))
    return out


def _dct2(block: np.ndarray) -> np.ndarray:
    """
    Separable application: 1-D DCT across rows then across columns of the
    transpose. Mathematically equivalent to the full 2-D DCT but avoids
    building the full N²×N² transform matrix.
    """
    temp = np.apply_along_axis(_dct1d, 1, block)
    return np.apply_along_axis(_dct1d, 1, temp.T).T


def compute_phash(path: Path) -> str:
    """
    Produces a 64-bit fingerprint that is stable across minor edits (JPEG
    recompression, slight resize, minor colour shift). The 32×32 grayscale
    resize discards fine detail so only structural content drives the hash;
    the top-left 8×8 DCT coefficients capture the dominant low-frequency
    energy that persists across those edits. Compared via Hamming distance
    rather than equality, so the threshold controls how aggressively
    near-duplicates are suppressed.
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
    """
    XOR of the two integers sets a bit wherever the hashes disagree; bit_count()
    then counts those positions. The equal-length guard defends against comparing
    hashes produced with different block sizes if the algorithm is ever changed.
    """
    if len(a) != len(b):
        raise ValueError("phash strings must have equal length")
    ai = int(a, 16)
    bi = int(b, 16)
    return (ai ^ bi).bit_count()


def is_phash_duplicate(phash: str, phash_ints: list[int], max_distance: int) -> bool:
    """
    Linear scan is acceptable because this only runs on images that already
    passed SHA-256 dedup. For typical collection sizes the scan is fast enough
    that a BK-tree would add complexity without measurable gain.

    `phash_ints` is pre-parsed to integers once when the DedupIndex loads
    (PERF-8): the index list is rescanned for every candidate, so parsing each
    stored hex string per pair made hex parsing O(N²) per scrape. Here only the
    candidate is parsed (once), then XOR + bit_count() against the cached ints.
    No truthiness filter on the list: an all-zero hash (a solid/near-uniform
    image) is a legitimate stored fingerprint, and int 0 is falsy — filtering it
    would silently drop it as a comparison target. NULL/empty phashes are
    excluded upstream, on the string side, before they ever become ints.
    """
    if not phash or not phash_ints:
        return False
    value = int(phash, 16)
    return any((value ^ other).bit_count() <= max_distance for other in phash_ints)
