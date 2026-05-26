"""Fetch images from a Mastodon account's media timeline, with cursor-based resume."""

from pathlib import Path

from retina._mastoapi import (
    _filename_for_item as _shared_filename,
    download_images as _shared_download,
    iter_image_items,
)

__all__ = ["iter_image_items", "download_images", "_filename_for_item"]


def _filename_for_item(account_handle: str, image_url: str) -> str:
    """Deterministic filename for a Mastodon image (mastodon_<slug>_<hash><ext>)."""
    return _shared_filename("mastodon", account_handle, image_url)


def download_images(
    items: list[tuple[str, str]],
    output_dir: Path,
    account_handle: str,
    *,
    rate_limit_sec: float = 0.5,
    skip_dirs: list[Path] | None = None,
    skip_paths: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    """
    Download each image to output_dir. source_label in the result is account_handle.

    Thin wrapper over _mastoapi.download_images — see that module for full docs.
    """
    return _shared_download(
        items,
        output_dir,
        account_handle,
        prefix="mastodon",
        rate_limit_sec=rate_limit_sec,
        skip_dirs=skip_dirs,
        skip_paths=skip_paths,
    )
