"""Embedding generations: which stored vectors are current, and re-encoding the rest."""

from pathlib import Path

from loguru import logger

from core import brain
from ratings.models import Image


def has_current_embedding(image: Image) -> bool:
    """
    The single definition of "this row's vector may be compared with others".
    Both halves matter: a NULL blob has nothing to compare, and a blob stamped
    with another encoder lives in a different space (contracts V1, V2).
    """
    return image.embedding is not None and image.embedding_model == brain.ENCODER_ID


def stale_images():
    """
    Non-purged rows whose vector is missing or from another encoder.

    Purged rows are excluded on purpose: their file is gone, so there is
    nothing to encode, and they only need to keep blocking re-downloads by
    content hash (contract V4).
    """
    return Image.objects.filter(is_purged=False).exclude(
        embedding__isnull=False, embedding_model=brain.ENCODER_ID
    )


def reencode_stale_embeddings(
    data_dir: Path,
    *,
    encoder=None,
    transform=None,
    chunk_size: int = 256,
    batch_size: int = 32,
    limit: int | None = None,
    progress_label: str = "reencode",
) -> dict[str, int]:
    """
    Bring every stale row up to the current encoder (contract V3).

    Work is cut into chunks and each chunk's rows are saved as soon as it is
    encoded, so an interrupted run resumes where it stopped instead of
    repeating hours of GPU time: brain.encode() only returns at the very end of
    its list, so handing it the whole library would hold all progress in memory
    with nothing durable. chunk_size is that durability granularity; batch_size
    is the GPU batch inside brain.encode.

    Rows whose file is missing are skipped and counted: marking them purged is
    repair_orphans' job. Returns counts so the management command and the tests
    can report what happened.
    """
    queryset = stale_images().order_by("downloaded_at")
    if limit:
        queryset = queryset[:limit]
    images = list(queryset)
    present = [img for img in images if (data_dir / img.file_path).exists()]
    missing_file = len(images) - len(present)
    result = {"encoded": 0, "missing_file": missing_file, "unreadable": 0}
    if not present:
        return result

    if encoder is None:
        encoder = brain.get_encoder()
    if transform is None:
        transform = brain.get_transform()

    total = len(present)
    for start in range(0, total, chunk_size):
        chunk = present[start : start + chunk_size]
        paths = [data_dir / img.file_path for img in chunk]
        embeddings, valid_paths = brain.encode(
            encoder,
            paths,
            transform=transform,
            batch_size=batch_size,
            progress_label=progress_label,
        )
        path_to_emb = dict(zip(valid_paths, embeddings, strict=True))
        for img, path in zip(chunk, paths, strict=True):
            emb = path_to_emb.get(path)
            if emb is None:
                result["unreadable"] += 1
                continue
            img.embedding = brain.embedding_to_bytes(emb)
            img.embedding_model = brain.ENCODER_ID
            img.save(update_fields=["embedding", "embedding_model"])
            result["encoded"] += 1
        done = min(start + chunk_size, total)
        logger.info("{}: {}/{} images re-encoded with {}", progress_label, done, total, brain.ENCODER_ID)
    return result
