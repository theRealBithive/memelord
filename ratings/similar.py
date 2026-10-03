"""Similar images: nearest neighbours by cosine over the DINOv3 taste vectors."""

from core import brain
from ratings.embeddings import has_current_embedding
from ratings.models import Image
from ratings.vector_bank import VectorBank, first_allowed

TASTE_BANK = VectorBank("embedding", "embedding_model", brain.ENCODER_ID)


def rank_similar(anchor: Image, candidates, limit: int) -> list[tuple[str, float]] | None:
    """
    The most similar other images to `anchor`, highest first, restricted to
    `candidates` (contract V17).

    Returns None when the anchor has no vector from the current encoder, so
    the view can say so instead of showing an empty page: a stale anchor is
    the normal state right after an encoder switch, until Train re-encodes
    it. Only the anchor itself is dropped from the ranking; another image with
    an identical vector (a true duplicate) is exactly what the user wants to
    see first (risk R18). DINOv3 vectors only, never the search vectors (V2).
    """
    if not has_current_embedding(anchor):
        return None
    allowed = set(candidates.values_list("content_hash", flat=True))
    allowed.discard(anchor.content_hash)
    if not allowed:
        return []
    anchor_vector = brain.bytes_to_embedding(bytes(anchor.embedding))
    ranked_hashes, similarities = TASTE_BANK.rank(anchor_vector)
    return first_allowed(ranked_hashes, similarities, allowed, limit)
