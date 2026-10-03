"""
The taste feature of an Image row: both vector generations, read from the row.

Why a module of its own: the DINOv3 half lives in ratings/embeddings.py, the
SigLIP2 half in ratings/search.py, and neither may import the other (search
would pull the vector bank into the embedding helpers). The two gates and the
feature builder meet here, so the trainer, classify_images and the lazy view
prediction share one definition of "this row can be judged" (taste contract
V12, V13).
"""

import numpy as np

from core import brain, taste
from ratings.embeddings import has_current_embedding
from ratings.models import Image
from ratings.search import has_search_embedding


def has_taste_features(image: Image) -> bool:
    """
    Both vectors present and from the current generation (V13). Either half
    missing means the row is neither trained on nor predicted; it keeps
    predicted_score NULL, which the review queue reads as "show anyway".
    """
    return has_current_embedding(image) and has_search_embedding(image)


def taste_features(image: Image) -> np.ndarray:
    """The 1536-d feature of a row that passed has_taste_features (V12)."""
    taste_vector = brain.bytes_to_embedding(bytes(image.embedding))
    search_vector = brain.bytes_to_embedding(bytes(image.search_embedding))
    return taste.combine_features(taste_vector, search_vector)
