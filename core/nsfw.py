"""Personal NSFW classifier on DINO embeddings (high-recall policy)."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from core import brain

# A head fitted on fewer decided examples than this says more about the few
# pictures than about the operator's line (NSFW contract N6); the same floor
# as a taste group's own model.
MIN_NSFW_EXAMPLES = 10
MIN_SAFE_EXAMPLES = 10


def has_enough_examples(nsfw_count: int, safe_count: int) -> bool:
    """Whether the decided examples suffice to fit the head (NSFW contract N6)."""
    return nsfw_count >= MIN_NSFW_EXAMPLES and safe_count >= MIN_SAFE_EXAMPLES


def train_nsfw_classifier(
    X: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    """
    Fit logistic regression biased toward recall on the NSFW class.

    class_weight {1: 3} triples the penalty for false negatives (missed NSFW)
    relative to false positives (safe images flagged as NSFW). The goal is to
    err on the side of flagging rather than missing — the user reviews flagged
    images manually in the NSFW queue, so false positives just add a small
    review burden while false negatives slip through into the main queue
    undetected.
    """
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight={0: 1, 1: 3},
    )
    classifier.fit(X, y)
    return classifier


def predict_nsfw(
    classifier: LogisticRegression,
    embedding: np.ndarray,
    threshold: float,
) -> bool:
    """
    Return True when P(NSFW) >= threshold.

    A low threshold (default 0.30) keeps recall high at the cost of more false
    positives — preferable here because the NSFW queue provides a manual safety
    net, whereas missed NSFW images bypass it entirely.
    """
    prob = float(brain.predict_proba(classifier, embedding))
    return prob >= threshold
