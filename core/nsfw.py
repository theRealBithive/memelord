"""Personal NSFW classifier on DINO embeddings (high-recall policy)."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from core import brain


def train_nsfw_classifier(
    X: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    """Fit logistic regression biased toward recall on NSFW class."""
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
    """True when P(NSFW) >= threshold (use a low threshold for high recall)."""
    prob = float(brain.predict_proba(classifier, embedding))
    return prob >= threshold
