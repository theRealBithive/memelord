"""Tests for core.nsfw."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from core import nsfw


def test_predict_nsfw_uses_low_threshold() -> None:
    """predict_nsfw returns True when prob meets threshold."""
    clf = LogisticRegression()
    X = np.vstack([np.zeros(768), np.ones(768) * 2])
    y = np.array([0, 1])
    clf.fit(X, y)
    assert nsfw.predict_nsfw(clf, np.ones(768) * 2, threshold=0.30)
    assert not nsfw.predict_nsfw(clf, np.zeros(768), threshold=0.99)


def test_train_nsfw_classifier_has_class_weight() -> None:
    """NSFW trainer uses higher weight on positive class."""
    X = np.random.randn(20, 768).astype(np.float32)
    y = np.array([1] * 10 + [0] * 10)
    clf = nsfw.train_nsfw_classifier(X, y)
    assert clf.class_weight == {0: 1, 1: 3}
