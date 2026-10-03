"""
The taste model: one shared classifier plus one classifier per download source.

Why per source: the library mixes categories with different criteria (jokes
from /b/ and tumblr, painted miniatures from /tg/, wallpapers). One linear
classifier over all of them learns the cheapest explanation, which is the
category itself as soon as the operator likes one category more often than
another, and the visibility dial then hides whole sources instead of bad
images. A classifier per source can only learn taste *within* that source.
The source is `Image.source_label`, fixed at download time, so the category
costs nothing and never moves (contract V1). Sources with too few ratings fall
back to the shared classifier, which is trained on everything exactly as
before (V2, V4). This is the "generic model plus per-domain adaptation" shape
the personalised-aesthetics literature uses, in its simplest form.

This module is pure (numpy, sklearn, pickle); the ORM side lives in
core/trainer.py and ratings/. Keeping it pure keeps it mutation-testable
without the Django fixtures.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from core import brain

# A source gets its own classifier only once it has this many liked (score >= 3)
# and this many disliked (score <= 2, trash included) images with a current
# vector (V2). Below that, a 768-d logistic regression is noise; the operator
# chose 10/10 so a source starts judging itself early and accepts that it is
# wobbly at first.
MIN_GOOD_PER_SOURCE = 10
MIN_BAD_PER_SOURCE = 10


def balance_class_weights(y: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    """
    Scale the sample weights so the liked and the disliked side carry the same
    total weight, while the per-score weights keep their ratios inside a side
    (contract V6).

    Why not sklearn's class_weight="balanced": it balances by *count* of rows
    per class, not by their weight, so ten sixes (weight 3 each) against a
    hundred ones (weight 1 each) would still tilt the fit 3:1 after sklearn's
    correction. Scaling each side to half of the total weight is exact and
    keeps the fit's overall scale (and so the strength of the L2 penalty) where
    it was. The library is 3:1 disliked locally; without this the classifier
    calls almost everything bad at the 0.5 threshold.
    """
    y = np.asarray(y)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    total = sample_weight.sum()
    balanced = np.empty_like(sample_weight)
    for class_value in (0, 1):
        in_class = y == class_value
        class_total = sample_weight[in_class].sum()
        balanced[in_class] = sample_weight[in_class] * (total / 2.0) / class_total
    return balanced


def fit_classifier(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
) -> LogisticRegression:
    """
    The one place that knows how a taste classifier is fitted, so the shared
    model and every per-source model are the same kind of estimator with the
    same balancing (V6). Callers guarantee both classes are present: the
    trainer refuses to run without a liked and a disliked image, and a source
    only gets here past has_enough_examples().
    """
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X, y, sample_weight=balance_class_weights(y, sample_weight))
    return classifier


@dataclass
class SourceGroup:
    """The training rows of one source, with the counts the threshold rule reads."""

    X: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray
    good_count: int
    bad_count: int


def group_by_source(
    labels: list[str],
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, SourceGroup]:
    """
    Split the training set by source label, in first-seen order.

    `labels[i]` must describe row i of X, y and sample_weight: the caller
    builds all four from the same filtered path list, after rows without a
    vector were dropped, so the counts here are counts of rows that actually
    train (V2 says "with a current vector").
    """
    rows_by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        rows_by_label.setdefault(label, []).append(index)

    groups: dict[str, SourceGroup] = {}
    for label, rows in rows_by_label.items():
        y_source = y[rows]
        groups[label] = SourceGroup(
            X=X[rows],
            y=y_source,
            sample_weight=sample_weight[rows],
            good_count=int(np.count_nonzero(y_source == 1)),
            bad_count=int(np.count_nonzero(y_source == 0)),
        )
    return groups


def has_enough_examples(group: SourceGroup) -> bool:
    """Contract V2: at least MIN_GOOD liked and MIN_BAD disliked rows, both inclusive."""
    enough_good = group.good_count >= MIN_GOOD_PER_SOURCE
    enough_bad = group.bad_count >= MIN_BAD_PER_SOURCE
    return enough_good and enough_bad


@dataclass
class TasteModel:
    """The shared classifier and the per-source ones, keyed by `Image.source_label`."""

    shared: LogisticRegression
    per_source: dict[str, LogisticRegression] = field(default_factory=dict)

    def classifier_for(self, source_label: str) -> LogisticRegression:
        """
        The source's own classifier when it has one, otherwise the shared one
        (V4, V5). An unknown or empty label is simply a source without its own
        model; it must never raise, because a stray row with a label nobody
        trained on would otherwise stop a whole classify run.
        """
        return self.per_source.get(source_label, self.shared)


def save_taste_model(model: TasteModel, path: Path) -> None:
    """
    Write one pickle for the whole taste model (V8).

    The shared classifier stays under the key "classifier" that
    brain.save_classifier uses, so brain.load_classifier applied to this file
    still yields the shared model and a file written before per-source models
    existed still loads here (V7). The encoder stamp is the same guard as in
    brain: a classifier is a hyperplane in one embedding space and is
    meaningless in another (brain's contract V5).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder": brain.ENCODER_ID,
        "classifier": model.shared,
        "per_source": dict(model.per_source),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_taste_model(path: Path) -> TasteModel | None:
    """
    Read a taste model, or None (with a warning) when the file must not be
    applied: a bare estimator from before encoder stamping, or a stamp from
    another encoder (V7). None makes every caller behave as if no classifier
    existed, the state the UI already handles.

    This repeats brain.load_classifier's stamp rule on purpose instead of
    calling it: that function returns only the estimator and drops the dict
    with the per-source models, and it is under the mutation suite with a
    recorded disposition for every mutant, so it stays as it is. Same log texts
    so the operator sees one message for one problem whichever path hit it.
    """
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        logger.warning(
            "{}: classifier predates encoder stamping; ignored until the next train run",
            path,
        )
        return None
    trained_on = payload.get("encoder")
    if trained_on != brain.ENCODER_ID:
        logger.warning(
            "{}: classifier was trained on {} but the encoder is {}; "
            "ignored until the next train run",
            path, trained_on, brain.ENCODER_ID,
        )
        return None
    return TasteModel(
        shared=payload["classifier"],
        per_source=dict(payload.get("per_source", {})),
    )
