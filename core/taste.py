"""
The taste model: one shared classifier plus one classifier per rating category.

Why per category: the library mixes categories with different criteria (jokes
from /b/ and tumblr, painted miniatures from /tg/, wallpapers). One linear
classifier over all of them learns the cheapest explanation, which is the
category itself as soon as the operator likes one category more often than
another, and the visibility dial then hides whole sources instead of bad
images. A classifier per category can only learn taste *within* that category.
Categories with too few ratings fall back to the shared classifier, which is
trained on everything exactly as before (V2, V4). This is the "generic model
plus per-domain adaptation" shape the personalised-aesthetics literature uses,
in its simplest form.

A category is the download source (`Image.source_label`, fixed at download
time) for a safe image, and the one NSFW group for a flagged image, whatever
its source (V1, V23): the operator judges NSFW by a criterion of its own in
which the origin plays no part, so per-source NSFW rows would only split that
criterion across sources that each lack the 10/10. `taste_group()` is the one
place that maps a row to its category; everything below it speaks of "source"
for historical reasons (the pickle key `per_source` must stay readable) and
means "category" throughout.

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

from core import brain, siglip

# The taste feature is the DINOv3 vector followed by the SigLIP2 search vector
# (contract V12). DINOv3 carries what is in the picture and how it is built,
# SigLIP2 carries the style and the aesthetic reading the CLIP family is known
# for; the aesthetic predictors in the literature sit on CLIP-like vectors.
FEATURE_DIM = brain.EMBEDDING_DIM + siglip.SEARCH_DIM

# A source gets its own classifier only once it has this many liked (score >= 3)
# and this many disliked (score <= 2, trash included) images with a current
# vector (V2). Below that, a 768-d logistic regression is noise; the operator
# chose 10/10 so a source starts judging itself early and accepts that it is
# wobbly at first.
MIN_GOOD_PER_SOURCE = 10
MIN_BAD_PER_SOURCE = 10

# The category of every flagged image. Parentheses, because no scraper can
# produce them in a label (boards and topics are alphanumeric, blog names use
# letters and dashes, handles carry `@`), so the group can never collide with a
# real source (V23). The name is what the train log and the stats page show.
NSFW_GROUP = "(nsfw)"


def taste_group(source_label: str, is_nsfw: bool) -> str:
    """
    The rating category of an image: its source, or the NSFW group when it is
    flagged (V1, V23).

    Takes the two scalars rather than the Image row so this module stays
    ORM-free; the trainer, classify_images and the view prediction all pass
    `(img.source_label, img.is_nsfw)`. A flagged image leaves its source's
    category entirely, it does not belong to both (V24).
    """
    if is_nsfw:
        return NSFW_GROUP
    return source_label


def _unit(vector: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    """One block of the feature at unit length; a zero vector has no direction and is refused."""
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    if flat.shape[0] != expected_dim:
        raise ValueError(f"{name} vector must have {expected_dim} values, got {flat.shape[0]}")
    norm = float(np.linalg.norm(flat))
    if norm == 0.0:
        raise ValueError(f"{name} vector is all zeros and cannot be normalised")
    return flat / norm


def combine_features(taste_vector: np.ndarray, search_vector: np.ndarray) -> np.ndarray:
    """
    The taste feature of one image: unit DINOv3 block, then unit SigLIP2 block
    (contract V12).

    Why normalise each block: a DINOv3 CLS vector has a norm around 15 while
    SigLIP2 vectors are stored at unit length. Fed raw into one logistic
    regression with an L2 penalty, the big block would get the small
    coefficients and the whole say, and the SigLIP2 block would be regularised
    into silence. At unit length both blocks compete on equal terms and the
    fit decides which one matters for a given source.
    """
    taste_block = _unit(taste_vector, brain.EMBEDDING_DIM, "taste")
    search_block = _unit(search_vector, siglip.SEARCH_DIM, "search")
    return np.concatenate([taste_block, search_block])


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

    max_iter is ten times sklearn's default: 1536-d features on a few thousand
    rows need more than 100 lbfgs steps to converge. There is no random_state
    on purpose: lbfgs is deterministic and sklearn reads the seed only for the
    sag, saga and liblinear solvers, so the knob would be dead.
    """
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, y, sample_weight=balance_class_weights(y, sample_weight))
    return classifier


@dataclass
class SourceGroup:
    """The training rows of one category (a source or the NSFW group), with the counts the threshold rule reads."""

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
    Split the training set by category label, in first-seen order.

    `labels[i]` must describe row i of X, y and sample_weight: the caller
    builds all four from the same filtered path list, after rows without a
    vector were dropped, so the counts here are counts of rows that actually
    train (V2 says "with a current vector"). The labels are whatever
    `taste_group()` returned, so the NSFW group is one label like any source
    and needs no special case here (V24).
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
    """The shared classifier and the per-category ones, keyed by `taste_group()`."""

    shared: LogisticRegression
    per_source: dict[str, LogisticRegression] = field(default_factory=dict)

    def classifier_for(self, group: str) -> LogisticRegression:
        """
        The category's own classifier when it has one, otherwise the shared one
        (V4, V5, V25). An unknown or empty label is simply a category without
        its own model; it must never raise, because a stray row with a label
        nobody trained on would otherwise stop a whole classify run. A file
        from before the NSFW group existed has no `(nsfw)` key, so flagged
        images take the shared model until the next training (V7).
        """
        return self.per_source.get(group, self.shared)


def save_taste_model(model: TasteModel, path: Path) -> None:
    """
    Write one pickle for the whole taste model (V8).

    The shared classifier stays under the key "classifier" that
    brain.save_classifier uses, so brain.load_classifier applied to this file
    still yields the shared model and a file written before per-source models
    existed still loads here (V7). The two encoder stamps are the same guard
    as in brain, once per feature block: a classifier is a hyperplane in one
    feature space and is meaningless in another (brain's contract V5, V14).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder": brain.ENCODER_ID,
        "search_encoder": siglip.SEARCH_ENCODER_ID,
        "classifier": model.shared,
        "per_source": dict(model.per_source),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_taste_model(path: Path) -> TasteModel | None:
    """
    Read a taste model, or None (with a warning) when the file must not be
    applied: a bare estimator from before encoder stamping, or a stamp from
    another encoder for either feature block (V7, V14). A file from before the
    SigLIP2 block has no search stamp and is foreign by the same rule: its
    hyperplane lives in 768 dimensions and the features now have 1536. None
    makes every caller behave as if no classifier existed, the state the UI
    already handles.

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
    search_trained_on = payload.get("search_encoder")
    if search_trained_on != siglip.SEARCH_ENCODER_ID:
        logger.warning(
            "{}: classifier was trained on search encoder {} but the search encoder is {}; "
            "ignored until the next train run",
            path, search_trained_on, siglip.SEARCH_ENCODER_ID,
        )
        return None
    return TasteModel(
        shared=payload["classifier"],
        per_source=dict(payload.get("per_source", {})),
    )
