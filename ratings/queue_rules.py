"""Single source of truth for review and below-cutoff queue filters.

The 1-6 dial on /config maps to a predicted_score cutoff. Unrated images at
or above the cutoff go to /review/; unrated images below it go to /below-cutoff/
so nothing is silently dropped. User-rated scores <= 2 also stay in below-cutoff.
"""

from collections.abc import Callable
from dataclasses import dataclass

from django.db.models import F, FloatField, Q, Value
from django.db.models.functions import Abs, Coalesce

from ratings.models import Image, ReviewThresholds


def bucket_to_cutoff(bucket: int) -> float:
    """Map a 1-6 threshold dial to an inclusive probability lower bound.

    Bucket 1 means "show everything" (cutoff 0.0); bucket 6 is the strictest.
    Images with predicted_score >= cutoff pass the visibility filter. Out-of-
    range buckets are clamped because the value comes from user-edited TOML.
    """
    return (max(1, min(6, int(bucket))) - 1) / 6


def review_settings_row() -> ReviewThresholds:
    """Fetch the review-settings singleton, seeding it from Django settings on first access.

    DB-first so UI edits on /config persist immediately; config.toml values
    only matter on the very first call (or after the row is manually deleted)
    because get_or_create writes them once and never re-reads. Every reader of
    the row goes through here so the seeding happens the same way no matter
    which setting is asked for first.
    """
    from django.conf import settings

    row, _ = ReviewThresholds.objects.get_or_create(
        pk=1,
        defaults={
            "sfw_threshold": getattr(settings, "SFW_THRESHOLD_BUCKET", 1),
            "nsfw_threshold": getattr(settings, "NSFW_THRESHOLD_BUCKET", 1),
        },
    )
    return row


def get_review_thresholds() -> tuple[int, int]:
    """Return (sfw_bucket, nsfw_bucket) from the DB singleton."""
    row = review_settings_row()
    return row.sfw_threshold, row.nsfw_threshold


# The sort value of an image without a prediction in the "uncertain first"
# order. A real uncertainty is at most 0.5 (a prediction of exactly 0 or 1),
# so 1.0 puts every unpredicted image after every predicted one (contract
# V11): they get their value from the index chain later and slot in then.
UNPREDICTED_UNCERTAINTY = 1.0
UNCERTAINTY_FIELD = "uncertainty"


def uncertainty_of(predicted_score: float | None) -> float:
    """
    Distance of a prediction from the coin flip, the Python twin of
    UNCERTAINTY_SQL (contract V11).

    Both must compute the same double for the same row, because the window
    filters compare the current image's Python value against the SQL
    annotation of the other rows. `abs(x - 0.5)` is one IEEE subtraction and
    one sign flip on both sides, so the values are bit-identical.
    """
    if predicted_score is None:
        return UNPREDICTED_UNCERTAINTY
    return abs(predicted_score - 0.5)


UNCERTAINTY_SQL = Coalesce(
    Abs(F("predicted_score") - Value(0.5)),
    Value(UNPREDICTED_UNCERTAINTY),
    output_field=FloatField(),
)


@dataclass(frozen=True)
class QueueOrder:
    """
    One way of serving a review queue: unseen images first, then by `field`,
    ties broken by content_hash.

    The order_by fields, their reverse (used to find an image's predecessor)
    and the prev/next window filters have to agree exactly, or prev/next
    navigation drifts away from the sequence the user actually sees. Keeping
    all three on one object derived from the same values is what rules that
    drift out.

    queue_seen_at stays the primary key in every mode (contract V3): the
    option only decides how the unseen block and each seen block are sorted
    inside. SQLite sorts NULL first in ASC and last in DESC, which is what
    makes the reversed fields the exact mirror of the forward ones.

    content_hash is the last key in every mode (contract V12): two downloads
    in the same second, or a whole block of images without a prediction, tie
    on `field`, and a window filter built on `field` alone would then neither
    find the neighbours nor count the position. The hash is unique, so the
    compound key never ties.

    `field` is a model column, or the name of an annotation when `expression`
    is set; the queue builders add `annotations()` to the queryset, and
    `value_of()` computes the same value for the current image in Python
    (`python_value`), because that image is fetched on its own and carries no
    annotation.
    """

    field: str
    descending: bool
    expression: object | None = None
    python_value: Callable[[Image], object] | None = None

    def annotations(self) -> dict:
        """What the queue queryset must annotate for `field` to exist in SQL."""
        if self.expression is None:
            return {}
        return {self.field: self.expression}

    def value_of(self, image) -> object:
        """The image's sort value, from the row or from `python_value` for a derived field."""
        if self.python_value is not None:
            return self.python_value(image)
        return getattr(image, self.field)

    def order_fields(self) -> tuple[str, str, str]:
        """order_by arguments for the queue as the user walks it."""
        if self.descending:
            return ("queue_seen_at", f"-{self.field}", "content_hash")
        return ("queue_seen_at", self.field, "content_hash")

    def reversed_order_fields(self) -> tuple[str, str, str]:
        """order_by arguments that walk the queue backwards (predecessor lookups)."""
        if self.descending:
            return ("-queue_seen_at", self.field, "-content_hash")
        return ("-queue_seen_at", f"-{self.field}", "-content_hash")

    def position_filters(self, image) -> tuple[Q, Q]:
        """
        Build (prev_filter, next_filter) Q-pairs that split a review queue
        around `image` in this order.

        Walking the compound key explicitly is what lets callers do windowed
        prev/next lookups instead of materialising the entire queue's hashes:
        the queue grows with every scrape, and a `list(qs.values_list(...))`
        pass would be an O(N) DB scan plus transport on every navigation click.
        """
        value = self.value_of(image)
        if self.descending:
            field_before = Q(**{f"{self.field}__gt": value})
            field_after = Q(**{f"{self.field}__lt": value})
        else:
            field_before = Q(**{f"{self.field}__lt": value})
            field_after = Q(**{f"{self.field}__gt": value})
        same_value = Q(**{self.field: value})
        before_me = field_before | (same_value & Q(content_hash__lt=image.content_hash))
        after_me = field_after | (same_value & Q(content_hash__gt=image.content_hash))

        seen_at = image.queue_seen_at
        unseen = Q(queue_seen_at__isnull=True)
        if seen_at is None:
            # image sits inside the leading unseen block.
            prev_filter = unseen & before_me
            next_filter = (unseen & after_me) | Q(queue_seen_at__isnull=False)
        else:
            # image sits strictly after the unseen block.
            same_seen_time = Q(queue_seen_at=seen_at)
            prev_filter = (
                unseen | Q(queue_seen_at__lt=seen_at) | (same_seen_time & before_me)
            )
            next_filter = Q(queue_seen_at__gt=seen_at) | (same_seen_time & after_me)
        return prev_filter, next_filter


# "shuffle" sorts by the SHA-256 content hash: it carries no information about
# source or download time, so the sequence is as good as random yet stable
# across requests, sessions and restarts (contract V8). A real `order_by("?")`
# would re-roll on every request and break prev/next, the position counter and
# rate-and-advance.
# "uncertain" sorts by the distance of the prediction from 0.5: the images the
# taste model is least sure about come first, the ones it is sure about last,
# unpredicted ones after all of them (contract V11). This is uncertainty
# sampling, the plain form of active learning: a rating near the model's
# boundary moves the boundary, a rating on an image it was already sure about
# teaches it almost nothing.
QUEUE_ORDERS: dict[str, QueueOrder] = {
    ReviewThresholds.ORDER_OLDEST: QueueOrder("downloaded_at", descending=False),
    ReviewThresholds.ORDER_NEWEST: QueueOrder("downloaded_at", descending=True),
    ReviewThresholds.ORDER_SHUFFLE: QueueOrder("content_hash", descending=False),
    ReviewThresholds.ORDER_UNCERTAIN: QueueOrder(
        UNCERTAINTY_FIELD,
        descending=False,
        expression=UNCERTAINTY_SQL,
        python_value=lambda image: uncertainty_of(image.predicted_score),
    ),
}


def normalize_queue_order(value) -> str:
    """Whitelist a queue-order name; anything unknown becomes the default.

    The value arrives from a POST field or from a DB row somebody may have
    edited by hand, and it ends up choosing order_by fields, so it is matched
    against the table instead of being trusted (OWASP A03; contract V9).
    """
    if isinstance(value, str) and value in QUEUE_ORDERS:
        return value
    return ReviewThresholds.ORDER_OLDEST


def get_queue_order() -> QueueOrder:
    """The configured review-queue order, read from the DB singleton."""
    row = review_settings_row()
    return QUEUE_ORDERS[normalize_queue_order(row.queue_order)]


def pred_score_visible(cutoff: float) -> Q:
    """Per-side predicted_score predicate shared by every review-queue filter.

    Two cases keep an image in the queue:
    - predicted_score >= cutoff: the classifier's confidence clears the dial.
    - predicted_score IS NULL: image was never scored (fresh scrape, no weights
      file yet) — show it so the queue isn't silently empty on first install.
    """
    return Q(predicted_score__gte=cutoff) | Q(predicted_score__isnull=True)


def visibility_q(sfw_bucket: int, nsfw_bucket: int, show_nsfw: bool) -> Q:
    """Combined NSFW + threshold Q used by both _review_qs and _counts."""
    sfw_visible = Q(is_nsfw=False) & pred_score_visible(bucket_to_cutoff(sfw_bucket))
    if not show_nsfw:
        return sfw_visible
    nsfw_visible = Q(is_nsfw=True) & pred_score_visible(bucket_to_cutoff(nsfw_bucket))
    return sfw_visible | nsfw_visible


def user_dislike_q() -> Q:
    """User-rated training negatives (trash 0 through score 2)."""
    return Q(score__isnull=False, score__lte=2)


def pred_score_below_cutoff(cutoff: float) -> Q:
    """Unrated rows the classifier scored below the config dial.

    Requires a concrete predicted_score — NULL means "not classified yet" and
    those rows stay in review only, not below-cutoff.
    """
    return (
        Q(score__isnull=True)
        & Q(predicted_score__lt=cutoff)
        & Q(predicted_score__isnull=False)
    )


def below_cutoff_q(sfw_bucket: int, nsfw_bucket: int, show_nsfw: bool) -> Q:
    """Combined Below-tab Q: user dislikes plus unrated model rejects per side."""
    sfw_reject = Q(is_nsfw=False) & pred_score_below_cutoff(
        bucket_to_cutoff(sfw_bucket)
    )
    if not show_nsfw:
        return user_dislike_q() | sfw_reject
    nsfw_reject = Q(is_nsfw=True) & pred_score_below_cutoff(
        bucket_to_cutoff(nsfw_bucket)
    )
    return user_dislike_q() | sfw_reject | nsfw_reject
