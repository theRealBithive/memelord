"""Single source of truth for review and below-cutoff queue filters.

The 1-6 dial on /config maps to a predicted_score cutoff. Unrated images at
or above the cutoff go to /review/; unrated images below it go to /below-cutoff/
so nothing is silently dropped. User-rated scores <= 2 also stay in below-cutoff.
"""

from dataclasses import dataclass

from django.db.models import Q

from ratings.models import ReviewThresholds


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


@dataclass(frozen=True)
class QueueOrder:
    """
    One way of serving a review queue: unseen images first, then by `field`.

    The order_by fields, their reverse (used to find an image's predecessor)
    and the prev/next window filters have to agree exactly, or prev/next
    navigation drifts away from the sequence the user actually sees. Keeping
    all three on one object derived from the same two values is what rules
    that drift out.

    queue_seen_at stays the primary key in every mode (contract V3): the
    option only decides how the unseen block and each seen block are sorted
    inside. SQLite sorts NULL first in ASC and last in DESC, which is what
    makes the reversed fields the exact mirror of the forward ones.
    """

    field: str
    descending: bool

    def order_fields(self) -> tuple[str, str]:
        """order_by arguments for the queue as the user walks it."""
        if self.descending:
            return ("queue_seen_at", f"-{self.field}")
        return ("queue_seen_at", self.field)

    def reversed_order_fields(self) -> tuple[str, str]:
        """order_by arguments that walk the queue backwards (predecessor lookups)."""
        if self.descending:
            return ("-queue_seen_at", self.field)
        return ("-queue_seen_at", f"-{self.field}")

    def position_filters(self, image) -> tuple[Q, Q]:
        """
        Build (prev_filter, next_filter) Q-pairs that split a review queue
        around `image` in this order.

        Walking the compound key explicitly is what lets callers do windowed
        prev/next lookups instead of materialising the entire queue's hashes:
        the queue grows with every scrape, and a `list(qs.values_list(...))`
        pass would be an O(N) DB scan plus transport on every navigation click.
        """
        value = getattr(image, self.field)
        if self.descending:
            before_me = Q(**{f"{self.field}__gt": value})
            after_me = Q(**{f"{self.field}__lt": value})
        else:
            before_me = Q(**{f"{self.field}__lt": value})
            after_me = Q(**{f"{self.field}__gt": value})

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
QUEUE_ORDERS: dict[str, QueueOrder] = {
    ReviewThresholds.ORDER_OLDEST: QueueOrder("downloaded_at", descending=False),
    ReviewThresholds.ORDER_NEWEST: QueueOrder("downloaded_at", descending=True),
    ReviewThresholds.ORDER_SHUFFLE: QueueOrder("content_hash", descending=False),
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
