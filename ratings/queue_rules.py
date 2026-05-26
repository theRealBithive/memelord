"""Single source of truth for review and below-cutoff queue filters.

The 1-6 dial on /config maps to a predicted_score cutoff. Unrated images at
or above the cutoff go to /review/; unrated images below it go to /below-cutoff/
so nothing is silently dropped. User-rated scores <= 2 also stay in below-cutoff.
"""

from django.db.models import Q

from ratings.models import ReviewThresholds


def bucket_to_cutoff(bucket: int) -> float:
    """Map a 1-6 threshold dial to an inclusive probability lower bound.

    Bucket 1 means "show everything" (cutoff 0.0); bucket 6 is the strictest.
    Images with predicted_score >= cutoff pass the visibility filter. Out-of-
    range buckets are clamped because the value comes from user-edited TOML.
    """
    return (max(1, min(6, int(bucket))) - 1) / 6


def get_review_thresholds() -> tuple[int, int]:
    """Return (sfw_bucket, nsfw_bucket) from the DB singleton, seeding from settings on first access.

    DB-first so UI edits on /config persist immediately; config.toml values
    only matter on the very first call (or after the row is manually deleted)
    because get_or_create writes them once and never re-reads.
    """
    from django.conf import settings

    row, _ = ReviewThresholds.objects.get_or_create(
        pk=1,
        defaults={
            "sfw_threshold": getattr(settings, "SFW_THRESHOLD_BUCKET", 1),
            "nsfw_threshold": getattr(settings, "NSFW_THRESHOLD_BUCKET", 1),
        },
    )
    return row.sfw_threshold, row.nsfw_threshold


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
