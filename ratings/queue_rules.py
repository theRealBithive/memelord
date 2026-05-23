"""Single source of truth for what makes an image visible in /review/.

Two systems make "is this image worth showing?" decisions and they MUST
stay aligned, so they live in one file:

1. classify_inbox uses AUTO_PROMOTE_THRESHOLD / AUTO_TRASH_THRESHOLD to
   pre-route incoming images before the user sees them at all — anything
   the classifier is confident about skips manual review.
2. The 1-6 dial on /config uses bucket_to_cutoff + pred_score_visible to
   further hide low-confidence rows so the user only reviews things the
   model thinks they'll like.

The relationship is load-bearing. AUTO_PROMOTE_THRESHOLD (0.75) is below
the strictest dial cutoff (bucket_to_cutoff(6) ≈ 0.833), so without the
location=CORPUS escape hatch in pred_score_visible, auto-promoted images
with predicted_score in [0.75, 0.833) would silently vanish from /review/
forever at dial=6 — unrated, in corpus, invisible. Keep the two systems
co-located so the next person tweaking either remembers the other exists.
"""

from django.db.models import Q

from ratings.models import Image, ReviewThresholds


AUTO_PROMOTE_THRESHOLD = 0.75
"""classify_inbox routes inbox → corpus when predicted_score >= this."""

AUTO_TRASH_THRESHOLD = 0.25
"""classify_inbox routes inbox → void when predicted_score <= this."""


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

    Three escape hatches keep an image in the queue:
    - predicted_score >= cutoff: the classifier's confidence clears the dial.
    - predicted_score IS NULL: image was never scored (fresh scrape, no weights
      file yet) — show it so the queue isn't silently empty on first install.
    - location = CORPUS: classify_inbox auto-promotes at AUTO_PROMOTE_THRESHOLD,
      which sits below bucket_to_cutoff(6); without this hatch, auto-promoted
      images in [AUTO_PROMOTE_THRESHOLD, bucket_to_cutoff(6)) would vanish at
      dial=6. Anything in corpus with score=NULL has already been judged worth
      the user's time and must always be reachable.
    """
    return (
        Q(predicted_score__gte=cutoff)
        | Q(predicted_score__isnull=True)
        | Q(location=Image.CORPUS)
    )


def visibility_q(sfw_bucket: int, nsfw_bucket: int, show_nsfw: bool) -> Q:
    """Combined NSFW + threshold Q used by both _review_qs and _counts."""
    sfw_visible = Q(is_nsfw=False) & pred_score_visible(bucket_to_cutoff(sfw_bucket))
    if not show_nsfw:
        return sfw_visible
    nsfw_visible = Q(is_nsfw=True) & pred_score_visible(bucket_to_cutoff(nsfw_bucket))
    return sfw_visible | nsfw_visible
