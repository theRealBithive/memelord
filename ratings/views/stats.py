"""The stats dashboard: counts, the score distribution, the per-category training data and the last training run."""

from datetime import timedelta

from django.contrib.auth.decorators import login_required
from django.db.models import Case, CharField, Count, F, Q, Value, When
from django.shortcuts import render
from django.utils import timezone

from core import taste
from ratings.models import Image, Tag
from ratings.views.common import get_taste_model, nav_counts, weights_last_modified
from ratings.views.jobs import task_outcome


def _last_train_info() -> dict | None:
    """
    The most recent django-q training run regardless of outcome, so the page
    can tell "trained successfully yesterday" from "tried this morning and
    crashed" — without it a failed retrain shows the same stale weights mtime
    as before the attempt.
    """
    from django_q.models import Task

    last_task = Task.objects.filter(func="ratings.tasks.run_train").order_by("-stopped").first()
    if last_task is None:
        return None
    ok, error = task_outcome(last_task)
    return {
        "ok": ok,
        "stopped": last_task.stopped,
        "started": last_task.started,
        "error": error,
    }


def _taste_group_breakdown(scored_qs) -> list[dict]:
    """
    Per category: how many liked / disliked rows feed the trainer and whether
    the category has reached its own classifier (taste contract V10, V27).

    The category is the source, or the NSFW group for flagged rows, computed in
    SQL the same way taste.taste_group does it in Python; the split uses the
    trainer's constants so the page never disagrees with the log lines. With
    NSFW hidden the NSFW rows are already filtered out of scored_qs, so the
    group row is absent and the source rows count safe images only, as the
    trainer sees them.
    """
    from core.trainer import HIGH_SCORE, LOW_SCORE

    taste_group_sql = Case(
        When(is_nsfw=True, then=Value(taste.NSFW_GROUP)),
        default=F("source_label"),
        output_field=CharField(),
    )
    breakdown = list(
        scored_qs.annotate(taste_group=taste_group_sql)
        .values("taste_group")
        .annotate(
            n=Count("content_hash"),
            good=Count("content_hash", filter=Q(score__gte=HIGH_SCORE)),
            bad=Count("content_hash", filter=Q(score__lte=LOW_SCORE)),
        )
        .order_by("-n")
    )
    taste_model = get_taste_model()
    groups_with_own_model = set(taste_model.per_source) if taste_model else set()
    for row in breakdown:
        row["own_model"] = row["taste_group"] in groups_with_own_model
    return breakdown


def _avg_inbox_hours() -> int | None:
    """
    Average hours between download and rating, computed in Python: SQLite
    doesn't aggregate timedeltas natively and fetching the (downloaded_at,
    rated_at) pairs is cheap at corpus scale. Pairs where the rating predates
    the download (clock skew, imported rows) are left out.
    """
    pairs = Image.objects.filter(
        score__isnull=False, rated_at__isnull=False, is_purged=False
    ).values_list("downloaded_at", "rated_at")
    durations = [
        (rated - downloaded).total_seconds()
        for downloaded, rated in pairs
        if rated > downloaded
    ]
    if not durations:
        return None
    return round(sum(durations) / len(durations) / 3600)


@login_required
def stats(request):
    """
    Render the stats dashboard.

    Every list view (gallery, below_cutoff, the review queue, the nav counts)
    excludes purged rows, so the charts and counters here do too.
    """
    show_nsfw = request.session.get("show_nsfw", False)

    scored_qs = Image.objects.filter(score__isnull=False, is_purged=False)
    if not show_nsfw:
        scored_qs = scored_qs.filter(is_nsfw=False)

    # The "Gallery" headline and tagging stats mirror the gallery page (score >= 1,
    # i.e. trash excluded); the distribution chart below still shows the 0 bucket.
    gallery_qs = scored_qs.filter(score__gte=1)
    gallery_total = gallery_qs.count()
    # Disjoint from below_cutoff_count (score <= 2) and matches the trainer's split,
    # so the "Training data" line on the page doesn't double-count scores 0-2.
    positive_count = scored_qs.filter(score__gte=3).count()

    score_dist = list(
        scored_qs.values("score").annotate(n=Count("content_hash")).order_by("-score")
    )
    score_dist_max = max((row["n"] for row in score_dist), default=1)

    seven_days_ago = timezone.now() - timedelta(days=7)
    scraped_7d = Image.objects.filter(
        downloaded_at__gte=seven_days_ago, is_purged=False
    ).count()
    rated_7d = Image.objects.filter(
        rated_at__gte=seven_days_ago, score__isnull=False, is_purged=False
    ).count()

    tag_breakdown = list(
        Tag.objects.annotate(n=Count("images")).filter(n__gt=0).order_by("-n")[:20]
    )
    tagged_count = gallery_qs.filter(tags__isnull=False).distinct().count()

    return render(
        request,
        "ratings/stats.html",
        {
            **nav_counts(show_nsfw),
            "show_nsfw": show_nsfw,
            "last_trained": weights_last_modified(),
            "last_train": _last_train_info(),
            "gallery_total": gallery_total,
            "positive_count": positive_count,
            "mode": "stats",
            "score_dist": score_dist,
            "score_dist_max": score_dist_max,
            "source_breakdown": _taste_group_breakdown(scored_qs),
            "scraped_7d": scraped_7d,
            "rated_7d": rated_7d,
            "tag_breakdown": tag_breakdown,
            "tagged_count": tagged_count,
            "untagged_count": gallery_total - tagged_count,
            "avg_inbox_hours": _avg_inbox_hours(),
        },
    )
