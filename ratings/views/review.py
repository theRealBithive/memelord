"""
The review queues (SFW and NSFW): the card with prev/next navigation, and the
rate / purge / NSFW-flip actions that advance to the next image.
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from ratings.models import Image
from ratings.queue_rules import (
    QueueOrder,
    bucket_to_cutoff,
    get_queue_order,
    get_review_thresholds,
    pred_score_visible,
    visibility_q,
)
from ratings.toast import with_toast
from ratings.utils import purge_image
from ratings.views.common import (
    apply_score,
    flip_nsfw_and_repredict,
    nav_counts,
    score_from_post,
    taste_prediction,
)

# How many upcoming queue images the card preloads (review latency contract
# R2). One is the image rate-and-advance shows next; the second covers a quick
# second rating while a large first one is still downloading. More would only
# cost bandwidth on images the user may never reach.
PRELOAD_AHEAD = 2


def _queue_neighbor_hash(qs, image: Image, order: QueueOrder) -> str | None:
    """
    Next content_hash after `image` in queue order, falling back to the prior
    one if image is the tail, or None for an empty queue.

    Used by score/purge/toggle for rate-and-advance navigation; callers should
    only invoke this once they've confirmed `image` belongs to `qs` (a one-row
    PK `.exists()` check), otherwise an out-of-queue caller will navigate
    relative to its stale queue position instead of staying put. `order` must
    be the one `qs` was built with, otherwise the window filters split the
    queue differently from how it is sorted.
    """
    prev_filter, next_filter = order.position_filters(image)
    next_hash = (
        qs.filter(next_filter).values_list("content_hash", flat=True).first()
    )
    if next_hash is not None:
        return next_hash
    # The reversed fields walk the queue backwards, so `.first()` here is the
    # immediate predecessor regardless of whether it sits in the unseen block.
    return (
        qs.filter(prev_filter)
        .order_by(*order.reversed_order_fields())
        .values_list("content_hash", flat=True)
        .first()
    )


def _browse_ctx(
    qs,
    content_hash: str | None,
    mode: str,
    show_nsfw: bool,
    request,
    order: QueueOrder,
    extra: dict | None = None,
) -> dict:
    """
    Build browse context for a single image, with prev/next navigation when the
    image is part of the queue.

    Prev/next/position/total are resolved with windowed lookups against the
    queue's (queue_seen_at, secondary key) ordering (see
    `QueueOrder.position_filters`) instead of pulling every queued content_hash
    into Python on each request. `order` must be the one `qs` was sorted with.
    Two `.first()`s and two `.count()`s scale with the queue much better than
    the old materialise-and-index pass.

    Navigation stays correct under concurrent edits: if a sibling is rated or
    purged between requests, the windowed query simply finds whichever real
    neighbour exists at query time. The stronger "consistent snapshot" claim
    of the old code only ever held within a single request anyway.

    A requested content_hash that is NOT in the queue (e.g. an already-scored
    image opened from the gallery for re-review) is still shown — standalone,
    with no queue position and prev/next disabled — instead of silently
    falling back to the head of the queue and showing the wrong picture.
    """
    base = {"show_nsfw": show_nsfw, **nav_counts(show_nsfw)}
    if extra:
        base.update(extra)

    image = None
    in_queue = False
    if content_hash:
        # Try the queue first; in the common in-queue case this saves the
        # separate "does it exist anywhere?" lookup.
        image = (
            qs.filter(content_hash=content_hash).prefetch_related("tags").first()
        )
        in_queue = image is not None
        if image is None:
            image = (
                Image.objects.filter(content_hash=content_hash)
                .prefetch_related("tags")
                .first()
            )

    if image is not None and not in_queue:
        return {
            "image": image,
            "prev_hash": None,
            "next_hash": None,
            "preload_paths": [],
            "position": None,
            "total": None,
            "mode": mode,
            "prediction": taste_prediction(image),
            **base,
        }

    # No content_hash, or the requested image is gone (purged/deleted) — show
    # the head of the queue so the user lands on something predictable.
    if image is None:
        image = qs.prefetch_related("tags").first()

    if image is None:
        return {"image": None, "mode": mode, "preload_paths": [], **base}

    prev_filter, next_filter = order.position_filters(image)
    prev_qs = qs.filter(prev_filter)
    next_qs = qs.filter(next_filter)
    prev_hash = (
        prev_qs.order_by(*order.reversed_order_fields())
        .values_list("content_hash", flat=True)
        .first()
    )
    # One query for the successor and the preload list: the first upcoming
    # row is where rate-and-advance and the next arrow go.
    upcoming = list(next_qs.values_list("content_hash", "file_path")[:PRELOAD_AHEAD])
    next_hash = upcoming[0][0] if upcoming else None
    preload_paths = [file_path for _content_hash, file_path in upcoming]
    prev_count = prev_qs.count()
    # Derive total from the two halves + the image itself to skip a third
    # COUNT(*) on the queue.
    total = prev_count + 1 + next_qs.count()
    return {
        "image": image,
        "prev_hash": prev_hash,
        "next_hash": next_hash,
        "preload_paths": preload_paths,
        "position": prev_count + 1,
        "total": total,
        "mode": mode,
        "prediction": taste_prediction(image),
        **base,
    }


def _review_qs(show_nsfw: bool = False, order: QueueOrder | None = None):
    """
    Build the ordered queue for the primary review flow.

    Unscored images (score IS NULL) feed the queue. Unseen images
    (queue_seen_at IS NULL) sort first in SQLite ASC; inside that block the
    configured order (oldest / newest download, shuffled, or least certain
    prediction first) decides. The [vision] threshold further hides
    low-confidence images so the user only reviews things the model thinks
    they'll like. The order's annotations ride along so a derived sort key
    (the uncertainty) exists for order_by and for the window filters.

    `order` defaults to the DB setting; callers that also build navigation
    pass the one they already fetched so queue and window filters agree.
    """
    if order is None:
        order = get_queue_order()
    sfw_bucket, nsfw_bucket = get_review_thresholds()
    return (
        Image.objects.filter(score__isnull=True, is_purged=False)
        .filter(visibility_q(sfw_bucket, nsfw_bucket, show_nsfw))
        .annotate(**order.annotations())
        .order_by(*order.order_fields())
    )


def _review_nsfw_qs(show_nsfw: bool = False, order: QueueOrder | None = None):
    """
    Same queue shape as _review_qs but filtered to NSFW images only.

    show_nsfw is accepted but ignored — this queue is always NSFW-only by
    definition. The parameter exists so qs_fn callers can treat both queues
    with the same (show_nsfw, order) → QuerySet signature.
    """
    if order is None:
        order = get_queue_order()
    _, nsfw_bucket = get_review_thresholds()
    return (
        Image.objects.filter(is_nsfw=True, score__isnull=True, is_purged=False)
        .filter(pred_score_visible(bucket_to_cutoff(nsfw_bucket)))
        .annotate(**order.annotations())
        .order_by(*order.order_fields())
    )


def _review_ctx(
    content_hash: str | None, show_nsfw: bool = False, request=None
) -> dict:
    """Build browse context for the primary corpus review queue."""
    order = get_queue_order()
    return _browse_ctx(
        _review_qs(show_nsfw, order),
        content_hash,
        "corpus",
        show_nsfw,
        request,
        order,
        extra={
            "scores": range(1, 7),
            "score_url": "score_corpus",
            "nsfw_url": "toggle_nsfw",
            "hx_target": "#review-card",
            "hx_swap": "outerHTML",
            "purge_url": "purge_corpus",
            "image_url": "review_corpus_image",
        },
    )


def _review_nsfw_ctx(
    content_hash: str | None, show_nsfw: bool = False, request=None
) -> dict:
    """Build browse context for the NSFW corpus review queue."""
    order = get_queue_order()
    return _browse_ctx(
        _review_nsfw_qs(show_nsfw, order),
        content_hash,
        "nsfw_corpus",
        show_nsfw,
        request,
        order,
        extra={
            "scores": range(1, 7),
            "score_url": "score_nsfw_corpus",
            "nsfw_url": "toggle_nsfw",
            "hx_target": "#review-card",
            "hx_swap": "outerHTML",
            "purge_url": "purge_nsfw_corpus",
            "image_url": "review_nsfw_corpus_image",
        },
    )


def _mark_left_image_seen(request) -> None:
    """
    Stamp the image the user just moved on from, named by `?left=<hash>` on a
    prev/next navigation, so it drops behind every unseen image (contract V3).

    Stamping on leave rather than on render keeps the card and the DB in step:
    prev/next and the position were computed for the state the row had at
    render time, and rate-and-advance reads that same state afterwards. With
    the earlier stamp-at-render the row moved to the back while its card still
    showed the unseen position, so scoring the first image of a session jumped
    to the far end of the unseen block instead of to its successor. A reload
    of the card is not "moving on", so it no longer skips the image either.

    `left` is client input used only as a parametrised primary-key match
    (OWASP A03); the only effect is an idempotent seen-stamp on an unrated row,
    so a forged value can do no more than re-order the queue.
    """
    left = request.GET.get("left")
    if not left:
        return
    Image.objects.filter(
        content_hash=left, score__isnull=True, queue_seen_at__isnull=True
    ).update(queue_seen_at=timezone.now())


def _render_review(request, ctx: dict):
    """Full page on a plain visit, the card alone on an htmx navigation."""
    if request.htmx:
        return render(request, "ratings/_review_htmx.html", ctx)
    return render(request, "ratings/review.html", ctx)


@login_required
def review_corpus(request, content_hash: str | None = None):
    """
    Main corpus review page — full render on first visit, HTMX partial on navigation.

    The image being left (prev/next carry `?left=`) is stamped before the new
    card is built, so the unseen-first ordering persists across page loads and
    the new card already reflects the skipped image's move to the back.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    _mark_left_image_seen(request)
    return _render_review(request, _review_ctx(content_hash, show_nsfw, request))


@login_required
def rate_nsfw_corpus(request, content_hash: str | None = None):
    """
    Entry point for the NSFW corpus review queue — same layout as review_corpus
    but filtered to is_nsfw=True images.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    _mark_left_image_seen(request)
    return _render_review(request, _review_nsfw_ctx(content_hash, show_nsfw, request))


def _score_impl(request, content_hash: str, qs_fn, ctx_fn):
    """
    Shared score logic for both the normal and NSFW review queues.

    Scoring is allowed on any image, not just unscored ones, so an already-rated
    image opened from the gallery for re-review can be re-scored here. Navigation
    differs by origin: rating an *unrated* image is a rate-and-advance, so it
    moves to the queue neighbour; re-scoring an *already-rated* image (gallery
    re-review) stays put so the user sees the updated score instead of being
    thrown into the rate queue. next_hash is captured before scoring because
    scoring changes which images are in-queue.

    The advance/stay decision keys on whether the image was already rated, NOT
    on current queue membership: `taste_prediction` lazily writes a
    predicted_score while rendering the card, which can drop a just-shown
    unrated image below the vision cutoff and out of the queue. Keying on
    queue membership there would wrongly treat it as a re-review and re-render
    the same image, stalling the rate flow (the user has to tap again).
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    order = get_queue_order()
    qs = qs_fn(show_nsfw, order)
    if image.score is None:
        next_hash = _queue_neighbor_hash(qs, image, order)
    else:
        next_hash = content_hash
    score_val = score_from_post(request)
    if score_val is not None:
        apply_score(image, score_val)
    ctx = ctx_fn(next_hash, show_nsfw, request)
    return render(request, "ratings/_review_htmx.html", ctx)


def _purge_impl(request, content_hash: str, qs_fn, ctx_fn):
    """
    Shared purge logic for both the normal and NSFW review queues.

    The neighbour is captured first so we know where to navigate after the
    image is hard-deleted from disk. An out-of-queue purge (rare — only when
    re-reviewing an already-scored image from the gallery) falls back to the
    head of the queue, preserving the legacy `_neighbor_hash` behaviour.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    order = get_queue_order()
    qs = qs_fn(show_nsfw, order)
    if qs.filter(content_hash=content_hash).exists():
        next_hash = _queue_neighbor_hash(qs, image, order)
    else:
        next_hash = qs.values_list("content_hash", flat=True).first()
    purge_image(image)
    ctx = ctx_fn(next_hash, show_nsfw, request)
    response = render(request, "ratings/_review_htmx.html", ctx)
    return with_toast(response, "Image purged")


@login_required
@require_POST
def score_corpus(request, content_hash: str):
    return _score_impl(request, content_hash, _review_qs, _review_ctx)


@login_required
@require_POST
def purge_corpus(request, content_hash: str):
    return _purge_impl(request, content_hash, _review_qs, _review_ctx)


@login_required
@require_POST
def score_nsfw_corpus(request, content_hash: str):
    return _score_impl(request, content_hash, _review_nsfw_qs, _review_nsfw_ctx)


@login_required
@require_POST
def purge_nsfw_corpus(request, content_hash: str):
    return _purge_impl(request, content_hash, _review_nsfw_qs, _review_nsfw_ctx)


def _flip_removed_image_from_queue(mode: str, image: Image, show_nsfw: bool) -> bool:
    """
    After the NSFW flip, is the image gone from the queue it was shown in?

    NSFW queue: marking safe removes it — it is by definition no longer in the
    NSFW queue whatever show_nsfw says. Normal queue: marking NSFW only removes
    it while NSFW is hidden; with show_nsfw on it stays visible.
    """
    if mode == "nsfw_corpus":
        return not image.is_nsfw
    return image.is_nsfw and not show_nsfw


@login_required
@require_POST
def toggle_nsfw(request, content_hash: str):
    """
    Toggle is_nsfw on an image, then navigate appropriately for the current mode.

    Navigation mirrors _score_impl: an in-queue image advances to its neighbour
    when the toggle removes it from the current view; an out-of-queue image (an
    already-scored picture opened from the gallery for re-review) stays put so
    the user keeps seeing it instead of being teleported elsewhere.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    image = get_object_or_404(Image, content_hash=content_hash)
    mode = request.POST.get("mode", "corpus")
    order = get_queue_order()

    if mode == "nsfw_corpus":
        qs_fn, ctx_fn = _review_nsfw_qs, _review_nsfw_ctx
    else:
        qs_fn, ctx_fn = _review_qs, _review_ctx
    qs = qs_fn(show_nsfw, order)
    in_queue = qs.filter(content_hash=content_hash).exists()
    neighbor = _queue_neighbor_hash(qs, image, order) if in_queue else None

    flip_nsfw_and_repredict(image)

    if in_queue and _flip_removed_image_from_queue(mode, image, show_nsfw):
        target = neighbor
    else:
        target = content_hash
    ctx = ctx_fn(target, show_nsfw, request)
    return render(request, "ratings/_review_htmx.html", ctx)
