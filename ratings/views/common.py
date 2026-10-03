"""
Helpers every view module shares: the taste-model cache and the lazy
prediction, the nav badge counts, score parsing and writing, form-int
clamping, plus the two tiny views that belong to no area (the index redirect
and the show-NSFW switch).

Names other view modules import from here are public; the module-private
names are the cache cells the tests reset between runs.
"""

from datetime import UTC, datetime
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.shortcuts import redirect
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from core import taste
from ratings import features
from ratings.models import Image
from ratings.queue_rules import (
    below_cutoff_q,
    bucket_to_cutoff,
    get_review_thresholds,
    pred_score_visible,
)

WEIGHTS_PATH = Path(settings.WEIGHTS_PATH)
DATA_DIR = Path(settings.DATA_DIR)

_taste_model_cache = None
_taste_model_mtime: float | None = None


def get_taste_model():
    """
    Load (and cache) the taste model, reloading if weights change on disk.

    The cached value may legitimately be None: load_taste_model returns None
    for a file trained on another encoder (V5). Keying the reload on the file's
    mtime alone, rather than on "cache is None", keeps that None from being
    unpickled again on every render. A retrain writes a new file, so its new
    per-source dict arrives with the next mtime change.
    """
    global _taste_model_cache, _taste_model_mtime
    if not WEIGHTS_PATH.exists():
        return None
    mtime = WEIGHTS_PATH.stat().st_mtime
    if mtime != _taste_model_mtime:
        _taste_model_cache = taste.load_taste_model(WEIGHTS_PATH)
        _taste_model_mtime = mtime
    return _taste_model_cache


def weights_last_modified() -> datetime | None:
    """
    When the taste weights were last saved, or None before the first training.

    The weights file is the ground truth for "last successful save": a DB
    timestamp could drift if the file was replaced out of band.
    """
    try:
        mtime = WEIGHTS_PATH.stat().st_mtime
    except FileNotFoundError:
        return None
    return datetime.fromtimestamp(mtime, tz=UTC)


def taste_prediction(image) -> int | None:
    """Return P(corpus) as integer percentage 0–100, or None if unavailable.

    Prefers the stored ``predicted_score`` — classify_images() writes it at
    scrape time, so the common case is a single float read. Without this short
    circuit every review/gallery render re-ran predict_proba and copied the
    ~3 KB embedding blob via ``bytes(image.embedding)``, dominating the
    per-page cost. Legacy rows (embedded before a classifier existed) take
    the fallback once and persist the result, so subsequent renders are fast.
    """
    if image is None:
        return None
    if image.predicted_score is not None:
        return round(float(image.predicted_score) * 100)
    # A vector from an older encoder must never meet the current classifier
    # (V2), and a row without its SigLIP2 half has no feature yet (taste
    # contract V13); either way the row stays unpredicted until it is encoded.
    if not features.has_taste_features(image):
        return None
    taste_model = get_taste_model()
    if taste_model is None:
        return None
    from core import brain

    group = taste.taste_group(image.source_label, image.is_nsfw)
    classifier = taste_model.classifier_for(group)
    prob = float(brain.predict_proba(classifier, features.taste_features(image)))
    image.predicted_score = prob
    image.save(update_fields=["predicted_score"])
    return round(prob * 100)


def flip_nsfw_and_repredict(image) -> None:
    """
    Toggle is_nsfw and renew the taste prediction for the new category (taste
    contract V26).

    The flag decides the image's rating category (source or the NSFW group),
    so the stored predicted_score came from the wrong model the moment the
    flag changed. Dropping it and calling taste_prediction re-runs the one
    predict_proba against the cached feature with the cached model, or leaves
    NULL ("show anyway") when either is missing. Rated images get the same
    treatment: the stored value is cheap to refresh and the gallery shows it.
    """
    image.is_nsfw = not image.is_nsfw
    # A hand toggle is a person's decision in either direction (NSFW contract
    # N2); from here on the model leaves this flag alone (N3).
    image.nsfw_judged = True
    image.predicted_score = None
    image.save(update_fields=["is_nsfw", "nsfw_judged", "predicted_score"])
    taste_prediction(image)


def nav_counts(show_nsfw: bool = False) -> dict:
    """
    Aggregate image counts across all queues and locations in a single DB query.

    Used by every page for nav badges; a separate query per badge would be 6×
    the DB round-trips per request. show_nsfw controls whether NSFW images are
    folded into the main counts or kept separate so the user can see SFW and
    NSFW numbers independently. The queue counts respect the [vision]
    threshold so the badge matches what the user will actually see in the
    review queue — a stale "12 to review" badge that opens onto an empty
    page would be worse than no badge at all.
    """
    sfw_bucket, nsfw_bucket = get_review_thresholds()
    sfw_pred_visible = pred_score_visible(bucket_to_cutoff(sfw_bucket))
    nsfw_pred_visible = pred_score_visible(bucket_to_cutoff(nsfw_bucket))

    qs = Image.objects.filter(is_purged=False)
    sfw_queue = Q(score__isnull=True, is_nsfw=False) & sfw_pred_visible
    nsfw_queue = Q(score__isnull=True, is_nsfw=True) & nsfw_pred_visible
    below_q = below_cutoff_q(sfw_bucket, nsfw_bucket, show_nsfw=True)

    if show_nsfw:
        return qs.aggregate(
            queue_count=Count("pk", filter=sfw_queue | nsfw_queue),
            below_cutoff_count=Count("pk", filter=below_q),
            nsfw_queue_count=Count("pk", filter=nsfw_queue),
        )
    return qs.aggregate(
        queue_count=Count("pk", filter=sfw_queue),
        below_cutoff_count=Count("pk", filter=below_q & Q(is_nsfw=False)),
        nsfw_queue_count=Count("pk", filter=nsfw_queue),
    )


def int_in_range(raw, low: int, high: int, default: int) -> int:
    """
    Parse a form or query field as an int clamped to [low, high].

    A missing or unparsable value becomes the default instead of an error: the
    forms only ever send valid numbers, so a bad value means a stale page or a
    crafted request, and neither may store or query an out-of-range setting
    (OWASP A03).
    """
    try:
        value = int(raw)
    except (ValueError, TypeError):
        return default
    return max(low, min(high, value))


def score_from_post(request) -> int | None:
    """
    The score a form submitted, or None when it is missing or outside 0–6.

    0 is "trash": below the 1–6 scale but a real rating. Anything else (a
    missing field, text, 7) is ignored instead of answered with an error: the
    buttons only ever send 0–6, so a bad value means a stale page, and
    re-rendering the current state is the right answer to that.
    """
    try:
        value = int(request.POST.get("score", -1))
    except (ValueError, TypeError):
        return None
    if 0 <= value <= 6:
        return value
    return None


def apply_score(image: Image, score: int) -> None:
    """
    Write a rating as one DB update; the file never moves (the rating model).

    rated_at is stamped together with the score so the stats dwell time and
    the re-encode order ("most recently rated first") see the same moment.

    A rating also records that a person decided the NSFW flag as it stands
    (NSFW contract N2): whoever rates has looked at the picture, so a flag
    left on is confirmed and a flag left off is a safe example for the head.
    """
    image.score = score
    image.rated_at = timezone.now()
    image.nsfw_judged = True
    image.save(update_fields=["score", "rated_at", "nsfw_judged"])


@login_required
def index(request):
    return redirect("review_corpus")


@login_required
@require_POST
def nsfw_toggle(request):
    """
    Session toggle for the show-NSFW preference; redirects back to the page
    the switch was pressed on.

    The Referer header is client-supplied, so it is only followed when it
    points at this host (OWASP A01: unvalidated redirect); anything else
    falls back to the index.
    """
    request.session["show_nsfw"] = not request.session.get("show_nsfw", False)
    referer = request.META.get("HTTP_REFERER", "")
    referer_is_local = url_has_allowed_host_and_scheme(
        referer, allowed_hosts={request.get_host()}, require_https=request.is_secure()
    )
    if referer_is_local:
        return redirect(referer)
    return redirect("index")
