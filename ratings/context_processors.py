"""Template context processors for the ratings app."""

from django.conf import settings
from django.contrib import messages
from django.contrib.messages import get_messages


def app_version(request):
    """
    Expose APP_VERSION (resolved once at settings import) to every template
    so the nav footer can show the running build's git tag.
    """
    return {"app_version": getattr(settings, "APP_VERSION", "dev")}


def notification_config(request):
    """
    Inject enabled NotificationChannels into every template context.

    share_channels is the list of configured+enabled channels — used by
    _share_button.html to decide whether to render the share button at all,
    and to populate the channel inputs. A try/except guards against the first
    request before migrations run.
    """
    try:
        from ratings.models import NotificationChannel

        channels = list(
            NotificationChannel.objects.filter(enabled=True).order_by("name")
        )
    except Exception:
        return {}
    return {
        "share_channels": [ch for ch in channels if ch.is_configured],
    }


def server_toasts(request):
    """
    Django messages as toast data for base.html (UI contract V7).

    A full-page POST (clearing the log) cannot deliver an HX-Trigger header to
    a page that is being redirected to, so it flashes a message instead and the
    next page renders it into #server-toasts with json_script; toast.js shows
    it like any htmx toast. Iterating the storage marks the messages consumed,
    which is right because base.html is the only template that renders them.
    """
    kinds = {messages.SUCCESS: "ok", messages.ERROR: "error"}
    toasts = []
    for message in get_messages(request):
        toasts.append({"message": str(message), "kind": kinds.get(message.level, "info")})
    return {"server_toasts": toasts}


def active_jobs(request):
    """
    Background-job state for the nav indicator on every page (UI contract V8).

    The session bookkeeping lives in views next to trigger_train/trigger_scrape
    and their pollers; it is imported lazily here because Django loads this
    module while reading settings, before the app registry is ready.
    """
    if not hasattr(request, "session"):
        return {}
    from ratings import views

    return {
        **views._training_ctx(request),
        **views._scrape_ctx(request),
        **_index_job_ctx(),
        **_reencode_job_ctx(),
    }


def _reencode_job_ctx() -> dict:
    """
    State of the taste re-encode chain for the nav indicator (taste contract
    V21), a twin of _index_job_ctx for the DINOv3 generation with the same
    queue-not-session reasoning and the same quiet failure before django-q's
    tables exist.
    """
    try:
        from ratings import embeddings

        if not embeddings.reencode_job_queued():
            return {"active_reencode": False, "reencode_remaining": 0}
        current, total = embeddings.taste_vector_counts()
        return {"active_reencode": True, "reencode_remaining": total - current}
    except Exception:
        return {"active_reencode": False, "reencode_remaining": 0}


def _index_job_ctx() -> dict:
    """
    State of the search index chain, read from the queue table rather than
    the session (V15): the chain is many short tasks with new IDs, and other
    browsers and the second gunicorn worker must see the same thing. The
    try/except mirrors notification_config: this runs on every page, including
    the first request before django-q's tables exist.
    """
    try:
        from ratings import search

        if not search.index_job_queued():
            return {"active_index": False, "index_remaining": 0}
        indexed, total = search.index_counts()
        return {"active_index": True, "index_remaining": total - indexed}
    except Exception:
        return {"active_index": False, "index_remaining": 0}
