"""
The Config page and everything it saves: review-queue settings, the
auto-scrape schedule, scrape sources, notification channels, and sharing an
image to those channels.
"""

from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST

import ratings.notifiers as notifiers
from ratings.models import (
    Image,
    NotificationChannel,
    ReviewThresholds,
    ScrapeSchedule,
    Source,
)
from ratings.queue_rules import normalize_queue_order, review_settings_row
from ratings.toast import with_toast
from ratings.views.common import DATA_DIR, int_in_range, nav_counts
from ratings.views.jobs import (
    classify_status_ctx,
    index_status_ctx,
    reencode_status_ctx,
)
from retina import flickr

_INTERVAL_CHOICES = [1, 2, 4, 6, 12, 24, 48, 72, 168]


def _vision_ctx() -> dict:
    """Review-queue form state for the /config page (DB singleton via get_or_create)."""
    row = review_settings_row()
    return {
        "vision_sfw": row.sfw_threshold,
        "vision_nsfw": row.nsfw_threshold,
        "vision_buckets": range(1, 7),
        "queue_order": normalize_queue_order(row.queue_order),
        "queue_order_choices": ReviewThresholds.ORDER_CHOICES,
    }


def _schedule_ctx() -> dict:
    """
    Build schedule context by joining the user-facing ScrapeSchedule row with the
    live django-q Schedule entry. They are kept as separate records because
    ScrapeSchedule stores user intent while the q-schedule stores the actual
    next_run timestamp and worker state.
    """
    from django_q.models import Schedule as QSchedule

    schedule = ScrapeSchedule.objects.filter(pk=1).first()
    q = QSchedule.objects.filter(name="auto_scrape").first()
    return {
        "schedule": schedule,
        "next_run": q.next_run if q else None,
        "interval_choices": _INTERVAL_CHOICES,
    }


def _channel_list_ctx() -> dict:
    """Channel list context for the config page sharing section."""
    return {"channels": list(NotificationChannel.objects.all())}


@login_required
def config_view(request):
    """Render the configuration page combining sources, schedule, channels, and job status."""
    show_nsfw = request.session.get("show_nsfw", False)
    return render(
        request,
        "ratings/config.html",
        {
            "mode": "config",
            "sources": Source.objects.all(),
            "show_nsfw": show_nsfw,
            **nav_counts(show_nsfw),
            **_schedule_ctx(),
            **_vision_ctx(),
            **_channel_list_ctx(),
            **index_status_ctx(),
            **reencode_status_ctx(),
            **classify_status_ctx(),
        },
    )


@login_required
@require_POST
def set_vision_thresholds(request):
    """
    Persist the review-queue settings (SFW/NSFW hide thresholds and queue
    order) from the config page.

    Both thresholds are clamped to [1, 6] so a malformed POST can't disable
    the review queue with an out-of-range value, and the order is matched
    against the whitelist so an unknown name can never reach order_by (OWASP
    A03; a missing or unknown value is stored as the default, contract V9).
    update_or_create writes the singleton in one statement.
    """
    sfw = int_in_range(request.POST.get("sfw_threshold"), 1, 6, 1)
    nsfw = int_in_range(request.POST.get("nsfw_threshold"), 1, 6, 1)
    queue_order = normalize_queue_order(request.POST.get("queue_order"))
    ReviewThresholds.objects.update_or_create(
        pk=1,
        defaults={
            "sfw_threshold": sfw,
            "nsfw_threshold": nsfw,
            "queue_order": queue_order,
        },
    )
    show_nsfw = request.session.get("show_nsfw", False)
    return render(
        request,
        "ratings/_vision_thresholds_htmx.html",
        {
            **_vision_ctx(),
            **nav_counts(show_nsfw),
            "show_nsfw": show_nsfw,
            "mode": "config",
        },
    )


@login_required
@require_POST
def set_scrape_schedule(request):
    """
    Persist scrape schedule settings and sync the django-q cron entry.

    ScrapeSchedule (pk=1 singleton) stores user intent; sync_scrape_q_schedule
    then creates or updates the actual django-q Schedule record so the worker
    picks up the new interval without a restart. Pressing Enable starts the
    interval from now (schedule contract V1): the next automatic scrape is one
    interval away, not immediate, which is what the old delete-and-recreate
    did by accident.
    """
    interval_hours = int_in_range(request.POST.get("interval_hours"), 1, 168, 6)
    enabled = request.POST.get("enabled") == "1"

    ScrapeSchedule.objects.update_or_create(
        pk=1,
        defaults={"interval_hours": interval_hours, "enabled": enabled},
    )

    from ratings.schedule_sync import sync_scrape_q_schedule

    sync_scrape_q_schedule(restart_interval=True)

    return render(request, "ratings/_schedule_status.html", _schedule_ctx())


# ── Sources ──────────────────────────────────────────────────────────────────


def _source_error(request, error: str):
    return render(request, "ratings/_source_error.html", {"error": error})


@login_required
@require_POST
def source_add(request):
    """Validate and create a new scrape source, re-enabling it if previously disabled."""
    stype = request.POST.get("type", "").strip()
    name = request.POST.get("name", "").strip()

    if stype not in dict(Source.TYPE_CHOICES):
        return _source_error(request, "Invalid source type.")
    if not name:
        return _source_error(request, "Name is required.")
    # Mastodon and Pixelfed both target an account via the same @user@instance
    # handle (both speak the Mastodon-compatible API), so they validate alike.
    if stype in (Source.MASTODON, Source.PIXELFED) and "@" not in name.lstrip("@"):
        service = dict(Source.TYPE_CHOICES)[stype]
        host = "pixelfed.social" if stype == Source.PIXELFED else "mastodon.social"
        return _source_error(
            request, f"{service} handle must include an instance, e.g. @user@{host}"
        )

    # Flickr: every spelling of one group or user (pasted URL, short form) is
    # stored under one canonical name, so the unique (type, name) pair also
    # catches duplicates. The whitelist here is the input check (OWASP A03).
    if stype == Source.FLICKR:
        canonical = flickr.normalize_source(name)
        if canonical is None:
            return _source_error(
                request,
                "Flickr source must be a group or user URL, e.g. "
                "https://www.flickr.com/groups/419512@N22/pool/ or group/419512@N22",
            )
        name = canonical

    source, created = Source.objects.get_or_create(type=stype, name=name)
    if not created:
        source.enabled = True
        source.save(update_fields=["enabled"])
    return render(request, "ratings/_source_row.html", {"source": source})


@login_required
@require_POST
def source_toggle(request, pk):
    """Toggle the enabled flag on a source without removing its history."""
    source = get_object_or_404(Source, pk=pk)
    source.enabled = not source.enabled
    source.save(update_fields=["enabled"])
    return render(request, "ratings/_source_row.html", {"source": source})


@login_required
@require_POST
def source_delete(request, pk):
    """Permanently remove a source; past scraped images are unaffected."""
    get_object_or_404(Source, pk=pk).delete()
    return with_toast(HttpResponse(""), "Source deleted")


@login_required
@require_POST
def source_import(request):
    """Bulk-import sources from config.toml, creating only records not already in the DB."""
    from ratings.scraper import import_from_config

    n = import_from_config(Path(settings.CONFIG_PATH))
    response = render(request, "ratings/_source_list.html", {"sources": Source.objects.all()})
    noun = "source" if n == 1 else "sources"
    return with_toast(response, f"Imported {n} new {noun} from config.toml", kind="info")


# ── Notification channels and sharing ────────────────────────────────────────


def _channel_error(request, error: str):
    return render(request, "ratings/_channel_error.html", {"error": error})


@login_required
@require_POST
def channel_add(request):
    """Create a new NotificationChannel from the config page form.

    The form targets the inline #channel-add-error slot. On success the refreshed
    list is returned as an out-of-band swap (so #channel-list updates while the
    empty main body clears any prior error); validation failures render into the
    slot without disturbing the existing list. Names are unique, so a duplicate is
    rejected outright rather than silently returning a channel of the wrong service
    (get_or_create would ignore the chosen service for an existing name).
    """
    name = request.POST.get("name", "").strip()
    service = request.POST.get("service", "").strip()
    if not name:
        return _channel_error(request, "Name is required.")
    if service not in dict(NotificationChannel.SERVICE_CHOICES):
        return _channel_error(request, "Invalid service.")
    if NotificationChannel.objects.filter(name=name).exists():
        return _channel_error(request, f"A channel named “{name}” already exists.")
    NotificationChannel.objects.create(name=name, service=service)
    return render(
        request,
        "ratings/_channel_list.html",
        {**_channel_list_ctx(), "oob": True},
    )


@login_required
@require_POST
def channel_save(request, pk: int):
    """Persist credential fields for an existing channel."""
    ch = get_object_or_404(NotificationChannel, pk=pk)
    ch.name = request.POST.get("name", ch.name).strip() or ch.name
    if ch.service == NotificationChannel.MATTERMOST:
        ch.mm_base_url = request.POST.get("mm_base_url", "").strip()
        ch.mm_token = request.POST.get("mm_token", "").strip()
        ch.mm_channel_id = request.POST.get("mm_channel_id", "").strip()
        ch.mm_message_prefix = request.POST.get("mm_message_prefix", "").strip()
    elif ch.service == NotificationChannel.SIGNAL:
        ch.signal_api_url = request.POST.get("signal_api_url", "").strip()
        ch.signal_sender = request.POST.get("signal_sender", "").strip()
        ch.signal_recipients = request.POST.get("signal_recipients", "").strip()
        ch.signal_message_prefix = request.POST.get("signal_message_prefix", "").strip()
    ch.save()
    return render(request, "ratings/_channel_row.html", {"channel": ch})


@login_required
@require_POST
def channel_toggle(request, pk: int):
    """Toggle enabled on a channel without removing its credentials."""
    ch = get_object_or_404(NotificationChannel, pk=pk)
    ch.enabled = not ch.enabled
    ch.save(update_fields=["enabled"])
    return render(request, "ratings/_channel_row.html", {"channel": ch})


@login_required
@require_POST
def channel_delete(request, pk: int):
    """Permanently delete a notification channel."""
    get_object_or_404(NotificationChannel, pk=pk).delete()
    return with_toast(HttpResponse(""), "Channel deleted")


@login_required
@require_POST
def share_image(request, content_hash):
    """
    Share an image to one or more named NotificationChannels.

    The caller POSTs a list of channel PKs (channels[]=1&channels[]=3).
    Sending is synchronous — both APIs are expected to be on the same LAN so
    latency is negligible. The image bytes are uploaded directly because
    /media/ is @login_required and therefore unreachable for message recipients.
    """
    image = get_object_or_404(Image, content_hash=content_hash)
    image_path = DATA_DIR / image.file_path
    selected_pks = request.POST.getlist("channels")
    channels = NotificationChannel.objects.filter(pk__in=selected_pks, enabled=True)
    sent, errors = [], []
    for ch in channels:
        try:
            if ch.service == NotificationChannel.MATTERMOST:
                notifiers.send_to_mattermost(ch, image_path, image.source_label or "")
            elif ch.service == NotificationChannel.SIGNAL:
                notifiers.send_to_signal(ch, image_path, image.source_label or "")
            sent.append(ch.name)
        except Exception as exc:
            errors.append(f"{ch.name}: {exc}")
    # Nothing to swap: the outcome travels as a toast (204 keeps htmx from
    # touching the DOM while still processing the HX-Trigger header).
    parts = []
    if sent:
        parts.append("Shared to " + ", ".join(sent))
    parts.extend(errors)
    if not parts:
        parts.append("No channels selected")
    kind = "ok" if sent and not errors else "error"
    return with_toast(HttpResponse(status=204), " · ".join(parts), kind)
