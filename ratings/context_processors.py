"""Template context processors for the ratings app."""

from django.conf import settings


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
