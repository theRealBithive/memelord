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
    Inject NotificationConfig singleton into every template context.

    A try/except guards against the first request before migrations run — the
    table won't exist yet and we don't want to crash the login page.
    """
    try:
        from ratings.models import NotificationConfig

        cfg, _ = NotificationConfig.objects.get_or_create(pk=1)
    except Exception:
        return {}
    return {
        "notification_cfg": cfg,
        "mm_enabled": cfg.mattermost_enabled and bool(cfg.mattermost_token),
        "signal_enabled": cfg.signal_enabled and bool(cfg.signal_api_url),
    }
