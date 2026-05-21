"""Template context processors for the ratings app."""


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
