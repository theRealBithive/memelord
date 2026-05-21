"""HTTP clients for Mattermost and Signal image sharing."""

import requests


def send_to_mattermost(cfg, image_url: str, source_label: str) -> None:
    """
    Post an image URL to a Mattermost channel via the REST API.

    Uses a Personal Access Token so the message appears from the user's own
    account. The token is never sent to the image server — only to Mattermost.
    Raises requests.HTTPError on non-2xx.
    """
    prefix = cfg.mattermost_message_prefix
    message = f"{prefix} {image_url}".strip() if prefix else image_url
    resp = requests.post(
        f"{cfg.mattermost_base_url.rstrip('/')}/api/v4/posts",
        json={"channel_id": cfg.mattermost_channel_id, "message": message},
        headers={"Authorization": f"Bearer {cfg.mattermost_token}"},
        timeout=10,
    )
    resp.raise_for_status()


def send_to_signal(cfg, image_url: str, source_label: str) -> None:
    """
    Send an image URL to Signal recipients via signal-cli-rest-api.

    signal-cli-rest-api must be running and registered at cfg.signal_api_url.
    Recipients are comma-separated phone numbers stored in cfg.signal_recipients.
    Raises requests.HTTPError on non-2xx.
    """
    recipients = [r.strip() for r in cfg.signal_recipients.split(",") if r.strip()]
    if not recipients:
        return
    prefix = cfg.signal_message_prefix
    message = f"{prefix} {image_url}".strip() if prefix else image_url
    resp = requests.post(
        f"{cfg.signal_api_url.rstrip('/')}/v2/send",
        json={"message": message, "number": cfg.signal_sender, "recipients": recipients},
        timeout=10,
    )
    resp.raise_for_status()
