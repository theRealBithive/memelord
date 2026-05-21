"""HTTP clients for Mattermost and Signal image sharing."""

from pathlib import Path

import requests


def send_to_mattermost(cfg, image_path: Path, source_label: str) -> None:
    """
    Upload an image file to a Mattermost channel as an attachment.

    Uploads the bytes via /api/v4/files then creates a post referencing the
    returned file_id. We attach the file directly instead of posting a URL
    because memelord's /media/ endpoint is @login_required — a bare URL would
    be unreachable for anyone reading the message in Mattermost.

    Uses a Personal Access Token so the post appears from the user's own
    account. Raises requests.HTTPError on non-2xx.
    """
    base = cfg.mattermost_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {cfg.mattermost_token}"}
    with open(image_path, "rb") as fh:
        upload = requests.post(
            f"{base}/api/v4/files",
            headers=headers,
            data={"channel_id": cfg.mattermost_channel_id},
            files={"files": (image_path.name, fh)},
            timeout=30,
        )
    upload.raise_for_status()
    file_id = upload.json()["file_infos"][0]["id"]

    prefix = cfg.mattermost_message_prefix
    message = prefix.strip() if prefix else ""
    post = requests.post(
        f"{base}/api/v4/posts",
        json={
            "channel_id": cfg.mattermost_channel_id,
            "message": message,
            "file_ids": [file_id],
        },
        headers=headers,
        timeout=10,
    )
    post.raise_for_status()


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
