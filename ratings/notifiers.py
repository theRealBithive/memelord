"""HTTP clients for Mattermost and Signal image sharing."""

import base64
import mimetypes
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


def _encode_signal_attachment(image_path: Path) -> str:
    """Build a data-URI attachment for signal-cli-rest-api's base64_attachments field."""
    content_type, _ = mimetypes.guess_type(str(image_path))
    content_type = content_type or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{content_type};filename={image_path.name};base64,{encoded}"


def send_to_signal(cfg, image_path: Path, source_label: str) -> None:
    """
    Send an image file to Signal recipients via signal-cli-rest-api.

    Attachments are uploaded as base64 (same rationale as Mattermost): memelord's
    /media/ endpoint is @login_required, so a bare URL would be unreachable for
    recipients reading the message on their phone.

    signal-cli-rest-api must be running and registered at cfg.signal_api_url.
    Recipients are comma-separated phone numbers stored in cfg.signal_recipients.
    Raises requests.HTTPError on non-2xx.
    """
    recipients = [r.strip() for r in cfg.signal_recipients.split(",") if r.strip()]
    if not recipients:
        return
    prefix = cfg.signal_message_prefix.strip() if cfg.signal_message_prefix else ""
    caption = prefix or source_label or ""
    resp = requests.post(
        f"{cfg.signal_api_url.rstrip('/')}/v2/send",
        json={
            "message": caption,
            "number": cfg.signal_sender,
            "recipients": recipients,
            "base64_attachments": [_encode_signal_attachment(image_path)],
        },
        timeout=60,
    )
    resp.raise_for_status()
