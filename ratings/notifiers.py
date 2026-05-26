"""HTTP clients for Mattermost and Signal image sharing."""

import base64
import mimetypes
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import requests
from PIL import Image as PilImage


@contextmanager
def _mattermost_upload_file(image_path: Path) -> Iterator[tuple[Path, str]]:
    """
    Yield (path, filename) for Mattermost's multipart upload.

    Mattermost's mobile clients cannot preview WebP attachments, so WebP
    sources are transcoded to JPEG in a temporary file that is removed on exit.
    All other formats are uploaded unchanged.
    """
    if image_path.suffix.lower() != ".webp":
        yield image_path, image_path.name
        return

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        with PilImage.open(image_path) as img:
            img.convert("RGB").save(out, "JPEG", quality=92, optimize=True)
        yield out, f"{image_path.stem}.jpg"
    finally:
        out.unlink(missing_ok=True)


def send_to_mattermost(ch, image_path: Path, source_label: str) -> None:
    """
    Upload an image file to a Mattermost channel as an attachment.

    Uploads the bytes via /api/v4/files then creates a post referencing the
    returned file_id. We attach the file directly instead of posting a URL
    because memelord's /media/ endpoint is @login_required — a bare URL would
    be unreachable for anyone reading the message in Mattermost.

    Uses a Personal Access Token so the post appears from the user's own
    account. Raises requests.HTTPError on non-2xx.
    """
    base = ch.mm_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {ch.mm_token}"}
    with _mattermost_upload_file(image_path) as (upload_path, upload_name):
        with open(upload_path, "rb") as fh:
            upload = requests.post(
                f"{base}/api/v4/files",
                headers=headers,
                data={"channel_id": ch.mm_channel_id},
                files={"files": (upload_name, fh)},
                timeout=30,
            )
    upload.raise_for_status()
    file_id = upload.json()["file_infos"][0]["id"]

    message = ch.mm_message_prefix.strip() if ch.mm_message_prefix else ""
    post = requests.post(
        f"{base}/api/v4/posts",
        json={
            "channel_id": ch.mm_channel_id,
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


def send_to_signal(ch, image_path: Path, source_label: str) -> None:
    """
    Send an image file to Signal recipients via signal-cli-rest-api.

    Attachments are uploaded as base64 (same rationale as Mattermost): memelord's
    /media/ endpoint is @login_required, so a bare URL would be unreachable for
    recipients reading the message on their phone.

    signal-cli-rest-api must be running and registered at ch.signal_api_url.
    Recipients are comma-separated phone numbers stored in ch.signal_recipients.
    Raises requests.HTTPError on non-2xx.
    """
    recipients = [r.strip() for r in ch.signal_recipients.split(",") if r.strip()]
    if not recipients:
        return
    prefix = ch.signal_message_prefix.strip() if ch.signal_message_prefix else ""
    caption = prefix or source_label or ""
    resp = requests.post(
        f"{ch.signal_api_url.rstrip('/')}/v2/send",
        json={
            "message": caption,
            "number": ch.signal_sender,
            "recipients": recipients,
            "base64_attachments": [_encode_signal_attachment(image_path)],
        },
        timeout=60,
    )
    resp.raise_for_status()
