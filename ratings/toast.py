"""Toast feedback for htmx responses (UI contract V7)."""

import json

from django.http import HttpResponse

TOAST_HEADER = "HX-Trigger"
TOAST_KINDS = ("ok", "error", "info")


def with_toast(response: HttpResponse, message: str, kind: str = "ok") -> HttpResponse:
    """
    Attach a toast to an htmx response.

    htmx parses the HX-Trigger header as JSON and fires one DOM event per key;
    static/toast.js listens for "toast" and renders the message through
    textContent. The message is therefore data, never markup (OWASP A03):
    json.dumps escapes it at the only place the browser parses it, and its
    default ensure_ascii keeps the header value plain ASCII, which HTTP headers
    require. A view sets at most one toast, so an earlier header is replaced.
    """
    if kind not in TOAST_KINDS:
        raise ValueError(f"Unknown toast kind {kind!r}; expected one of {TOAST_KINDS}")
    response[TOAST_HEADER] = json.dumps({"toast": {"message": message, "kind": kind}})
    return response
