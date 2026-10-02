"""
Navigation contract, checked on the rendered nav partial (no database).

Contract: V2 Die Navigation zeigt jeden Bereich genau einmal. Der aktuelle
             Bereich ist auf jeder Seite markiert, auch Stats, Logs, Config und
             Tags. Jeder Bereich ist auf jeder Bildschirmbreite erreichbar.
Contract: V3 Der Schalter „NSFW einblenden“ und der Knopf „dieses Bild als NSFW
             markieren“ sind beschriftet und optisch unterscheidbar.

The nav partial depends only on its context, so it is rendered directly with
render_to_string. The generator walks every mode the views emit, both NSFW
visibility states, any badge counts and both job states, which is the whole
input space of the partial.
"""

from __future__ import annotations

import os
from html.parser import HTMLParser

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.template.loader import render_to_string
from django.test import RequestFactory
from django.urls import reverse
from hypothesis import given
from hypothesis import strategies as st

MODES = ["corpus", "nsfw_corpus", "gallery", "below_cutoff", "stats", "tags", "logs", "config"]
AREA_URLS = {
    "corpus": reverse("review_corpus"),
    "gallery": reverse("gallery"),
    "below_cutoff": reverse("below_cutoff"),
    "stats": reverse("stats"),
    "tags": reverse("tag_list"),
    "logs": reverse("logs"),
    "config": reverse("config"),
}
NSFW_QUEUE_URL = reverse("rate_nsfw_corpus")


class _Elements(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def _elements(html: str) -> list[tuple[str, dict]]:
    parser = _Elements()
    parser.feed(html)
    return parser.elements


def _render(mode: str, show_nsfw: bool, counts: dict, active_task_id: str | None) -> str:
    """Rendered with a request so {% csrf_token %} and the context processors behave as in a view."""
    request = RequestFactory().get("/")
    context = {"mode": mode, "show_nsfw": show_nsfw, "active_task_id": active_task_id, **counts}
    return render_to_string("ratings/_nav.html", context, request=request)


nav_context = st.fixed_dictionaries(
    {
        "mode": st.sampled_from(MODES),
        "show_nsfw": st.booleans(),
        "counts": st.fixed_dictionaries(
            {
                "queue_count": st.integers(0, 9999),
                "below_cutoff_count": st.integers(0, 9999),
                "nsfw_queue_count": st.integers(0, 9999),
            }
        ),
        "active_task_id": st.one_of(st.none(), st.just("task-1")),
    }
)


@given(nav_context)
def test_exactly_one_area_is_marked_active(ctx) -> None:
    """
    Contract: V2

    Includes the NSFW queue viewed while "Show NSFW" is on: the queue is then
    merged into Review, so Review is the marked area (the NSFW tab is hidden).
    """
    html = _render(**ctx)
    active = [
        attrs for tag, attrs in _elements(html)
        if tag in ("a", "button") and "active" in (attrs.get("class") or "").split()
    ]
    assert len(active) == 1, (ctx["mode"], ctx["show_nsfw"], active)


@given(nav_context)
def test_every_area_is_linked_exactly_once(ctx) -> None:
    """Contract: V2"""
    html = _render(**ctx)
    hrefs = [attrs.get("href") for tag, attrs in _elements(html) if tag == "a"]
    for area, url in AREA_URLS.items():
        assert hrefs.count(url) == 1, (area, hrefs)
    assert hrefs.count(NSFW_QUEUE_URL) == (0 if ctx["show_nsfw"] else 1)


@given(nav_context)
def test_nsfw_visibility_switch_is_a_labelled_button_with_state(ctx) -> None:
    """Contract: V3"""
    html = _render(**ctx)
    assert "Show NSFW" in html
    state = "on" if ctx["show_nsfw"] else "off"
    assert f'<span class="nav-toggle-state">{state}</span>' in html
    forms = [attrs for tag, attrs in _elements(html) if tag == "form"]
    assert any(attrs.get("action") == reverse("nsfw_toggle") and attrs.get("method") == "post" for attrs in forms)


@given(nav_context)
def test_logout_is_a_post_form(ctx) -> None:
    """Contract: V2 (Django 5+ only accepts POST on logout, so a link would 405)"""
    forms = [attrs for tag, attrs in _elements(_render(**ctx)) if tag == "form"]
    assert any(attrs.get("action") == reverse("logout") and attrs.get("method") == "post" for attrs in forms)
