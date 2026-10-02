"""
Source-level guards for the UI overhaul (no database needed).

Contract: V1  Jede Seite teilt Kopf, Navigation und Script-Grundgerüst. Keine
              Seite bringt ein eigenes Stylesheet oder Inline-Styles mit.
Contract: V5  Jede irreversible Aktion (Purge, Source löschen, Channel löschen,
              Tag löschen, Log leeren) verlangt eine Bestätigung im selben Sheet.
              Scores, NSFW-Markierung und Tags brauchen keine.
Contract: V7  Rückmeldungen erscheinen als Toast an einer festen Stelle, auf
              allen Seiten gleich, und verschwinden von selbst.
Contract: V6  Keine Information und keine Funktion hängt allein an Hover:
              Score-Badges in der Gallery sind immer sichtbar, die Tastatur-Hilfe
              ist antippbar.
Contract: V9  Review und Gallery-Lightbox benutzen dieselbe Score-Zeile und
              denselben Tag-Editor. Eine Änderung in der Lightbox ist sofort im
              Grid sichtbar, die Reihenfolge prev/next entspricht dem Grid.
Contract: V10 Die Oberfläche lädt nichts von Drittservern und ist als
              Home-Screen-App installierbar.
Contract: V11 Nach dem Aufräumen gibt es keine ungenutzte CSS-Klasse, keine
              View-Funktion ohne Aufrufer und keinen 1.x-Begriff (liked, fav,
              void) in der Oberfläche.

Why source scans instead of rendered pages: a rendered page only shows the
classes the current data happens to produce, while a scan of templates and
scripts sees every class a page can ever emit. Dynamic class names
(`score-btn--{{ s }}`, `lb-score--${raw}`) are recognised by their prefix up
to the template or JS expression.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = sorted(ROOT.glob("templates/**/*.html"))
SCRIPTS = sorted(ROOT.glob("static/*.js"))
STYLESHEET = ROOT / "static" / "app.css"

# Classes htmx toggles at runtime; they never appear in our own markup.
RUNTIME_CLASSES = {"htmx-request", "htmx-settling", "htmx-swapping"}


def _css_without_comments() -> str:
    return re.sub(r"/\*.*?\*/", "", STYLESHEET.read_text(), flags=re.S)


def _markup_corpus() -> str:
    return "\n".join(p.read_text() for p in TEMPLATES + SCRIPTS)


def _dynamic_prefixes(corpus: str) -> set[str]:
    """Class-name stems that templates or scripts complete at runtime."""
    stems = re.findall(r"([a-zA-Z][\w-]*?-{1,2})(?:\{\{|\{%|\$\{)", corpus)
    return set(stems)


def _is_used(class_name: str, corpus: str, prefixes: set[str]) -> bool:
    if class_name in RUNTIME_CLASSES:
        return True
    if re.search(r"(?<![\w-])" + re.escape(class_name) + r"(?![\w-])", corpus):
        return True
    return any(class_name.startswith(prefix) for prefix in prefixes)


def test_every_css_class_is_used_somewhere() -> None:
    """Contract: V11"""
    defined = set(re.findall(r"\.([a-zA-Z][\w-]*)", _css_without_comments()))
    corpus = _markup_corpus()
    prefixes = _dynamic_prefixes(corpus)
    unused = sorted(c for c in defined if not _is_used(c, corpus, prefixes))
    assert unused == [], f"unused CSS classes: {unused}"


def test_every_view_function_has_a_caller() -> None:
    """Contract: V11"""
    views_source = (ROOT / "ratings" / "views.py").read_text()
    other_sources = "\n".join(
        p.read_text()
        for p in (ROOT / "ratings").glob("*.py")
        if p.name not in ("views.py",)
    )
    tree = ast.parse(views_source)
    dead = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name
        references_inside_views = len(re.findall(r"\b" + name + r"\b", views_source)) - 1
        references_elsewhere = len(re.findall(r"\b" + name + r"\b", other_sources))
        if references_inside_views + references_elsewhere == 0:
            dead.append(name)
    assert dead == [], f"view functions without a caller: {dead}"


def test_pages_have_no_inline_styles_and_extend_the_base() -> None:
    """Contract: V1"""
    offenders = []
    for template in TEMPLATES:
        text = template.read_text()
        is_page = "<!doctype" in text.lower() or "{% extends" in text
        if "<style" in text:
            offenders.append(f"{template.name}: <style> block")
        inline_styles = [m for m in re.findall(r'style="([^"]*)"', text) if not m.startswith("width:")]
        if inline_styles:
            offenders.append(f"{template.name}: style attributes {inline_styles}")
        if is_page and template.name != "base.html" and "{% extends \"base.html\" %}" not in text:
            offenders.append(f"{template.name}: does not extend base.html")
    assert offenders == [], offenders


def test_no_third_party_script_or_style_urls() -> None:
    """Contract: V10"""
    corpus = _markup_corpus()
    remote = re.findall(r'(?:src|href)="(https?://[^"]+)"', corpus)
    assets = [url for url in remote if re.search(r"\.(js|css)(\?|$)", url)]
    assert assets == [], f"remote assets: {assets}"


def test_base_declares_manifest_and_icons() -> None:
    """Contract: V10"""
    base = (ROOT / "templates" / "base.html").read_text()
    assert 'rel="manifest"' in base
    assert 'rel="apple-touch-icon"' in base
    for asset in ("manifest.json", "icons/icon-192.png", "icons/icon-512.png", "icons/apple-touch-icon.png", "vendor/htmx.min.js"):
        assert (ROOT / "static" / asset).exists(), asset


def test_no_legacy_wording_in_templates() -> None:
    """Contract: V11"""
    offenders = []
    for template in TEMPLATES:
        for word in re.findall(r"\b(liked|fav|void)\b", template.read_text(), flags=re.I):
            offenders.append(f"{template.name}: {word}")
    assert offenders == [], offenders


# --- V5 / V7: one confirmation sheet, one toast ---------------------------------

# How a template names the URL of an action: a literal name, or the context
# variable the review card receives (`purge_url` → purge_corpus / purge_nsfw_corpus).
DESTRUCTIVE_URL_REFS = {
    "'purge_corpus'", "'purge_nsfw_corpus'", "purge_url",
    "'source_delete'", "'channel_delete'", "'tag_delete'", "'log_clear'", "'fresh_start'",
    "'lightbox_purge'",
}
HARMLESS_URL_REFS = {
    "score_url", "'toggle_nsfw'", "'update_image_tags'",
    "'channel_toggle'", "'source_toggle'", "'nsfw_toggle'", "'trigger_scrape'", "'trigger_train'",
    "'lightbox_score'", "'lightbox_nsfw'",
}
DESTRUCTIVE_VIEWS = {
    "purge_corpus", "purge_nsfw_corpus", "source_delete",
    "channel_delete", "tag_delete", "log_clear", "fresh_start_view", "lightbox_purge",
}


def _action_tags():
    for path in TEMPLATES:
        for match in re.finditer(r"<(?:button|form|a)\b[^>]*>", path.read_text(), re.S):
            yield path, match.group(0)


def test_every_irreversible_action_asks_the_sheet_and_nothing_else_does() -> None:
    """Contract: V5"""
    for path, tag in _action_tags():
        url_refs = set(re.findall(r"\{%\s*url\s+(\S+)", tag))
        asks = "hx-confirm=" in tag or "data-confirm=" in tag
        if url_refs & DESTRUCTIVE_URL_REFS:
            assert asks, (path.name, tag[:120])
        if url_refs & HARMLESS_URL_REFS:
            assert not asks, (path.name, tag[:120])


def test_base_provides_the_single_sheet_and_toast_slot() -> None:
    """Contract: V5, V7"""
    base = (ROOT / "templates" / "base.html").read_text()
    assert base.count('id="confirm-sheet"') == 1
    assert base.count('id="toast-slot"') == 1
    assert "sheet.js" in base and "toast.js" in base
    corpus = _markup_corpus()
    assert corpus.count('id="toast-slot"') == 1, "no page may bring its own toast slot"


def test_scripts_never_fall_back_to_browser_dialogs() -> None:
    """Contract: V5, V7 (window.confirm/alert would bypass the sheet and the toast)"""
    for script in SCRIPTS:
        assert not re.search(r"(?<![\w.])(confirm|alert|prompt)\(", script.read_text()), script.name


def test_destructive_views_require_login_and_post() -> None:
    """OWASP A01: every state-destroying endpoint is authenticated and never reachable by GET."""
    tree = ast.parse((ROOT / "ratings" / "views.py").read_text())
    seen = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in DESTRUCTIVE_VIEWS:
            names = {d.id if isinstance(d, ast.Name) else getattr(d, "attr", "") for d in node.decorator_list}
            assert {"login_required", "require_POST"} <= names, node.name
            seen.add(node.name)
    assert seen == DESTRUCTIVE_VIEWS


def test_touch_targets_are_at_least_44px_by_token() -> None:
    """Contract: V4 (sizes live in tokens; the tokens are the only thing a source scan can pin)"""
    css = _css_without_comments()
    tokens = dict(re.findall(r"--(tap|pill|score-h|action-h|nav-h):\s*(\d+)px", css))
    assert int(tokens["score-h"]) >= 44
    assert int(tokens["action-h"]) >= 44
    assert int(tokens["tap"]) >= 44
    assert "--reviewnav-h" not in css


# --- V6 / V9: nothing hover-only, one score row and one tag editor ------------


def test_review_card_and_lightbox_share_the_score_row_and_the_tag_editor() -> None:
    """Contract: V9"""
    for name in ("_review_card.html", "_lightbox.html"):
        template = (ROOT / "templates" / "ratings" / name).read_text()
        assert '{% include "ratings/_score_actions.html" %}' in template, name
        assert '{% include "ratings/_tag_editor.html" %}' in template, name
    corpus = _markup_corpus()
    assert corpus.count('class="score-row"') == 1, "the score row is defined once"
    assert corpus.count('class="card-tags"') == 1, "the tag editor is defined once"


def test_hover_never_reveals_or_hides_anything() -> None:
    """Contract: V6 (hover may recolour, never show or hide)"""
    css = _css_without_comments()
    assert "gallery-item-overlay" not in css
    for rule in re.finditer(r"([^{}]+)\{([^}]*)\}", css):
        selector, body = rule.group(1).strip(), rule.group(2)
        if ":hover" not in selector:
            continue
        for forbidden in ("opacity", "display", "visibility"):
            assert forbidden not in body, (selector, forbidden)


def test_keyboard_help_is_a_tappable_disclosure() -> None:
    """Contract: V6"""
    nav = (ROOT / "templates" / "ratings" / "_nav.html").read_text()
    assert '<details class="nav-help">' in nav
    assert "help-tooltip" not in nav


# --- template comments never reach the browser ---------------------------------


def test_no_template_comment_leaks_as_text() -> None:
    """`{# … #}` is a comment only within one line; wrapped, Django renders it as text.

    Lexing every template the way Django does and looking for comment markers
    inside TEXT tokens catches exactly the wrapped ones. Multi-line notes
    belong in `{% comment %} … {% endcomment %}`.
    """
    from django.template.base import Lexer, TokenType

    leaks = []
    for template in sorted((ROOT / "templates").rglob("*.html")):
        for token in Lexer(template.read_text()).tokenize():
            if token.token_type is not TokenType.TEXT:
                continue
            if "{#" in token.contents or "#}" in token.contents:
                leaks.append((template.relative_to(ROOT).as_posix(), token.contents.strip()[:60]))
    assert leaks == []
