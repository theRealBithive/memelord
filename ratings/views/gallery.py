"""
The grids and what opens from them: the gallery (plain, text search, similar
images), the Below cutoff grid, the lightbox actions and the tag endpoints.
"""

from urllib.parse import urlencode

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST

from core.brain import EncoderUnavailableError
from ratings import search, similar
from ratings.models import Image, Tag
from ratings.queue_rules import below_cutoff_q, get_review_thresholds
from ratings.toast import with_toast
from ratings.utils import purge_image
from ratings.views.common import (
    apply_score,
    flip_nsfw_and_repredict,
    int_in_range,
    nav_counts,
    score_from_post,
    taste_prediction,
)

_SORT_CHOICES = ("newest", "oldest", "random")
_GRID_LIMIT = 500


def _normalize_sort(raw: str | None) -> str:
    """Whitelist the ?sort= value so an unknown name never reaches order_by (OWASP A03)."""
    if raw in _SORT_CHOICES:
        return raw
    return "newest"


def _apply_sort(qs, sort: str):
    if sort == "random":
        return qs.order_by("?")
    if sort == "oldest":
        return qs.order_by("downloaded_at")
    return qs.order_by("-downloaded_at")


def _grid_page(qs) -> list[Image]:
    """
    The plain grids render every card into the DOM at once, so they stop at
    _GRID_LIMIT rows; the ranked modes have their own, smaller cap.
    """
    return list(qs.prefetch_related("tags")[:_GRID_LIMIT])


@login_required
def below_cutoff(request):
    """
    Safety-net grid for images the model rejected or the user scored ≤ 2.

    Unrated rows with predicted_score below the /config/ dial land here instead
    of vanishing from review. User-rated trash (0) and scores 1–2 stay here too.
    Re-score upward or purge from the gallery lightbox.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    sort = _normalize_sort(request.GET.get("sort"))
    active_tag = request.GET.get("tag", "").strip().lower()

    sfw_bucket, nsfw_bucket = get_review_thresholds()
    qs = Image.objects.filter(is_purged=False).filter(
        below_cutoff_q(sfw_bucket, nsfw_bucket, show_nsfw)
    )
    if not show_nsfw:
        qs = qs.filter(is_nsfw=False)
    if active_tag:
        qs = qs.filter(tags__name=active_tag)

    images = _grid_page(_apply_sort(qs, sort))
    all_tags = list(Tag.objects.values_list("name", flat=True))

    return render(
        request,
        "ratings/gallery.html",
        {
            **nav_counts(show_nsfw),
            "images": images,
            "total": len(images),
            "min_score": 1,
            "sort": sort,
            "scores": range(1, 7),
            "show_nsfw": show_nsfw,
            "mode": "below_cutoff",
            "is_below_cutoff": True,
            "all_tags": all_tags,
            "active_tag": active_tag,
        },
    )


def _images_in_rank_order(ranked: list[tuple[str, float]]) -> list[Image]:
    """
    Fetch the ranked hashes in one query and put them back into rank order.

    in_bulk is one query for the whole page; a hash purged between ranking
    and fetch is simply absent from the dict and drops out of the page.
    """
    by_hash = Image.objects.prefetch_related("tags").in_bulk([h for h, _ in ranked])
    ordered: list[Image] = []
    for content_hash, _similarity in ranked:
        image = by_hash.get(content_hash)
        if image is not None:
            ordered.append(image)
    return ordered


def _gallery_links(min_score: int, sort: str, active_tag: str, scope: str, mode_params: dict) -> dict:
    """
    Query strings for the gallery template, built here so it never assembles
    a URL from the search text by hand: `sticky` is what every score and tag
    link appends to stay inside the current search or similarity view,
    `scope_urls` switch the scope inside that view, `clear_url` leaves it.
    urlencode quotes the query text, so a `&` or `<` typed into the search box
    cannot split a link or inject markup (OWASP A03).
    """
    base = {"min_score": min_score, "sort": sort}
    if active_tag:
        base["tag"] = active_tag
    if not mode_params:
        return {"sticky": "", "scope_urls": {}, "clear_url": "?" + urlencode(base)}
    scope_urls = {}
    for name in search.SCOPES:
        scope_urls[name] = "?" + urlencode({**base, "scope": name, **mode_params})
    return {
        "sticky": urlencode({"scope": scope, **mode_params}),
        "scope_urls": scope_urls,
        "clear_url": "?" + urlencode(base),
    }


@login_required
def gallery(request):
    """
    Scored-image gallery with filter, sort and tag controls, plus two ranked
    modes on the same grid: a text search (`?q=`, SigLIP2, contract V5) and
    "similar images" (`?similar=<hash>`, DINOv3, V17).

    A ranked mode replaces the sort and caps the page at RESULT_LIMIT; the
    plain gallery keeps its 500-item cap because it renders everything into
    the DOM at once. `scope` widens the candidate set to the unrated backlog
    and only means something inside a ranked mode: the plain gallery is rated
    images by definition. Both ranked modes fall back to the plain gallery
    with a toast when they cannot run (missing weights, an anchor without a
    current vector), never to a silent empty grid (V10, V17). The query text
    reaches only the tokenizer and the template's autoescape (V8).
    """
    show_nsfw = request.session.get("show_nsfw", False)
    min_score = int_in_range(request.GET.get("min_score"), 1, 6, 1)
    sort = _normalize_sort(request.GET.get("sort"))
    active_tag = request.GET.get("tag", "").strip().lower()
    scope = search.normalize_scope(request.GET.get("scope"))
    query = search.normalize_query(request.GET.get("q"))
    similar_hash = request.GET.get("similar", "").strip()

    anchor = None
    if similar_hash:
        anchor = get_object_or_404(Image, content_hash=similar_hash, is_purged=False)

    candidates = search.candidate_images(scope, min_score, active_tag, show_nsfw)
    ranked: list[tuple[str, float]] = []
    mode_params: dict[str, str] = {}
    unindexed = 0
    if anchor is not None:
        similar_ranked = similar.rank_similar(anchor, candidates, search.RESULT_LIMIT)
        if similar_ranked is None:
            messages.info(request, "This image has no current embedding yet. Run Train first.")
            anchor = None
        else:
            ranked = similar_ranked
            mode_params = {"similar": anchor.content_hash}
    elif query:
        try:
            ranked = search.rank_by_text(query, candidates, search.RESULT_LIMIT)
            unindexed = search.unindexed_count(candidates)
            mode_params = {"q": query}
        except EncoderUnavailableError as exc:
            messages.error(request, str(exc))
            query = ""

    if mode_params:
        images = _images_in_rank_order(ranked)
    else:
        scope = search.DEFAULT_SCOPE
        qs = search.candidate_images(scope, min_score, active_tag, show_nsfw)
        images = _grid_page(_apply_sort(qs, sort))

    all_tags = list(Tag.objects.values_list("name", flat=True))

    return render(
        request,
        "ratings/gallery.html",
        {
            **nav_counts(show_nsfw),
            "images": images,
            "total": len(images),
            "min_score": min_score,
            "sort": sort,
            "scores": range(1, 7),
            "show_nsfw": show_nsfw,
            "mode": "gallery",
            "all_tags": all_tags,
            "active_tag": active_tag,
            "q": query if "q" in mode_params else "",
            "scope": scope,
            "is_search": "q" in mode_params,
            "is_similar": "similar" in mode_params,
            "similar_to": anchor,
            "unindexed": unindexed,
            **_gallery_links(min_score, sort, active_tag, scope, mode_params),
        },
    )


# ── Gallery lightbox (UI contract V9) ────────────────────────────────────────


def _lightbox_ctx(image: Image) -> dict:
    """
    Context for the lightbox panel. It names the same shared partials as the
    review card (score row, action row, tag editor) with its own endpoints,
    and `lightbox=True` makes prev/next client-side so the order is the
    grid's, which only the browser knows (random sort, tag filter).
    """
    return {
        "image": image,
        "prediction": taste_prediction(image),
        "scores": range(1, 7),
        "lightbox": True,
        "score_url": "lightbox_score",
        "purge_url": "lightbox_purge",
        "nsfw_url": "lightbox_nsfw",
        "hx_target": "#lightbox-content",
        "hx_swap": "innerHTML",
    }


def _lightbox_update(request, image: Image | None, deleted_hash: str | None = None):
    """
    Answer a lightbox action with the panel plus, out of band, the grid card
    and the nav badges, so one response keeps every view of the image in step.
    A purged image has no panel; its card is deleted from the grid instead.
    """
    show_nsfw = request.session.get("show_nsfw", False)
    ctx = {**nav_counts(show_nsfw), "show_nsfw": show_nsfw, "deleted_hash": deleted_hash}
    if image is not None:
        ctx.update(_lightbox_ctx(image))
    return render(request, "ratings/_lightbox_update.html", ctx)


@login_required
def lightbox(request, content_hash: str):
    image = get_object_or_404(
        Image.objects.prefetch_related("tags"), content_hash=content_hash, is_purged=False
    )
    return render(request, "ratings/_lightbox.html", _lightbox_ctx(image))


@login_required
@require_POST
def lightbox_score(request, content_hash: str):
    """Re-score from the gallery or Below cutoff; the image stays in view (no rate-and-advance there)."""
    image = get_object_or_404(Image, content_hash=content_hash, is_purged=False)
    score_val = score_from_post(request)
    if score_val is not None:
        apply_score(image, score_val)
    return _lightbox_update(request, image)


@login_required
@require_POST
def lightbox_nsfw(request, content_hash: str):
    """Flip the NSFW mark from the gallery lightbox; the prediction follows the new category (V26)."""
    image = get_object_or_404(Image, content_hash=content_hash, is_purged=False)
    flip_nsfw_and_repredict(image)
    return _lightbox_update(request, image)


@login_required
@require_POST
def lightbox_purge(request, content_hash: str):
    """Hard-delete from the lightbox; the grid card leaves with the response and the toast reports it."""
    image = get_object_or_404(Image, content_hash=content_hash, is_purged=False)
    purge_image(image)
    response = _lightbox_update(request, None, deleted_hash=content_hash)
    return with_toast(response, "Image purged")


# ── Tags ──────────────────────────────────────────────────────────────────────


@login_required
def tag_autocomplete(request):
    """JSON endpoint for tag name suggestions; filtered by prefix when ?q= is given."""
    q = request.GET.get("q", "").strip().lower()
    qs = Tag.objects.all()
    if q:
        qs = qs.filter(name__startswith=q)
    return JsonResponse({"tags": list(qs.values_list("name", flat=True)[:20])})


@login_required
@require_POST
def update_image_tags(request, content_hash: str):
    """
    Replace all tags on an image with the submitted comma-separated list.

    M2M .set() does a diff internally (removes old, adds new) rather than
    clearing and re-inserting, so this is safe to call repeatedly without
    accumulating duplicates or racing against other requests.
    """
    image = get_object_or_404(Image, content_hash=content_hash)
    tag_str = request.POST.get("tags", "")
    tag_names = [t.strip().lower() for t in tag_str.split(",") if t.strip()]
    tags = [Tag.objects.get_or_create(name=name)[0] for name in tag_names]
    image.tags.set(tags)
    return JsonResponse({"tags": sorted(image.tags.values_list("name", flat=True))})


@login_required
def tag_list(request):
    """List all tags with image counts for management (rename / delete)."""
    tags = list(Tag.objects.annotate(n=Count("images")).order_by("-n", "name"))
    return render(request, "ratings/tags.html", {"tags": tags, "mode": "tags", **nav_counts()})


def _tag_row(request, tag: Tag):
    """Render one tag row with its image count, the answer to every tag_rename outcome."""
    tag.n = tag.images.count()
    return render(request, "ratings/_tag_row.html", {"tag": tag})


@login_required
def tag_rename(request, pk: int):
    """
    Rename a tag in-place.

    GET ?edit=1 swaps the row for an inline edit form.
    POST applies the rename; if the new name already exists the two tags are
    merged — all images from the old tag move to the existing one and the old
    record is deleted — so the user can consolidate typos without manual cleanup.
    """
    tag = get_object_or_404(Tag, pk=pk)
    if request.method == "GET":
        if not request.GET.get("edit"):
            return _tag_row(request, tag)
        tag.n = tag.images.count()
        return render(request, "ratings/_tag_row_edit.html", {"tag": tag})
    new_name = request.POST.get("name", "").strip().lower()
    if not new_name or new_name == tag.name:
        return _tag_row(request, tag)
    existing = Tag.objects.filter(name=new_name).exclude(pk=pk).first()
    if existing:
        # Merge: move all images from the old tag to the existing one, then
        # delete the old tag so there's no duplicate entry in the tag list.
        for image in tag.images.all():
            image.tags.add(existing)
        tag.delete()
        return _tag_row(request, existing)
    tag.name = new_name
    tag.save(update_fields=["name"])
    return _tag_row(request, tag)


@login_required
@require_POST
def tag_delete(request, pk: int):
    """Delete a tag and remove it from all images that carry it."""
    get_object_or_404(Tag, pk=pk).delete()
    return with_toast(HttpResponse(""), "Tag deleted")
