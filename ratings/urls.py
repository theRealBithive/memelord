from django.urls import path
from django.views.generic.base import RedirectView

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("stats/", views.stats, name="stats"),
    # Legacy swipe URLs → current review UI (bookmarks from older releases).
    path(
        "rate/inbox/",
        RedirectView.as_view(pattern_name="review_corpus", permanent=True),
    ),
    path(
        "rate/corpus/",
        RedirectView.as_view(pattern_name="review_corpus", permanent=True),
    ),
    path(
        "rate/void/",
        RedirectView.as_view(pattern_name="below_cutoff", permanent=True),
    ),
    path(
        "rate/nsfw/",
        RedirectView.as_view(pattern_name="rate_nsfw_corpus", permanent=True),
    ),
    path(
        "rate/nsfw/inbox/",
        RedirectView.as_view(pattern_name="rate_nsfw_corpus", permanent=True),
        name="rate_nsfw_inbox",
    ),
    path("rate/nsfw/corpus/", views.rate_nsfw_corpus, name="rate_nsfw_corpus"),
    path(
        "rate/nsfw/corpus/<str:content_hash>/",
        views.rate_nsfw_corpus,
        name="review_nsfw_corpus_image",
    ),
    path(
        "rate/nsfw/corpus/<str:content_hash>/score/",
        views.score_nsfw_corpus,
        name="score_nsfw_corpus",
    ),
    path(
        "rate/nsfw/corpus/<str:content_hash>/purge/",
        views.purge_nsfw_corpus,
        name="purge_nsfw_corpus",
    ),
    path("nsfw-toggle/", views.nsfw_toggle, name="nsfw_toggle"),
    path("gallery/", views.gallery, name="gallery"),
    path("below-cutoff/", views.below_cutoff, name="below_cutoff"),
    path(
        "gallery/<str:content_hash>/action/",
        views.gallery_action,
        name="gallery_action",
    ),
    path("tags/", views.tag_list, name="tag_list"),
    path("tags/autocomplete/", views.tag_autocomplete, name="tag_autocomplete"),
    path("tags/<int:pk>/rename/", views.tag_rename, name="tag_rename"),
    path("tags/<int:pk>/delete/", views.tag_delete, name="tag_delete"),
    path("tags/<str:content_hash>/", views.update_image_tags, name="update_image_tags"),
    # Corpus review
    path("review/", views.review_corpus, name="review_corpus"),
    path("review/<str:content_hash>/", views.review_corpus, name="review_corpus_image"),
    path("review/<str:content_hash>/score/", views.score_corpus, name="score_corpus"),
    path("review/<str:content_hash>/purge/", views.purge_corpus, name="purge_corpus"),
    path("toggle/<str:content_hash>/nsfw/", views.toggle_nsfw, name="toggle_nsfw"),
    path("config/", views.config_view, name="config"),
    path("config/schedule/", views.set_scrape_schedule, name="set_scrape_schedule"),
    path("config/vision/", views.set_vision_thresholds, name="set_vision_thresholds"),
    path("config/channels/add/", views.channel_add, name="channel_add"),
    path("config/channels/<int:pk>/save/", views.channel_save, name="channel_save"),
    path("config/channels/<int:pk>/toggle/", views.channel_toggle, name="channel_toggle"),
    path("config/channels/<int:pk>/delete/", views.channel_delete, name="channel_delete"),
    path("share/<str:content_hash>/", views.share_image, name="share_image"),
    path("config/add/", views.source_add, name="source_add"),
    path("config/import/", views.source_import, name="source_import"),
    path("config/<int:pk>/toggle/", views.source_toggle, name="source_toggle"),
    path("config/<int:pk>/delete/", views.source_delete, name="source_delete"),
    path("scrape/", views.trigger_scrape, name="trigger_scrape"),
    path("train/", views.trigger_train, name="trigger_train"),
    path("train/<str:task_id>/status/", views.train_status, name="train_status"),
    path("logs/", views.logs_page, name="logs"),
    path("logs/entries/", views.log_entries, name="log_entries"),
    path("logs/clear/", views.log_clear, name="log_clear"),
]
