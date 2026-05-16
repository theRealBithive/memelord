from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("stats/", views.stats, name="stats"),
    path("rate/inbox/", views.rate_inbox, name="rate_inbox"),
    path("rate/corpus/", views.rate_corpus, name="rate_corpus"),
    path("rate/void/", views.rate_void, name="rate_void"),
    path("rate/nsfw/", views.rate_nsfw_inbox, name="rate_nsfw_inbox"),
    path("rate/nsfw/inbox/", views.rate_nsfw_inbox, name="rate_nsfw_inbox"),
    path("rate/nsfw/corpus/", views.rate_nsfw_corpus, name="rate_nsfw_corpus"),
    path("rate/nsfw/void/", views.rate_nsfw_void, name="rate_nsfw_void"),
    path("rate/fav/", views.rate_fav, name="rate_fav"),
    path("rate/nsfw/fav/", views.rate_nsfw_fav, name="rate_nsfw_fav"),
    path("rate/<str:content_hash>/<str:action>/", views.submit_rating, name="submit_rating"),
    path("nsfw-toggle/", views.nsfw_toggle, name="nsfw_toggle"),
    path("config/", views.config_view, name="config"),
    path("config/add/", views.source_add, name="source_add"),
    path("config/import/", views.source_import, name="source_import"),
    path("config/<int:pk>/toggle/", views.source_toggle, name="source_toggle"),
    path("config/<int:pk>/nsfw/", views.source_nsfw_toggle, name="source_nsfw_toggle"),
    path("config/<int:pk>/delete/", views.source_delete, name="source_delete"),
    path("scrape/", views.trigger_scrape, name="trigger_scrape"),
    path("train/", views.trigger_train, name="trigger_train"),
    path("train/<str:task_id>/status/", views.train_status, name="train_status"),
]
