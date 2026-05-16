from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("stats/", views.stats, name="stats"),
    path("rate/inbox/", views.rate_inbox, name="rate_inbox"),
    path("rate/corpus/", views.rate_corpus, name="rate_corpus"),
    path("rate/void/", views.rate_void, name="rate_void"),
    path("rate/<str:content_hash>/<str:action>/", views.submit_rating, name="submit_rating"),
    path("config/", views.config_view, name="config"),
    path("config/add/", views.source_add, name="source_add"),
    path("config/import/", views.source_import, name="source_import"),
    path("config/<int:pk>/toggle/", views.source_toggle, name="source_toggle"),
    path("config/<int:pk>/delete/", views.source_delete, name="source_delete"),
    path("scrape/", views.trigger_scrape, name="trigger_scrape"),
    path("train/", views.trigger_train, name="trigger_train"),
]
