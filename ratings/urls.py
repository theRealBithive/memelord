from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("stats/", views.stats, name="stats"),
    path("rate/inbox/", views.rate_inbox, name="rate_inbox"),
    path("rate/corpus/", views.rate_corpus, name="rate_corpus"),
    path("rate/void/", views.rate_void, name="rate_void"),
    path("rate/<str:content_hash>/<str:action>/", views.submit_rating, name="submit_rating"),
    path("scrape/", views.trigger_scrape, name="trigger_scrape"),
    path("train/", views.trigger_train, name="trigger_train"),
]
