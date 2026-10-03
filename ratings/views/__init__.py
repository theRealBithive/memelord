"""
The ratings views, one module per area of the app. urls.py addresses every
view as ``views.<name>``, so this package re-exports them by name; the
helpers stay in their modules (``ratings.views.common`` holds the shared
ones).

- common: taste-model cache and lazy prediction, nav counts, score parsing,
  the index redirect and the show-NSFW switch
- review: the SFW and NSFW review queues and their rate/purge/flip actions
- gallery: gallery, Below cutoff, text search and similar images, lightbox, tags
- jobs: scrape and train (session jobs), the index and re-encode chains, the
  nav indicator, fresh start, the log viewer
- config: the Config page, review-queue settings, schedule, sources, channels,
  sharing
- stats: the stats dashboard
"""

from ratings.views.common import index, nsfw_toggle
from ratings.views.config import (
    channel_add,
    channel_delete,
    channel_save,
    channel_toggle,
    config_view,
    set_scrape_schedule,
    set_vision_thresholds,
    share_image,
    source_add,
    source_delete,
    source_import,
    source_toggle,
)
from ratings.views.gallery import (
    below_cutoff,
    gallery,
    lightbox,
    lightbox_nsfw,
    lightbox_purge,
    lightbox_score,
    tag_autocomplete,
    tag_delete,
    tag_list,
    tag_rename,
    update_image_tags,
)
from ratings.views.jobs import (
    classify_status,
    fresh_start_view,
    job_indicator,
    log_clear,
    log_entries,
    logs_page,
    scrape_status,
    search_index_status,
    taste_reencode_status,
    train_status,
    trigger_classify,
    trigger_scrape,
    trigger_search_index,
    trigger_taste_reencode,
    trigger_train,
)
from ratings.views.review import (
    purge_corpus,
    purge_nsfw_corpus,
    rate_nsfw_corpus,
    review_corpus,
    score_corpus,
    score_nsfw_corpus,
    toggle_nsfw,
)
from ratings.views.stats import stats

__all__ = [
    "below_cutoff",
    "channel_add",
    "channel_delete",
    "channel_save",
    "channel_toggle",
    "classify_status",
    "config_view",
    "fresh_start_view",
    "gallery",
    "index",
    "job_indicator",
    "lightbox",
    "lightbox_nsfw",
    "lightbox_purge",
    "lightbox_score",
    "log_clear",
    "log_entries",
    "logs_page",
    "nsfw_toggle",
    "purge_corpus",
    "purge_nsfw_corpus",
    "rate_nsfw_corpus",
    "review_corpus",
    "score_corpus",
    "score_nsfw_corpus",
    "scrape_status",
    "search_index_status",
    "set_scrape_schedule",
    "set_vision_thresholds",
    "share_image",
    "source_add",
    "source_delete",
    "source_import",
    "source_toggle",
    "stats",
    "tag_autocomplete",
    "tag_delete",
    "tag_list",
    "tag_rename",
    "taste_reencode_status",
    "toggle_nsfw",
    "train_status",
    "trigger_classify",
    "trigger_scrape",
    "trigger_search_index",
    "trigger_taste_reencode",
    "trigger_train",
    "update_image_tags",
]
