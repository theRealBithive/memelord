from django.db import models


class Tag(models.Model):
    """Free-form label that can be attached to any Image for gallery filtering."""

    name = models.CharField(max_length=100, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Image(models.Model):
    """
    Central record for a scraped image.

    content_hash (SHA-256) is the primary key so the scraper can skip
    re-downloading in O(1) without a filename-based lookup — filenames are
    unreliable across sources and can collide. file_path is stored relative
    to DATA_DIR so the entire data volume can be moved without a migration.
    """

    content_hash = models.CharField(primary_key=True, max_length=64)
    file_path = models.CharField(max_length=2048)  # relative to DATA_DIR
    source_url = models.CharField(max_length=2048, null=True, blank=True)
    source_label = models.CharField(max_length=255)
    downloaded_at = models.DateTimeField(auto_now_add=True)
    rated_at = models.DateTimeField(null=True, blank=True)
    is_nsfw = models.BooleanField(default=False)
    score = models.IntegerField(null=True, blank=True)
    is_purged = models.BooleanField(default=False)
    phash = models.CharField(max_length=16, blank=True, default="", db_index=True)
    embedding = models.BinaryField(null=True, blank=True)
    predicted_score = models.FloatField(null=True, blank=True, db_index=True)
    tags = models.ManyToManyField(Tag, blank=True, related_name="images")
    queue_seen_at = models.DateTimeField(null=True, blank=True)
    knn_tag_suggestions = models.CharField(max_length=500, blank=True, default="")

    class Meta:
        ordering = ["downloaded_at"]

    def __str__(self) -> str:
        return f"{self.content_hash[:8]} ({self.source_label})"

    @property
    def knn_tag_suggestions_list(self) -> list[str]:
        """
        Parsed knn_tag_suggestions field as a list, excluding already-applied tags.

        Suggestions are the user's own tags inherited from visually similar
        already-tagged images via DINOv2 kNN — they encode taste/theme, not
        objects, so they line up with how the user actually labels things.
        """
        applied = set(self.tags.values_list("name", flat=True))
        return [
            k.strip()
            for k in self.knn_tag_suggestions.split(",")
            if k.strip() and k.strip() not in applied
        ]


class LogEntry(models.Model):
    """Single log line from a scrape or train run, persisted for the in-app log viewer."""

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    level = models.CharField(max_length=10)
    source = models.CharField(max_length=20)  # "scrape" or "train"
    message = models.TextField()

    class Meta:
        ordering = ["pk"]


class Source(models.Model):
    """
    User-configured scrape source (a 4chan board, Imgur topic, Tumblr blog, or
    Mastodon/Pixelfed account handle).

    Sources in the DB take precedence over config.toml entries — the DB is
    the live config that the UI edits, while config.toml serves as the seed
    file imported on first setup via source_import.
    """

    FOURCHAN = "4chan"
    IMGUR = "imgur"
    TUMBLR = "tumblr"
    PIXELFED = "pixelfed"
    MASTODON = "mastodon"
    TYPE_CHOICES = [
        (FOURCHAN, "4chan"),
        (IMGUR, "Imgur"),
        (TUMBLR, "Tumblr"),
        (PIXELFED, "Pixelfed"),
        (MASTODON, "Mastodon"),
    ]

    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    name = models.CharField(max_length=255)  # board / topic / blog / account handle
    enabled = models.BooleanField(default=True)
    is_nsfw = models.BooleanField(default=False)
    added_at = models.DateTimeField(auto_now_add=True)
    # Highest status ID seen on the last scrape — lets Mastodon/Pixelfed account
    # scrapes resume forward instead of re-walking the whole timeline each run.
    cursor = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        unique_together = [("type", "name")]
        ordering = ["type", "name"]

    def __str__(self) -> str:
        return f"{self.type}/{self.name}"


class ScrapeSchedule(models.Model):
    """Singleton (pk=1) storing the user-configured auto-scrape interval."""

    interval_hours = models.PositiveSmallIntegerField(default=6)
    enabled = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        """Force pk=1 to maintain the singleton invariant."""
        self.pk = 1
        super().save(*args, **kwargs)


class ReviewThresholds(models.Model):
    """
    Singleton (pk=1) for the 1-6 hide thresholds on the SFW/NSFW review queues.

    Stored in the DB rather than read live from config.toml so the user can
    adjust the dial in the web UI without filesystem access. config.toml's
    [vision] section seeds this row on first access via get_or_create — that
    keeps backwards compatibility with deployments that set the threshold
    declaratively before this model existed.
    """

    sfw_threshold = models.PositiveSmallIntegerField(default=1)
    nsfw_threshold = models.PositiveSmallIntegerField(default=1)

    def save(self, *args, **kwargs):
        """Force pk=1 to maintain the singleton invariant."""
        self.pk = 1
        super().save(*args, **kwargs)


class NotificationChannel(models.Model):
    """
    A named sharing destination — one Mattermost channel or one set of Signal recipients.

    Named channels ("Aurea", "TownSquare", "Simon") replace the old singleton
    NotificationConfig so the user can send an image to specific people instead
    of always broadcasting to everyone at once. The service-specific fields are
    present on every row but only the fields relevant to the chosen service are used;
    the others stay blank.
    """

    MATTERMOST = "mattermost"
    SIGNAL = "signal"
    SERVICE_CHOICES = [
        (MATTERMOST, "Mattermost"),
        (SIGNAL, "Signal"),
    ]

    name = models.CharField(max_length=100, unique=True)
    service = models.CharField(max_length=20, choices=SERVICE_CHOICES)
    enabled = models.BooleanField(default=True)

    # Mattermost fields
    mm_base_url = models.CharField(max_length=255, blank=True)
    mm_token = models.CharField(max_length=255, blank=True)
    mm_channel_id = models.CharField(max_length=64, blank=True)
    mm_message_prefix = models.CharField(max_length=255, blank=True)

    # Signal fields
    signal_api_url = models.CharField(max_length=255, blank=True)
    signal_sender = models.CharField(max_length=32, blank=True)
    signal_recipients = models.TextField(blank=True)  # comma-separated phone numbers
    signal_message_prefix = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "notification channel"

    def __str__(self) -> str:
        return f"{self.name} ({self.service})"

    @property
    def is_configured(self) -> bool:
        """Return True when the channel has enough credentials to actually send."""
        if self.service == self.MATTERMOST:
            return bool(self.mm_token and self.mm_base_url and self.mm_channel_id)
        if self.service == self.SIGNAL:
            return bool(self.signal_api_url and self.signal_sender and self.signal_recipients)
        return False
