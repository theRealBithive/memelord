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

    INBOX = "inbox"
    CORPUS = "corpus"
    VOID = "void"
    LOCATION_CHOICES = [(INBOX, "inbox"), (CORPUS, "corpus"), (VOID, "void")]

    content_hash = models.CharField(primary_key=True, max_length=64)
    file_path = models.CharField(max_length=2048)  # relative to DATA_DIR
    source_url = models.CharField(max_length=2048, null=True, blank=True)
    source_label = models.CharField(max_length=255)
    location = models.CharField(max_length=32, default=INBOX, choices=LOCATION_CHOICES)
    downloaded_at = models.DateTimeField(auto_now_add=True)
    rated_at = models.DateTimeField(null=True, blank=True)
    is_favourite = models.BooleanField(default=False)
    is_nsfw = models.BooleanField(default=False)
    score = models.IntegerField(null=True, blank=True)
    file_deleted = models.BooleanField(default=False)
    is_purged = models.BooleanField(default=False)
    phash = models.CharField(max_length=16, blank=True, default="", db_index=True)
    embedding = models.BinaryField(null=True, blank=True)
    tags = models.ManyToManyField(Tag, blank=True, related_name="images")
    void_seen_at = models.DateTimeField(null=True, blank=True)
    inbox_seen_at = models.DateTimeField(null=True, blank=True)
    corpus_seen_at = models.DateTimeField(null=True, blank=True)
    queue_seen_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["downloaded_at"]

    def __str__(self) -> str:
        return f"{self.location} {self.content_hash[:8]} ({self.source_label})"


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
    User-configured scrape source (a 4chan board, Imgur topic, Tumblr blog, or Pixelfed instance).

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
    name = models.CharField(max_length=255)  # board / topic / blog / instance URL / account handle
    enabled = models.BooleanField(default=True)
    is_nsfw = models.BooleanField(default=False)
    added_at = models.DateTimeField(auto_now_add=True)
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
