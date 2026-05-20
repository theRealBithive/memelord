from django.db import models


class Image(models.Model):
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
    void_seen_at = models.DateTimeField(null=True, blank=True)
    inbox_seen_at = models.DateTimeField(null=True, blank=True)
    corpus_seen_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["downloaded_at"]

    def __str__(self) -> str:
        return f"{self.location} {self.content_hash[:8]} ({self.source_label})"


class LogEntry(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    level = models.CharField(max_length=10)
    source = models.CharField(max_length=20)  # "scrape" or "train"
    message = models.TextField()

    class Meta:
        ordering = ["pk"]


class Source(models.Model):
    FOURCHAN = "4chan"
    IMGUR = "imgur"
    TUMBLR = "tumblr"
    PIXELFED = "pixelfed"
    TYPE_CHOICES = [
        (FOURCHAN, "4chan"),
        (IMGUR, "Imgur"),
        (TUMBLR, "Tumblr"),
        (PIXELFED, "Pixelfed"),
    ]

    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    name = models.CharField(max_length=255)  # board / topic / blog / instance URL
    enabled = models.BooleanField(default=True)
    is_nsfw = models.BooleanField(default=False)
    added_at = models.DateTimeField(auto_now_add=True)

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
        self.pk = 1
        super().save(*args, **kwargs)
