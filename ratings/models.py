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
    file_deleted = models.BooleanField(default=False)

    class Meta:
        ordering = ["downloaded_at"]

    def __str__(self) -> str:
        return f"{self.location} {self.content_hash[:8]} ({self.source_label})"
