from django.conf import settings
from django.contrib import admin
from django.utils.html import format_html

from .models import Image, Source, Tag


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ("name", "created_at")
    search_fields = ("name",)


@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ("type", "name", "enabled", "is_nsfw", "added_at")
    list_filter = ("type", "enabled", "is_nsfw")
    list_editable = ("enabled", "is_nsfw")
    readonly_fields = ("cursor",)


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ("thumbnail", "source_label", "location", "is_favourite", "rated_at", "downloaded_at")
    list_filter = ("location", "is_favourite", "is_nsfw", "source_label", "file_deleted", "is_purged")
    search_fields = ("content_hash", "file_path", "source_url")
    readonly_fields = ("content_hash", "downloaded_at", "thumbnail_large")
    filter_horizontal = ("tags",)

    @admin.display(description="")
    def thumbnail(self, obj: Image):
        if (settings.DATA_DIR / obj.file_path).exists():
            return format_html(
                '<img src="{}{}" style="height:56px;border-radius:3px;object-fit:cover">',
                settings.MEDIA_URL,
                obj.file_path,
            )
        return "—"

    @admin.display(description="Preview")
    def thumbnail_large(self, obj: Image):
        if (settings.DATA_DIR / obj.file_path).exists():
            return format_html(
                '<img src="{}{}" style="max-height:400px;max-width:100%;border-radius:6px">',
                settings.MEDIA_URL,
                obj.file_path,
            )
        return "File not found on disk."
