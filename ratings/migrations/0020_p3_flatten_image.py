"""
P3: flatten Image model — wipe all existing images and drop the fields that
belonged to the inbox/corpus/void three-location model (location, is_favourite,
file_deleted, void_seen_at, inbox_seen_at, corpus_seen_at).

All images are deleted first because the flat data/images/ directory replaces
the old data/inbox|corpus|void/ tree; any existing rows would point to
non-existent paths under the new layout.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ratings", "0019_replace_notificationconfig_with_channels"),
    ]

    operations = [
        # Wipe rows so no file_path references the old inbox/corpus/void dirs.
        # Clear M2M first to avoid FK constraint violations during table rebuild.
        migrations.RunSQL("DELETE FROM ratings_image_tags;", migrations.RunSQL.noop),
        migrations.RunSQL("DELETE FROM ratings_image;", migrations.RunSQL.noop),
        migrations.RemoveField(model_name="image", name="location"),
        migrations.RemoveField(model_name="image", name="is_favourite"),
        migrations.RemoveField(model_name="image", name="file_deleted"),
        migrations.RemoveField(model_name="image", name="void_seen_at"),
        migrations.RemoveField(model_name="image", name="inbox_seen_at"),
        migrations.RemoveField(model_name="image", name="corpus_seen_at"),
    ]
