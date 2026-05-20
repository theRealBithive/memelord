from django.db import migrations, models


def backfill_queue_seen_at(apps, schema_editor):
    Image = apps.get_model("ratings", "Image")
    Image.objects.filter(corpus_seen_at__isnull=False).update(
        queue_seen_at=models.F("corpus_seen_at")
    )
    Image.objects.filter(
        queue_seen_at__isnull=True, inbox_seen_at__isnull=False
    ).update(queue_seen_at=models.F("inbox_seen_at"))


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("ratings", "0010_inbox_seen_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="image",
            name="queue_seen_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.RunPython(backfill_queue_seen_at, noop),
    ]
