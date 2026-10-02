from django.db import migrations, models

# The only encoder the app ever had before this migration.
LEGACY_ENCODER_ID = "dinov2_vitb14"


def label_legacy_embeddings(apps, schema_editor):
    """
    Stamp every existing vector as DINOv2 ViT-B/14. That is a plain fact, not a
    guess: no other encoder ever wrote to the column. Rows without a vector keep
    the empty default, nothing is deleted and no file is touched (contract V6).
    The stamp is what lets the new encoder's dedup and kNN code ignore these
    rows instead of mixing two embedding spaces, until they are re-encoded.
    """
    Image = apps.get_model("ratings", "Image")
    Image.objects.filter(embedding__isnull=False).update(
        embedding_model=LEGACY_ENCODER_ID
    )


class Migration(migrations.Migration):
    dependencies = [
        ("ratings", "0020_p3_flatten_image"),
    ]

    operations = [
        migrations.AddField(
            model_name="image",
            name="embedding_model",
            field=models.CharField(
                blank=True, db_index=True, default="", max_length=32
            ),
        ),
        migrations.RunPython(label_legacy_embeddings, migrations.RunPython.noop),
    ]
