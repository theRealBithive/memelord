from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ratings", "0005_image_phash_embedding"),
    ]

    operations = [
        migrations.AddField(
            model_name="image",
            name="score",
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
