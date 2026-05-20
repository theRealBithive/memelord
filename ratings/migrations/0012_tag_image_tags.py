from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ratings", "0011_queue_seen_at"),
    ]

    operations = [
        migrations.CreateModel(
            name="Tag",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(db_index=True, max_length=100, unique=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["name"],
            },
        ),
        migrations.AddField(
            model_name="image",
            name="tags",
            field=models.ManyToManyField(blank=True, related_name="images", to="ratings.tag"),
        ),
    ]
