from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("ratings", "0016_add_knn_tag_suggestions"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="image",
            name="keyword_suggestions",
        ),
    ]
