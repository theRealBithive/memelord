from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ratings', '0009_add_is_purged'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='inbox_seen_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='image',
            name='corpus_seen_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
