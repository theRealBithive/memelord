from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ratings', '0015_add_keyword_suggestions'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='knn_tag_suggestions',
            field=models.CharField(blank=True, default='', max_length=500),
        ),
    ]
