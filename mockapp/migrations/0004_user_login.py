# Generated by Django 5.1.7 on 2025-03-21 10:05

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mockapp', '0003_remove_user_login'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='Login',
            field=models.OneToOneField(default=0, on_delete=django.db.models.deletion.CASCADE, to='mockapp.login'),
            preserve_default=False,
        ),
    ]
