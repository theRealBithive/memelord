"""Every SQLite connection comes up in WAL mode with NORMAL sync.

WAL is what keeps the app responsive during a background scrape/train: it lets
the gallery/review reads proceed while the django-q worker writes. A regression
back to the default rollback journal would silently re-introduce the "UI freezes
at the end of a scrape" stall, so pin the PRAGMAs here.
"""

import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "memelord.settings")
django.setup()

from django.db import connection
from django.test import TestCase


class SqlitePragmaTests(TestCase):
    def test_connection_is_wal_with_normal_sync(self) -> None:
        with connection.cursor() as cursor:
            journal_mode = cursor.execute("PRAGMA journal_mode").fetchone()[0]
            synchronous = cursor.execute("PRAGMA synchronous").fetchone()[0]
        self.assertEqual(journal_mode.lower(), "wal")
        # synchronous=NORMAL is reported as the integer 1.
        self.assertEqual(synchronous, 1)
