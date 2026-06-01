"""The connection_created handler flips every SQLite connection to WAL + NORMAL.

WAL is what keeps the app responsive during a background scrape/train: it lets
the gallery/review reads proceed while the django-q worker writes. A regression
back to the default rollback journal would silently re-introduce the "UI freezes
at the end of a scrape" stall, so pin the handler's behaviour here.

This drives ``_set_sqlite_pragmas`` directly against a throwaway on-disk SQLite
file rather than the live Django connection. journal_mode is a *persistent*
file-level property, so asserting against the shared prod DB would pass on
ambient WAL state left by prior runs even if the handler never ran — and would
mutate that production file. A fresh temp file defaults to the rollback journal,
so the delete→wal flip can only come from the code under test.
"""

import sqlite3
import tempfile
from pathlib import Path

from ratings.apps import _set_sqlite_pragmas


class _CursorCtx:
    """Make a raw sqlite3 cursor a context manager like Django's cursor.

    ``_set_sqlite_pragmas`` does ``with connection.cursor() as cursor:`` — the
    Django DB API cursor supports the context-manager protocol but sqlite3's
    bare cursor does not, so wrap it to match what the handler expects.
    """

    def __init__(self, raw_conn: sqlite3.Connection) -> None:
        self._cursor = raw_conn.cursor()

    def __enter__(self) -> sqlite3.Cursor:
        return self._cursor

    def __exit__(self, *exc: object) -> bool:
        self._cursor.close()
        return False


class _FakeSqliteConnection:
    """Minimal stand-in exposing the two attributes the handler touches."""

    vendor = "sqlite"

    def __init__(self, raw_conn: sqlite3.Connection) -> None:
        self._raw = raw_conn

    def cursor(self) -> _CursorCtx:
        return _CursorCtx(self._raw)


def test_handler_flips_default_journal_to_wal_with_normal_sync() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = sqlite3.connect(Path(tmp) / "probe.db")
        try:
            # A fresh on-disk SQLite DB comes up in the rollback journal with
            # synchronous=FULL (reported as 2) — the very state we want to leave.
            assert raw.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
            assert raw.execute("PRAGMA synchronous").fetchone()[0] == 2

            _set_sqlite_pragmas(sender=None, connection=_FakeSqliteConnection(raw))

            journal_mode = raw.execute("PRAGMA journal_mode").fetchone()[0]
            synchronous = raw.execute("PRAGMA synchronous").fetchone()[0]
        finally:
            raw.close()

    assert journal_mode.lower() == "wal"
    # synchronous=NORMAL is reported as the integer 1.
    assert synchronous == 1


class _CannedCursor:
    """Cursor that reports WAL was *not* adopted, to exercise the warn path.

    On a real WAL-capable local filesystem the PRAGMA always takes, so the
    fallback branch can't be reached with a genuine sqlite3 connection — fake
    the readback to simulate a mount that silently keeps the rollback journal.
    """

    def execute(self, sql: str) -> None:
        pass

    def fetchone(self) -> tuple[str]:
        return ("delete",)

    def __enter__(self) -> "_CannedCursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _StuckOnDeleteConnection:
    vendor = "sqlite"

    def cursor(self) -> _CannedCursor:
        return _CannedCursor()


def test_handler_warns_when_wal_not_adopted(caplog) -> None:
    _set_sqlite_pragmas(sender=None, connection=_StuckOnDeleteConnection())
    assert "WAL mode not adopted" in caplog.text
    assert "delete" in caplog.text


def test_handler_is_a_noop_for_non_sqlite() -> None:
    """Guards the vendor check: Postgres/MySQL connections must be left alone."""

    class _OtherVendor:
        vendor = "postgresql"

        def cursor(self):  # pragma: no cover - must never be called
            raise AssertionError("handler touched a non-sqlite connection")

    _set_sqlite_pragmas(sender=None, connection=_OtherVendor())
