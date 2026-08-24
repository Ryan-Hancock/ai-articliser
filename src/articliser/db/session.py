"""Engine and session helpers.

SQLite is a deliberate fit here rather than a placeholder: the web process only
ever reads articles, and the single worker is the only writer, so there is no
write contention to design around. WAL mode is still enabled so a long
generation transaction can't block the reading site.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from articliser.config import settings

# Import for side effects: SQLModel.metadata is only populated once the table
# classes have been defined, so create_all() below would otherwise be a no-op.
from articliser.db import models  # noqa: F401

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        settings.ensure_dirs()
        _engine = create_engine(f"sqlite:///{settings.db_path}", echo=False)

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


# Columns added after the first release. SQLModel's create_all() only creates
# missing *tables*, so a database that predates these would keep working right up
# until the first query touched one, then fail with "no such column".
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("article", "series_id", "INTEGER"),
    ("article", "series_index", "INTEGER"),
    ("article", "series_part", "TEXT"),
    ("article", "series_chapter", "TEXT"),
    ("series", "total_parts", "INTEGER DEFAULT 0"),
)


def _add_missing_columns() -> None:
    """Bring an existing database up to the current model.

    A deliberately minimal migration: additive, idempotent, and no version table.
    Enough for columns, and honest about being nothing more -- a change that
    needed data rewritten or a column dropped would want a real migration tool.
    """
    engine = get_engine()
    with engine.begin() as connection:
        for table, column, column_type in _ADDED_COLUMNS:
            existing = {
                row[1] for row in connection.exec_driver_sql(f"PRAGMA table_info({table})")
            }
            if not existing:
                continue  # table doesn't exist yet; create_all will make it correctly
            if column not in existing:
                connection.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                )


def init_db() -> None:
    SQLModel.metadata.create_all(get_engine())
    _add_missing_columns()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Session that commits on success and rolls back on any exception."""
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency -- read-only in the web process, so no commit."""
    with Session(get_engine()) as session:
        yield session
