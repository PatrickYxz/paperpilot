"""SQLAlchemy configuration for the PaperPilot Web SQLite database."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, URL, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


DEFAULT_TASK_DB_PATH = Path("data/web/tasks.sqlite3")


def resolve_task_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve an explicit or configured task database path."""
    configured = db_path or os.getenv("PAPERPILOT_TASK_DB_PATH") or DEFAULT_TASK_DB_PATH
    return Path(configured).expanduser().resolve()


def build_sqlite_url(db_path: Path) -> URL:
    """Build a SQLite URL without string interpolation or path escaping."""
    return URL.create("sqlite+pysqlite", database=str(db_path))


def create_task_engine(db_path: Path) -> Engine:
    """Create an Engine with PaperPilot's SQLite concurrency policy."""
    resolved_path = resolve_task_db_path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        build_sqlite_url(resolved_path),
        connect_args={"timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection, _connection_record) -> None:
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            journal_mode = str(cursor.fetchone()[0]).strip().lower()
            if journal_mode != "wal":
                raise RuntimeError(
                    "SQLite WAL initialization failed: "
                    f"returned {journal_mode!r}"
                )
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

    @event.listens_for(engine, "begin")
    def begin_sqlite_transaction(connection) -> None:
        connection.exec_driver_sql("BEGIN")

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create short-lived synchronous Sessions for Web persistence."""
    return sessionmaker(bind=engine, expire_on_commit=False)
