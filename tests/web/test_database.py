"""SQLAlchemy engine configuration tests for the Web database."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from paperpilot.web.database import (
    build_sqlite_url,
    create_session_factory,
    create_task_engine,
    resolve_task_db_path,
)


def test_resolve_task_db_path_prefers_argument_over_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_path = tmp_path / "from-env.sqlite3"
    explicit_path = tmp_path / "explicit.sqlite3"
    monkeypatch.setenv("PAPERPILOT_TASK_DB_PATH", str(env_path))

    assert resolve_task_db_path(explicit_path) == explicit_path.resolve()
    assert resolve_task_db_path() == env_path.resolve()


def test_sqlite_url_preserves_spaces_in_absolute_path(tmp_path: Path) -> None:
    db_path = (tmp_path / "folder with spaces" / "tasks.sqlite3").resolve()

    url = build_sqlite_url(db_path)

    assert url.drivername == "sqlite+pysqlite"
    assert Path(url.database or "") == db_path


def test_engine_applies_sqlite_pragmas(tmp_path: Path) -> None:
    engine = create_task_engine(tmp_path / "tasks.sqlite3")
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
            assert (
                connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()
                == 30_000
            )
            assert connection.exec_driver_sql("PRAGMA synchronous").scalar_one() == 1
            assert (
                connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
                == "wal"
            )
    finally:
        engine.dispose()


def test_session_transactions_provide_repeatable_read_snapshot(
    tmp_path: Path,
) -> None:
    engine = create_task_engine(tmp_path / "tasks.sqlite3")
    sessions = create_session_factory(engine)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE probe (value INTEGER NOT NULL)")
            connection.exec_driver_sql("INSERT INTO probe(value) VALUES (1)")

        with sessions.begin() as first_session:
            first_value = first_session.execute(
                text("SELECT value FROM probe")
            ).scalar_one()
            with sessions.begin() as second_session:
                second_session.execute(text("UPDATE probe SET value = 2"))
            repeated_value = first_session.execute(
                text("SELECT value FROM probe")
            ).scalar_one()

        assert first_value == 1
        assert repeated_value == 1
    finally:
        engine.dispose()
