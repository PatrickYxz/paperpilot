"""SQLite-backed LangGraph checkpoint lifecycle for the Web runtime."""
from __future__ import annotations

import argparse
import fcntl
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

from paperpilot.web.config import WebRuntimeConfig

if TYPE_CHECKING:
    from langgraph.checkpoint.sqlite import SqliteSaver


_STRICT_MSGPACK_TRUE_VALUES = {"1", "true", "yes"}


def resolve_checkpoint_db_path(path: Path | str | None = None) -> Path:
    """Return an absolute checkpoint path from an explicit value or Web config."""
    configured = (
        Path(path)
        if path is not None
        else WebRuntimeConfig.from_env().checkpoint_db_path
    )
    return configured.expanduser().resolve()


def _require_strict_msgpack() -> None:
    raw = os.environ.get("LANGGRAPH_STRICT_MSGPACK")
    if raw is None:
        os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
        return
    if raw.strip().lower() not in _STRICT_MSGPACK_TRUE_VALUES:
        raise ValueError(
            "LANGGRAPH_STRICT_MSGPACK must be enabled with 'true', '1', or 'yes'"
        )


@contextmanager
def _checkpoint_setup_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f"{path.name}.setup.lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass
class SqliteCheckpointRuntime:
    """Own one SQLite connection and the LangGraph saver built on it."""

    path: Path
    connection: sqlite3.Connection
    saver: SqliteSaver
    _closed: bool = False

    @classmethod
    def open(
        cls, path: Path | str | None = None
    ) -> "SqliteCheckpointRuntime":
        _require_strict_msgpack()

        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from langgraph.checkpoint.sqlite import SqliteSaver

        resolved = resolve_checkpoint_db_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            resolved,
            timeout=30,
            check_same_thread=False,
        )
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            saver = SqliteSaver(
                connection,
                serde=JsonPlusSerializer(allowed_msgpack_modules=None),
            )
            with _checkpoint_setup_lock(resolved):
                saver.setup()
        except BaseException:
            connection.close()
            raise
        return cls(path=resolved, connection=connection, saver=saver)

    def check_health(self) -> None:
        """Raise if the owned SQLite connection is unavailable."""
        cursor = self.connection.execute("SELECT 1")
        try:
            cursor.fetchone()
        finally:
            cursor.close()

    def close(self) -> None:
        """Close the owned connection once."""
        if self._closed:
            return
        self.connection.close()
        self._closed = True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup",
        action="store_true",
        help="create and health-check the checkpoint database",
    )
    args = parser.parse_args(argv)
    if not args.setup:
        parser.error("--setup is required")

    runtime = SqliteCheckpointRuntime.open()
    try:
        runtime.check_health()
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
