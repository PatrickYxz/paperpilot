"""Programmatic Alembic entry points for the Web business database."""
from __future__ import annotations

from pathlib import Path
from threading import Lock

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory

from paperpilot.web.database import (
    build_sqlite_url,
    create_task_engine,
    resolve_task_db_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI_PATH = PROJECT_ROOT / "alembic.ini"
MIGRATIONS_PATH = PROJECT_ROOT / "migrations"
_MIGRATION_LOCK = Lock()


def build_alembic_config(db_path: Path | str) -> Config:
    """Build an Alembic Config targeting one explicit SQLite file."""
    resolved_path = resolve_task_db_path(db_path)
    config = Config(str(ALEMBIC_INI_PATH))
    config.set_main_option("script_location", str(MIGRATIONS_PATH))
    config.attributes["paperpilot_db_path"] = resolved_path
    rendered_url = build_sqlite_url(resolved_path).render_as_string(
        hide_password=False
    )
    config.set_main_option("sqlalchemy.url", rendered_url.replace("%", "%%"))
    return config


def get_database_heads(db_path: Path | str) -> set[str]:
    """Return the revisions currently stamped in a database."""
    engine = create_task_engine(resolve_task_db_path(db_path))
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            return set(context.get_current_heads())
    finally:
        engine.dispose()


def get_script_heads() -> set[str]:
    """Return the heads available in the repository migration scripts."""
    config = Config(str(ALEMBIC_INI_PATH))
    config.set_main_option("script_location", str(MIGRATIONS_PATH))
    return set(ScriptDirectory.from_config(config).get_heads())


def upgrade_database(db_path: Path | str, revision: str = "head") -> None:
    """Upgrade an explicit SQLite database to a migration revision."""
    command.upgrade(build_alembic_config(db_path), revision)


def ensure_database_current(db_path: Path | str) -> None:
    """Upgrade a database only when its stamped heads are not current."""
    resolved_path = resolve_task_db_path(db_path)
    with _MIGRATION_LOCK:
        if get_database_heads(resolved_path) == get_script_heads():
            return
        upgrade_database(resolved_path)
