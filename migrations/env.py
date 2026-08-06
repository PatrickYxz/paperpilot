"""Alembic environment for PaperPilot's Web business schema."""
from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import Connection, make_url

from paperpilot.web.database import create_task_engine, resolve_task_db_path
from paperpilot.web.db_models import Base


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(object_, name, type_, reflected, compare_to):
    """Keep tables owned by other PaperPilot runtimes out of autogenerate."""
    if type_ == "table" and reflected and name not in target_metadata.tables:
        return False
    return True


def configure_context(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=False,
        render_as_batch=True,
        include_object=include_object,
    )


def run_migrations_offline() -> None:
    raise RuntimeError(
        "Offline SQL is unsupported for the adoption migration because "
        "it must inspect the target SQLite schema"
    )


def run_migrations_online() -> None:
    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        configure_context(supplied_connection)
        with context.begin_transaction():
            context.run_migrations()
        return

    database_path = config.attributes.get("paperpilot_db_path")
    if database_path is None:
        configured_path = os.getenv("PAPERPILOT_TASK_DB_PATH")
        if configured_path is not None:
            database_path = resolve_task_db_path(configured_path)
        else:
            url = make_url(config.get_main_option("sqlalchemy.url"))
            if not url.database:
                raise RuntimeError("SQLite database path is required")
            database_path = Path(url.database)
    engine = create_task_engine(database_path)
    try:
        with engine.connect() as connection:
            configure_context(connection)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
