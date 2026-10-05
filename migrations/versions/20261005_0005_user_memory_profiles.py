"""Add materialized user memory profile table."""
from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20261005_0005"
down_revision: str | Sequence[str] | None = "20261005_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_memory_profiles",
        sa.Column(
            "user_id",
            sa.Text(),
            sa.ForeignKey(
                "users.id", name="fk_user_memory_profiles_user_id_users"
            ),
            primary_key=True,
        ),
        sa.Column("profile_text", sa.Text(), nullable=False),
        sa.Column("source_memory_count", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    raise RuntimeError(
        "downgrade is intentionally unsupported for 20261005_0005 profiles"
    )
