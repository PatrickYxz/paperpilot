"""Short transactions for internal context artifacts, archives, and breaker state."""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from paperpilot.web.db_models import (
    CompressionStateRow,
    ContextArtifactRow,
    ConversationRow,
    ResearchTaskRow,
    TaskEventRow,
    TurnArchiveRow,
)
from paperpilot.web.store.helpers import (
    SessionFactory,
    compression_state_from_row,
    context_artifact_from_row,
    turn_archive_from_row,
    utc_now,
)
from paperpilot.web.store.records import (
    CompressionOutcomeRecord,
    CompressionStateRecord,
    ContextArtifactRecord,
    NewContextArtifact,
    TurnArchiveRecord,
    TurnArchiveSeedRecord,
)


def create_context_artifact(
    session_factory: SessionFactory,
    *,
    record: NewContextArtifact,
) -> ContextArtifactRecord:
    with session_factory.begin() as session:
        existing = session.get(ContextArtifactRow, record.artifact_id)
        if existing is not None:
            _assert_same_artifact(existing, record)
            return context_artifact_from_row(existing)
        row = ContextArtifactRow(**record.__dict__)
        session.add(row)
        try:
            with session.begin_nested():
                session.flush()
        except IntegrityError:
            existing = session.scalar(
                select(ContextArtifactRow).where(
                    ContextArtifactRow.task_id == record.task_id,
                    ContextArtifactRow.tool_call_id == record.tool_call_id,
                    ContextArtifactRow.sha256 == record.sha256,
                )
            )
            if existing is None:
                raise
            _assert_same_artifact(existing, record)
            return context_artifact_from_row(existing)
        return context_artifact_from_row(row)


def get_context_artifact(
    session_factory: SessionFactory,
    artifact_id: str,
    *,
    conversation_id: str,
) -> ContextArtifactRecord | None:
    with session_factory() as session:
        row = session.scalar(
            select(ContextArtifactRow).where(
                ContextArtifactRow.artifact_id == artifact_id,
                ContextArtifactRow.conversation_id == conversation_id,
            )
        )
        return None if row is None else context_artifact_from_row(row)


def find_context_artifact_by_digest(
    session_factory: SessionFactory,
    *,
    conversation_id: str,
    tool_name: str,
    sha256: str,
) -> ContextArtifactRecord | None:
    with session_factory() as session:
        row = session.scalar(
            select(ContextArtifactRow)
            .where(
                ContextArtifactRow.conversation_id == conversation_id,
                ContextArtifactRow.tool_name == tool_name,
                ContextArtifactRow.sha256 == sha256,
            )
            .order_by(ContextArtifactRow.created_at.asc(), ContextArtifactRow.artifact_id.asc())
        )
        return None if row is None else context_artifact_from_row(row)


def seed_turn_archive(
    session_factory: SessionFactory,
    *,
    seed: TurnArchiveSeedRecord,
) -> TurnArchiveRecord:
    with session_factory.begin() as session:
        existing = session.scalar(
            select(TurnArchiveRow).where(
                TurnArchiveRow.conversation_id == seed.conversation_id,
                TurnArchiveRow.user_message_id == seed.user_message_id,
                TurnArchiveRow.archive_version == seed.archive_version,
            )
        )
        if existing is not None:
            _assert_same_seed(existing, seed)
            return turn_archive_from_row(existing)
        now = utc_now()
        row = TurnArchiveRow(
            archive_id=seed.archive_id,
            conversation_id=seed.conversation_id,
            task_id=seed.task_id,
            user_message_id=seed.user_message_id,
            terminal_status=seed.terminal_status,
            archive_version=seed.archive_version,
            seed_json=_canonical_json(seed.seed_json),
            narrative_summary=None,
            narrative_status="pending",
            supersedes_json=_canonical_json(seed.supersedes_json),
            created_at=seed.created_at,
            updated_at=now,
        )
        session.add(row)
        try:
            with session.begin_nested():
                session.flush()
        except IntegrityError:
            existing = session.scalar(
                select(TurnArchiveRow).where(
                    TurnArchiveRow.conversation_id == seed.conversation_id,
                    TurnArchiveRow.user_message_id == seed.user_message_id,
                    TurnArchiveRow.archive_version == seed.archive_version,
                )
            )
            if existing is None:
                raise
            _assert_same_seed(existing, seed)
            return turn_archive_from_row(existing)
        return turn_archive_from_row(row)


def list_turn_archives(
    session_factory: SessionFactory,
    conversation_id: str,
) -> list[TurnArchiveRecord]:
    with session_factory() as session:
        rows = list(
            session.scalars(
                select(TurnArchiveRow)
                .where(TurnArchiveRow.conversation_id == conversation_id)
                .order_by(
                    TurnArchiveRow.created_at.asc(),
                    TurnArchiveRow.archive_id.asc(),
                )
            )
        )
        return [turn_archive_from_row(row) for row in rows]


def claim_turn_archive_narrative(
    session_factory: SessionFactory,
    archive_id: str,
) -> TurnArchiveRecord | None:
    with session_factory.begin() as session:
        result = session.execute(
            update(TurnArchiveRow)
            .where(
                TurnArchiveRow.archive_id == archive_id,
                TurnArchiveRow.narrative_status.in_(["pending", "failed"]),
            )
            .values(narrative_status="running", updated_at=utc_now())
        )
        if result.rowcount != 1:
            return None
        row = session.get(TurnArchiveRow, archive_id)
        return None if row is None else turn_archive_from_row(row)


def complete_turn_archive_narrative(
    session_factory: SessionFactory,
    archive_id: str,
    *,
    narrative_summary: str,
) -> TurnArchiveRecord:
    if not narrative_summary.strip():
        raise ValueError("narrative_summary must not be blank")
    with session_factory.begin() as session:
        row = session.get(TurnArchiveRow, archive_id)
        if row is None:
            raise ValueError("archive not found")
        if row.narrative_status not in {"running", "complete"}:
            raise ValueError("archive narrative is not claimed")
        row.narrative_summary = narrative_summary
        row.narrative_status = "complete"
        row.updated_at = utc_now()
        session.flush()
        return turn_archive_from_row(row)


def fail_turn_archive_narrative(
    session_factory: SessionFactory,
    archive_id: str,
) -> TurnArchiveRecord:
    with session_factory.begin() as session:
        row = session.get(TurnArchiveRow, archive_id)
        if row is None:
            raise ValueError("archive not found")
        row.narrative_status = "failed"
        row.updated_at = utc_now()
        session.flush()
        return turn_archive_from_row(row)


def get_compression_state(
    session_factory: SessionFactory,
    conversation_id: str,
    compressor_version: str,
) -> CompressionStateRecord:
    with session_factory.begin() as session:
        row = session.scalar(
            select(CompressionStateRow).where(
                CompressionStateRow.conversation_id == conversation_id,
                CompressionStateRow.compressor_version == compressor_version,
            )
        )
        if row is None:
            row = CompressionStateRow(
                conversation_id=conversation_id,
                compressor_version=compressor_version,
                state="CLOSED",
                consecutive_failures=0,
                last_failure_type=None,
                last_input_digest=None,
                opened_at=None,
                updated_at=utc_now(),
            )
            session.add(row)
            session.flush()
        return compression_state_from_row(row)


def claim_compression_probe(
    session_factory: SessionFactory,
    conversation_id: str,
    compressor_version: str,
) -> CompressionStateRecord | None:
    """Atomically move one OPEN breaker to HALF_OPEN for a single probe."""
    with session_factory.begin() as session:
        result = session.execute(
            update(CompressionStateRow)
            .where(
                CompressionStateRow.conversation_id == conversation_id,
                CompressionStateRow.compressor_version == compressor_version,
                CompressionStateRow.state == "OPEN",
            )
            .values(state="HALF_OPEN", updated_at=utc_now())
        )
        if result.rowcount != 1:
            return None
        row = session.scalar(
            select(CompressionStateRow).where(
                CompressionStateRow.conversation_id == conversation_id,
                CompressionStateRow.compressor_version == compressor_version,
            )
        )
        if row is None:
            return None
        task_id = _latest_task_id(session, conversation_id)
        if task_id is not None:
            session.add(
                TaskEventRow(
                    task_id=task_id,
                    type="compression_circuit_half_open",
                    stage="full",
                    message="compression circuit half-open probe claimed",
                    payload_json=_canonical_json(
                        {
                            "stage": "full",
                            "compressor_version": compressor_version,
                            "reason": "probe_claimed",
                            "before_tokens": 0,
                            "after_tokens": 0,
                            "reclaimed_tokens": 0,
                            "protected_item_count": 0,
                            "archive_ref_count": 0,
                            "artifact_ref_count": 0,
                            "validation_failure_type": None,
                            "breaker_state": "HALF_OPEN",
                        }
                    ),
                    created_at=utc_now(),
                )
            )
        return compression_state_from_row(row)


def record_compression_outcome(
    session_factory: SessionFactory,
    *,
    outcome: CompressionOutcomeRecord,
) -> CompressionStateRecord:
    if outcome.failure_threshold < 1:
        raise ValueError("failure_threshold must be positive")
    with session_factory.begin() as session:
        row = session.scalar(
            select(CompressionStateRow).where(
                CompressionStateRow.conversation_id == outcome.conversation_id,
                CompressionStateRow.compressor_version == outcome.compressor_version,
            )
        )
        if row is None:
            row = CompressionStateRow(
                conversation_id=outcome.conversation_id,
                compressor_version=outcome.compressor_version,
                state="CLOSED",
                consecutive_failures=0,
                last_failure_type=None,
                last_input_digest=None,
                opened_at=None,
                updated_at=utc_now(),
            )
            session.add(row)
            session.flush()
        previous_state = row.state
        if outcome.success:
            row.state = "CLOSED"
            row.consecutive_failures = 0
            row.last_failure_type = None
            row.opened_at = None
            event_type = "compression_circuit_closed" if previous_state != "CLOSED" else None
        else:
            row.consecutive_failures += 1
            row.last_failure_type = outcome.failure_type
            row.last_input_digest = outcome.input_digest
            if row.consecutive_failures >= outcome.failure_threshold:
                row.state = "OPEN"
                if previous_state != "OPEN":
                    row.opened_at = utc_now()
            event_type = (
                "compression_candidate_rejected"
                if outcome.reason != "compression_failed"
                else None
            )
        row.updated_at = utc_now()

        task_id = outcome.task_id or _latest_task_id(session, outcome.conversation_id)
        if task_id is not None:
            payload = {
                "stage": outcome.stage,
                "compressor_version": outcome.compressor_version,
                "reason": outcome.reason,
                "before_tokens": outcome.before_tokens,
                "after_tokens": outcome.after_tokens,
                "reclaimed_tokens": outcome.reclaimed_tokens,
                "protected_item_count": outcome.protected_item_count,
                "archive_ref_count": outcome.archive_ref_count,
                "artifact_ref_count": outcome.artifact_ref_count,
                "validation_failure_type": outcome.failure_type,
                "breaker_state": row.state,
                "cache_hit_tokens": outcome.cache_hit_tokens,
                "cache_miss_tokens": outcome.cache_miss_tokens,
            }
            if event_type is not None:
                session.add(
                    TaskEventRow(
                        task_id=task_id,
                        type=event_type,
                        stage=outcome.stage,
                        message=outcome.reason,
                        payload_json=_canonical_json(payload),
                        created_at=utc_now(),
                    )
                )
            if (
                not outcome.success
                and row.state == "OPEN"
                and previous_state != "OPEN"
            ):
                session.add(
                    TaskEventRow(
                        task_id=task_id,
                        type="compression_circuit_open",
                        stage=outcome.stage,
                        message="compression circuit opened",
                        payload_json=_canonical_json(payload),
                        created_at=utc_now(),
                    )
                )
        session.flush()
        return compression_state_from_row(row)


def is_context_artifact_referenced(
    session_factory: SessionFactory,
    *,
    artifact_id: str,
) -> bool:
    with session_factory() as session:
        return session.get(ContextArtifactRow, artifact_id) is not None


def _latest_task_id(session, conversation_id: str) -> str | None:
    return session.scalar(
        select(ResearchTaskRow.id)
        .where(ResearchTaskRow.conversation_id == conversation_id)
        .order_by(ResearchTaskRow.created_at.desc(), ResearchTaskRow.id.desc())
        .limit(1)
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assert_same_artifact(row: ContextArtifactRow, record: NewContextArtifact) -> None:
    for field in record.__dataclass_fields__:
        if getattr(row, field) != getattr(record, field):
            raise ValueError("context artifact idempotency conflict")


def _assert_same_seed(row: TurnArchiveRow, seed: TurnArchiveSeedRecord) -> None:
    if (
        row.archive_id != seed.archive_id
        or row.task_id != seed.task_id
        or row.terminal_status != seed.terminal_status
        or row.seed_json != _canonical_json(seed.seed_json)
        or row.supersedes_json != _canonical_json(seed.supersedes_json)
    ):
        raise ValueError("turn archive seed idempotency conflict")
