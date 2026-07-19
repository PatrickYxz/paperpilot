from dataclasses import FrozenInstanceError

import pytest

from paperpilot.agent.models import (
    AgentRun,
    AgentStep,
    RUN_TERMINAL_STATUSES,
    RunCheckpoint,
    RunOutcome,
    StepPage,
    ToolExecution,
)
from paperpilot.agent.policy import RunPolicy


def test_agent_run_terminal_state_is_immutable():
    run = AgentRun(
        id="run-1",
        task_id="task-1",
        status="completed",
        attempt=1,
        current_step=2,
        cancel_requested_at=None,
        retry_at=None,
        failure_class=None,
        failure_message=None,
        policy=RunPolicy.for_depth("standard"),
        owner_id=None,
        lease_expires_at=None,
        schema_version=1,
        created_at="2026-07-19T00:00:00+00:00",
        updated_at="2026-07-19T00:01:00+00:00",
        started_at="2026-07-19T00:00:01+00:00",
        finished_at="2026-07-19T00:01:00+00:00",
    )

    assert run.is_terminal is True
    assert "completed" in RUN_TERMINAL_STATUSES
    with pytest.raises(FrozenInstanceError):
        run.status = "running"


def test_agent_run_serialization_excludes_private_lease_fields_for_public_views():
    run = _agent_run()

    assert run.to_dict() == {
        "id": "run-1",
        "task_id": "task-1",
        "status": "running",
        "attempt": 1,
        "current_step": 2,
        "cancel_requested_at": None,
        "retry_at": None,
        "failure_class": None,
        "failure_message": None,
        "policy": RunPolicy.for_depth("standard").to_dict(),
        "schema_version": 1,
        "created_at": "2026-07-19T00:00:00+00:00",
        "updated_at": "2026-07-19T00:01:00+00:00",
        "started_at": "2026-07-19T00:00:01+00:00",
        "finished_at": None,
        "owner_id": "worker-1",
        "lease_expires_at": "2026-07-19T00:06:00+00:00",
    }
    assert "owner_id" not in run.to_dict(public=True)
    assert "lease_expires_at" not in run.to_dict(public=True)


def test_agent_step_exactly_matches_persisted_step_shape_and_is_frozen():
    first_attempt = _agent_step(attempt=1)
    retry_attempt = _agent_step(attempt=2)

    assert first_attempt.to_dict() == {
        "id": "step-1",
        "run_id": "run-1",
        "sequence": 1,
        "kind": "tool",
        "status": "completed",
        "attempt": 1,
        "input": {"tool": "colbert.search", "query": "retrieval"},
        "output": {"count": 3},
        "error": None,
        "schema_version": 1,
        "started_at": "2026-07-19T00:00:01+00:00",
        "finished_at": "2026-07-19T00:00:02+00:00",
    }
    assert (first_attempt.run_id, first_attempt.sequence, first_attempt.attempt) != (
        retry_attempt.run_id,
        retry_attempt.sequence,
        retry_attempt.attempt,
    )
    with pytest.raises(FrozenInstanceError):
        first_attempt.attempt = 2


def test_checkpoint_exactly_matches_recovery_schema_and_is_frozen():
    checkpoint = RunCheckpoint(
        id="checkpoint-1",
        run_id="run-1",
        step_sequence=1,
        messages=[{"role": "user", "content": "Compare papers"}],
        runtime_state={"turn_count": 1, "tokens_used": 120},
        schema_version=1,
        created_at="2026-07-19T00:00:02+00:00",
    )

    assert checkpoint.step_sequence == 1
    assert checkpoint.messages[0]["content"] == "Compare papers"
    assert checkpoint.to_dict() == {
        "id": "checkpoint-1",
        "run_id": "run-1",
        "step_sequence": 1,
        "messages": [{"role": "user", "content": "Compare papers"}],
        "runtime_state": {"turn_count": 1, "tokens_used": 120},
        "schema_version": 1,
        "created_at": "2026-07-19T00:00:02+00:00",
    }
    with pytest.raises(FrozenInstanceError):
        checkpoint.step_sequence = 2


@pytest.mark.parametrize(
    ("status", "result_preview", "failure_class", "failure_message"),
    [
        ("started", None, None, None),
        ("completed", "3 results", None, None),
        ("failed", None, "transport", "connection closed"),
    ],
)
def test_tool_execution_exactly_matches_persisted_lifecycle_shape(
    status, result_preview, failure_class, failure_message
):
    execution = ToolExecution(
        id="tool-1",
        run_id="run-1",
        step_id="step-1",
        tool_name="colbert.search",
        arguments={"query": "retrieval", "k": 3},
        classification="read_only",
        status=status,
        result_preview=result_preview,
        failure_class=failure_class,
        failure_message=failure_message,
        duration_ms=25,
        schema_version=1,
        started_at="2026-07-19T00:00:01+00:00",
        finished_at="2026-07-19T00:00:02+00:00",
    )

    assert execution.to_dict() == {
        "id": "tool-1",
        "run_id": "run-1",
        "step_id": "step-1",
        "tool_name": "colbert.search",
        "arguments": {"query": "retrieval", "k": 3},
        "classification": "read_only",
        "status": status,
        "result_preview": result_preview,
        "failure_class": failure_class,
        "failure_message": failure_message,
        "duration_ms": 25,
        "schema_version": 1,
        "started_at": "2026-07-19T00:00:01+00:00",
        "finished_at": "2026-07-19T00:00:02+00:00",
    }
    with pytest.raises(FrozenInstanceError):
        execution.status = "failed"


def test_run_outcome_constructors_and_step_page_serialize_stable_values():
    completed = RunOutcome.completed("run-1", "step-2", "Final answer")
    waiting_retry = RunOutcome.waiting_retry("run-1", "transport", "closed")
    failed = RunOutcome.failed("run-1", "validation", "bad input")
    cancelled = RunOutcome.cancelled("run-1", "cancelled by user")
    page = StepPage(items=(_agent_step(),), next_after_sequence=1, has_more=True)

    assert completed.to_dict() == {
        "run_id": "run-1",
        "status": "completed",
        "final_step_id": "step-2",
        "final_text": "Final answer",
        "failure_class": None,
        "failure_message": None,
    }
    assert waiting_retry.failure_class == "transport"
    assert failed.failure_message == "bad input"
    assert cancelled.to_dict()["failure_class"] == "cancelled"
    assert page.to_dict() == {
        "items": [_agent_step().to_dict()],
        "next_after_sequence": 1,
        "has_more": True,
    }
    with pytest.raises(FrozenInstanceError):
        page.has_more = False


def _agent_run() -> AgentRun:
    return AgentRun(
        id="run-1",
        task_id="task-1",
        status="running",
        attempt=1,
        current_step=2,
        cancel_requested_at=None,
        retry_at=None,
        failure_class=None,
        failure_message=None,
        policy=RunPolicy.for_depth("standard"),
        owner_id="worker-1",
        lease_expires_at="2026-07-19T00:06:00+00:00",
        schema_version=1,
        created_at="2026-07-19T00:00:00+00:00",
        updated_at="2026-07-19T00:01:00+00:00",
        started_at="2026-07-19T00:00:01+00:00",
        finished_at=None,
    )


def _agent_step(*, attempt: int = 1) -> AgentStep:
    return AgentStep(
        id=f"step-{attempt}",
        run_id="run-1",
        sequence=1,
        kind="tool",
        status="completed",
        attempt=attempt,
        input={"tool": "colbert.search", "query": "retrieval"},
        output={"count": 3},
        error=None,
        schema_version=1,
        started_at="2026-07-19T00:00:01+00:00",
        finished_at="2026-07-19T00:00:02+00:00",
    )
