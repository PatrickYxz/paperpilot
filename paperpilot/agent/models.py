"""Stable, immutable records shared by the agent runtime and persistence layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from paperpilot.agent.policy import RunPolicy


RunStatus = Literal[
    "pending", "running", "waiting_retry", "cancelling", "completed", "failed", "cancelled"
]
StepKind = Literal["llm", "tool", "subagent", "compact", "finalize"]
StepStatus = Literal["started", "completed", "failed", "cancelled"]
FailureClass = Literal[
    "validation",
    "authentication",
    "rate_limit",
    "timeout",
    "transport",
    "capacity",
    "cancelled",
    "internal",
]
ToolClassification = Literal["read_only", "idempotent_write", "non_retryable"]

RUN_ACTIVE_STATUSES = frozenset({"pending", "running", "waiting_retry", "cancelling"})
RUN_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
STEP_KINDS = frozenset({"llm", "tool", "subagent", "compact", "finalize"})
STEP_STATUSES = frozenset({"started", "completed", "failed", "cancelled"})
FAILURE_CLASSES = frozenset(
    {
        "validation",
        "authentication",
        "rate_limit",
        "timeout",
        "transport",
        "capacity",
        "cancelled",
        "internal",
    }
)
TOOL_CLASSIFICATIONS = frozenset({"read_only", "idempotent_write", "non_retryable"})
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentRun:
    id: str
    task_id: str
    status: RunStatus
    attempt: int
    current_step: int
    cancel_requested_at: str | None
    retry_at: str | None
    failure_class: FailureClass | None
    failure_message: str | None
    policy: RunPolicy
    owner_id: str | None
    lease_expires_at: str | None
    schema_version: int
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None

    @property
    def is_terminal(self) -> bool:
        return self.status in RUN_TERMINAL_STATUSES

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "task_id": self.task_id,
            "status": self.status,
            "attempt": self.attempt,
            "current_step": self.current_step,
            "cancel_requested_at": self.cancel_requested_at,
            "retry_at": self.retry_at,
            "failure_class": self.failure_class,
            "failure_message": self.failure_message,
            "policy": self.policy.to_dict(),
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if not public:
            payload.update(
                {
                    "owner_id": self.owner_id,
                    "lease_expires_at": self.lease_expires_at,
                }
            )
        return payload


@dataclass(frozen=True)
class AgentStep:
    id: str
    run_id: str
    sequence: int
    kind: StepKind
    status: StepStatus
    attempt: int
    input: dict[str, Any]
    output: dict[str, Any] | None
    error: dict[str, Any] | None
    schema_version: int
    started_at: str | None
    finished_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "status": self.status,
            "attempt": self.attempt,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True)
class RunCheckpoint:
    id: str
    run_id: str
    step_sequence: int
    messages: list[dict[str, Any]]
    runtime_state: dict[str, Any]
    schema_version: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_sequence": self.step_sequence,
            "messages": self.messages,
            "runtime_state": self.runtime_state,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ToolExecution:
    id: str
    run_id: str
    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    classification: ToolClassification
    status: str
    result_preview: str | None
    failure_class: FailureClass | None
    failure_message: str | None
    duration_ms: int | None
    schema_version: int
    started_at: str | None
    finished_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "classification": self.classification,
            "status": self.status,
            "result_preview": self.result_preview,
            "failure_class": self.failure_class,
            "failure_message": self.failure_message,
            "duration_ms": self.duration_ms,
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: RunStatus
    final_step_id: str | None
    final_text: str | None
    failure_class: FailureClass | None
    failure_message: str | None

    @classmethod
    def completed(cls, run_id: str, final_step_id: str, final_text: str) -> "RunOutcome":
        return cls(run_id, "completed", final_step_id, final_text, None, None)

    @classmethod
    def waiting_retry(
        cls, run_id: str, failure_class: FailureClass, failure_message: str
    ) -> "RunOutcome":
        return cls(run_id, "waiting_retry", None, None, failure_class, failure_message)

    @classmethod
    def failed(
        cls, run_id: str, failure_class: FailureClass, failure_message: str
    ) -> "RunOutcome":
        return cls(run_id, "failed", None, None, failure_class, failure_message)

    @classmethod
    def cancelled(cls, run_id: str, failure_message: str | None = None) -> "RunOutcome":
        return cls(run_id, "cancelled", None, None, "cancelled", failure_message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "final_step_id": self.final_step_id,
            "final_text": self.final_text,
            "failure_class": self.failure_class,
            "failure_message": self.failure_message,
        }


@dataclass(frozen=True)
class StepPage:
    items: tuple[AgentStep, ...]
    next_after_sequence: int | None
    has_more: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "next_after_sequence": self.next_after_sequence,
            "has_more": self.has_more,
        }
