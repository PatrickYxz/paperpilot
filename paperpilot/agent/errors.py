"""Failure normalization and retry decisions for durable agent execution."""

from __future__ import annotations

from dataclasses import dataclass

from paperpilot.agent.models import FailureClass, ToolClassification
from paperpilot.tools.mcp_client import MCPToolTimeout, MCPTransportError

_MAX_FAILURE_MESSAGE_LENGTH = 2_000


@dataclass(frozen=True)
class ClassifiedFailure:
    failure_class: FailureClass
    message: str
    transient: bool

    @property
    def persisted_message(self) -> str:
        return self.message[:_MAX_FAILURE_MESSAGE_LENGTH]


class ToolExecutionFailure(RuntimeError):
    """A tool invocation failed after its durable execution record was started."""

    def __init__(
        self,
        failure: ClassifiedFailure,
        classification: ToolClassification,
        retryable: bool,
    ) -> None:
        super().__init__(failure.message)
        self.failure = failure
        self.classification = classification
        self.retryable = retryable


class ToolExecutionReplayRequired(RuntimeError):
    """The caller must recover a completed tool result from a checkpoint."""

    def __init__(self, execution: object) -> None:
        super().__init__("tool execution is completed and requires checkpoint replay")
        self.execution = execution


class ToolExecutionBlocked(RuntimeError):
    """A durable execution exists but policy forbids another external call."""

    def __init__(self, execution: object, disposition: str) -> None:
        super().__init__(f"tool execution cannot run: {disposition}")
        self.execution = execution
        self.disposition = disposition


def classify_exception(exc: BaseException) -> ClassifiedFailure:
    """Map provider and local exceptions to the stable runtime failure contract."""
    if isinstance(exc, (MCPToolTimeout, TimeoutError)):
        return _failure("timeout", str(exc), transient=True)
    if isinstance(exc, (MCPTransportError, ConnectionError, EOFError, OSError)):
        return _failure("transport", str(exc), transient=True)

    status_code = getattr(exc, "status_code", None)
    if status_code in {401, 403}:
        return _failure("authentication", str(exc), transient=False)
    if status_code == 429:
        return _failure("rate_limit", str(exc), transient=True)
    if status_code in {502, 503, 504}:
        return _failure("transport", str(exc), transient=True)
    if isinstance(exc, (KeyError, TypeError, ValueError)):
        return _failure("validation", str(exc), transient=False)
    return _failure("internal", f"{type(exc).__name__}: {exc}", transient=False)


def is_step_retryable(
    failure: ClassifiedFailure, classification: ToolClassification
) -> bool:
    return failure.transient and classification in {"read_only", "idempotent_write"}


def _failure(
    failure_class: FailureClass, message: str, *, transient: bool
) -> ClassifiedFailure:
    return ClassifiedFailure(
        failure_class=failure_class,
        message=message,
        transient=transient,
    )
