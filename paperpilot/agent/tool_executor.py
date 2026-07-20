"""Policy-gated tool execution for durable agent runs."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from paperpilot.agent.errors import (
    ClassifiedFailure,
    ToolExecutionBlocked,
    ToolExecutionFailure,
    ToolExecutionReplayRequired,
    classify_exception,
    is_step_retryable,
)
from paperpilot.agent.models import ToolClassification
from paperpilot.agent.policy import RunPolicy
from paperpilot.agent.store import RunStore
from paperpilot.core.adapter import Tool, ToolCall, ToolResult

READ_ONLY_TOOL_SUFFIXES = (
    "__search",
    "__planned_retrieval",
    "__get_paper",
    "__get_citations",
    "__get_references",
    "__analyze_figure",
)
IDEMPOTENT_WRITE_TOOL_SUFFIXES = ("__build_index",)
READ_ONLY_BUILTINS = frozenset(
    {
        "load_skill",
        "research_todo",
        "compact_context",
        "search_user_document",
    }
)
_SECRET_ARGUMENT_KEYS = frozenset(
    {"api_key", "authorization", "cookie", "password", "secret", "token"}
)
_LONG_TEXT_LIMIT = 2_000
_LONG_TEXT_PREVIEW_LENGTH = 200
_RESULT_PREVIEW_LENGTH = 1_000


@dataclass(frozen=True)
class ToolInvocationResult:
    execution_id: str
    tool_result: ToolResult
    result_preview: str
    duration_ms: int
    classification: ToolClassification
    sanitized_arguments: dict[str, Any]


class ToolPolicyRegistry:
    """Classify operation safety by the explicit runtime tool policy."""

    def classify(self, tool_name: str) -> ToolClassification:
        if tool_name == "paper_deep_read":
            return "non_retryable"
        if tool_name in READ_ONLY_BUILTINS or tool_name.endswith(READ_ONLY_TOOL_SUFFIXES):
            return "read_only"
        if tool_name.endswith(IDEMPOTENT_WRITE_TOOL_SUFFIXES):
            return "idempotent_write"
        return "non_retryable"


class ToolExecutor:
    """Invoke registered handlers while preserving the RunStore ownership fence."""

    def __init__(
        self,
        *,
        store: RunStore,
        tools: Sequence[Tool],
        registry: ToolPolicyRegistry | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._tools = {tool.name: tool for tool in tools}
        self._registry = registry or ToolPolicyRegistry()
        self._monotonic = monotonic or time.monotonic

    def execute(
        self,
        *,
        run_id: str,
        step_id: str,
        owner_id: str,
        tool_call: ToolCall,
        policy: RunPolicy,
    ) -> ToolInvocationResult:
        classification = self._registry.classify(tool_call.name)
        self._ensure_allowed(tool_call.name, policy)
        tool = self._tools.get(tool_call.name)
        if tool is None:
            self._raise_validation(f"tool is not registered: {tool_call.name}", classification)

        sanitized_arguments = sanitize_tool_arguments(tool_call.arguments)
        start = self._store.start_tool_execution(
            run_id=run_id,
            step_id=step_id,
            owner_id=owner_id,
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=sanitized_arguments,
            classification=classification,
        )
        if start.disposition == "completed":
            raise ToolExecutionReplayRequired(start.execution)
        if not start.should_invoke:
            raise ToolExecutionBlocked(start.execution, start.disposition)

        execution = start.execution
        started_at = self._monotonic()
        try:
            content = _stringify_tool_content(tool.handler(tool_call.arguments))
        except Exception as exc:
            failure = classify_exception(exc)
            duration_ms = _duration_milliseconds(started_at, self._monotonic())
            self._store.fail_tool_execution(
                execution.id,
                run_id=run_id,
                step_id=step_id,
                owner_id=owner_id,
                failure_class=failure.failure_class,
                failure_message=failure.persisted_message,
                duration_ms=duration_ms,
            )
            raise ToolExecutionFailure(
                failure,
                classification,
                is_step_retryable(failure, classification),
            ) from exc

        duration_ms = _duration_milliseconds(started_at, self._monotonic())
        return ToolInvocationResult(
            execution_id=execution.id,
            tool_result=ToolResult(id=tool_call.id, content=content),
            result_preview=content[:_RESULT_PREVIEW_LENGTH],
            duration_ms=duration_ms,
            classification=classification,
            sanitized_arguments=sanitized_arguments,
        )

    def _ensure_allowed(self, tool_name: str, policy: RunPolicy) -> None:
        if tool_name == "paper_deep_read" and not policy.allow_subagents:
            self._raise_validation("paper_deep_read is disabled by run policy", "non_retryable")
        if tool_name.startswith("mcp__"):
            server = _mcp_server_name(tool_name)
            if server is None or server not in policy.allowed_mcp_servers:
                self._raise_validation(
                    f"MCP server is not allowed: {tool_name}",
                    self._registry.classify(tool_name),
                )
            return
        if tool_name not in policy.allowed_builtin_tools:
            self._raise_validation(
                f"builtin tool is not allowed: {tool_name}",
                self._registry.classify(tool_name),
            )

    @staticmethod
    def _raise_validation(message: str, classification: ToolClassification) -> None:
        failure = ClassifiedFailure("validation", message, transient=False)
        raise ToolExecutionFailure(failure, classification, retryable=False)


def sanitize_tool_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic descriptor safe to persist in SQLite and events."""
    return {str(key): _sanitize_value(value, key=str(key)) for key, value in arguments.items()}


def _sanitize_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and key.lower() in _SECRET_ARGUMENT_KEYS:
        return "[REDACTED]"
    if isinstance(value, str):
        if len(value) <= _LONG_TEXT_LIMIT:
            return value
        return {
            "char_count": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "preview": value[:_LONG_TEXT_PREVIEW_LENGTH],
        }
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def _mcp_server_name(tool_name: str) -> str | None:
    parts = tool_name.split("__", 2)
    if len(parts) != 3 or parts[0] != "mcp" or not parts[1] or not parts[2]:
        return None
    return parts[1]


def _stringify_tool_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def _duration_milliseconds(started_at: float, finished_at: float) -> int:
    return max(0, round((finished_at - started_at) * 1_000))
