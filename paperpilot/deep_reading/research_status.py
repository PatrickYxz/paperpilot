"""Attempt-local planning and status contracts for the Research Agent."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from threading import Lock
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from langchain.agents.middleware.types import (
    AgentState,
    ModelRequest,
    PrivateStateAttr,
    ToolCallRequest,
)
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field


_LOGGER = logging.getLogger(__name__)


TodoStatus = Literal["pending", "in_progress", "completed"]


class ResearchTodo(TypedDict):
    id: str
    content: str
    status: TodoStatus


class ResearchAgentState(AgentState[Any]):
    todos: Annotated[NotRequired[list[ResearchTodo]], PrivateStateAttr]


class ResearchTodoValidationError(ValueError):
    """Raised when a proposed TODO list violates the Research contract."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _reject(reason_code: str) -> ResearchTodoValidationError:
    return ResearchTodoValidationError(reason_code)


def validate_research_todo_update(
    *,
    current: Sequence[Mapping[str, object]],
    proposed: Sequence[Mapping[str, object]],
    high_water_mark: int,
) -> tuple[list[ResearchTodo], int]:
    """Validate one replacement update without mutating the current plan."""
    if not isinstance(high_water_mark, int) or high_water_mark < 0:
        raise _reject("invalid_high_water_mark")

    proposed_copy = [dict(item) for item in proposed]
    if not 1 <= len(proposed_copy) <= 6:
        raise _reject("empty_plan" if not proposed_copy else "too_many_items")

    normalized: list[ResearchTodo] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for item in proposed_copy:
        if set(item) != {"id", "content", "status"}:
            raise _reject("invalid_item_fields")
        item_id = item["id"]
        content = item["content"]
        status = item["status"]
        if not isinstance(item_id, str):
            raise _reject("invalid_id")
        match = re.fullmatch(r"todo_([1-9][0-9]*)", item_id)
        if match is None:
            raise _reject("invalid_id")
        if item_id in seen_ids:
            raise _reject("duplicate_id")
        seen_ids.add(item_id)
        if not isinstance(content, str):
            raise _reject("invalid_content")
        normalized_content = content.strip()
        if not 1 <= len(normalized_content) <= 160:
            raise _reject("invalid_content")
        if normalized_content in seen_content:
            raise _reject("duplicate_content")
        seen_content.add(normalized_content)
        if not isinstance(status, str) or status not in {
            "pending",
            "in_progress",
            "completed",
        }:
            raise _reject("invalid_status")
        normalized.append(
            {
                "id": item_id,
                "content": normalized_content,
                "status": status,  # type: ignore[typeddict-item]
            }
        )

    in_progress_count = sum(item["status"] == "in_progress" for item in normalized)
    if in_progress_count > 1:
        raise _reject("multiple_in_progress")
    if in_progress_count == 0 and any(item["status"] != "completed" for item in normalized):
        raise _reject("in_progress_required")

    current_by_id = {str(item.get("id")): item for item in current}
    proposed_by_id = {item["id"]: item for item in normalized}
    for item_id, old_item in current_by_id.items():
        old_status = old_item.get("status")
        old_content = old_item.get("content")
        new_item = proposed_by_id.get(item_id)
        if old_status == "completed":
            if (
                new_item is None
                or new_item["content"] != old_content
                or new_item["status"] != "completed"
            ):
                raise _reject("completed_item_immutable")
        elif old_status == "in_progress":
            if new_item is None or new_item["content"] != old_content:
                raise _reject("in_progress_item_immutable")
            if new_item["status"] not in {"in_progress", "completed"}:
                raise _reject("in_progress_transition")
        elif old_status == "pending":
            if new_item is not None:
                if new_item["content"] != old_content:
                    raise _reject("pending_item_immutable")
                if new_item["status"] == "completed":
                    raise _reject("pending_transition")
        else:
            raise _reject("invalid_current_plan")

    for item in normalized:
        if item["id"] not in current_by_id:
            number = int(item["id"].removeprefix("todo_"))
            if number <= high_water_mark:
                raise _reject("id_not_monotonic")

    max_accepted_id = max(
        (int(item["id"].removeprefix("todo_")) for item in normalized),
        default=0,
    )
    return normalized, max(high_water_mark, max_accepted_id)


class ResearchTodoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^todo_[1-9][0-9]*$")
    content: str = Field(min_length=1, max_length=160)
    status: TodoStatus


class WriteResearchTodosInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    todos: list[ResearchTodoInput] = Field(min_length=1, max_length=6)


class ResearchTodoMiddleware(AgentMiddleware[ResearchAgentState, Any, Any]):
    """Provide the Research-specific, validated ``write_todos`` tool."""

    state_schema = ResearchAgentState

    def __init__(self) -> None:
        super().__init__()
        self._high_water_mark = 0
        self.tools = [
            StructuredTool.from_function(
                name="write_todos",
                description=(
                    "Replace the Research TODO plan. The initial plan must contain "
                    "1-6 items with exactly one in_progress item and the rest "
                    "pending. After a successful update, proceed to the next business "
                    "tool. On later updates, preserve every existing id and content "
                    "exactly and change only status; a fully completed plan has all "
                    "items completed."
                ),
                func=self._write_todos,
                args_schema=WriteResearchTodosInput,
                infer_schema=False,
            )
        ]

    def _write_todos(
        self,
        *,
        runtime: ToolRuntime,
        todos: list[ResearchTodoInput],
    ) -> Command[Any] | ToolMessage:
        current = runtime.state.get("todos", [])
        try:
            proposed = [
                item.model_dump()
                if isinstance(item, ResearchTodoInput)
                else dict(item)
                for item in todos
            ]
            normalized, new_high_water_mark = validate_research_todo_update(
                current=current if isinstance(current, list) else [],
                proposed=proposed,
                high_water_mark=self._high_water_mark,
            )
        except ResearchTodoValidationError as exc:
            return ToolMessage(
                content=f"Research TODO update rejected: {exc.reason_code}.",
                tool_call_id=runtime.tool_call_id,
                name="write_todos",
                status="error",
            )

        self._high_water_mark = new_high_water_mark
        return Command(
            update={
                "todos": normalized,
                "messages": [
                    ToolMessage(
                        content="Research TODO list updated.",
                        tool_call_id=runtime.tool_call_id,
                        name="write_todos",
                    )
                ],
            }
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        response = handler(request)
        ai_messages = [item for item in response.result if isinstance(item, AIMessage)]
        tool_calls = ai_messages[-1].tool_calls if ai_messages else []
        todo_calls = [call for call in tool_calls if call.get("name") == "write_todos"]
        if len(todo_calls) > 1 or (
            todo_calls
            and (
                len(tool_calls) != len(todo_calls)
                or response.structured_response is not None
            )
        ):
            return _reject_model_tool_calls(
                response,
                tool_calls,
                reason="parallel_calls",
            )

        todos = request.state.get("todos", [])
        if response.structured_response is not None and todos:
            if any(item.get("status") != "completed" for item in todos):
                answered_call_ids = {
                    item.tool_call_id
                    for item in response.result
                    if isinstance(item, ToolMessage) and item.tool_call_id is not None
                }
                structured_calls = [
                    call for call in tool_calls if call.get("id") in answered_call_ids
                ]
                return _reject_model_tool_calls(
                    response,
                    structured_calls,
                    reason="incomplete_plan",
                )
        return response


def _reject_model_tool_calls(
    response: ModelResponse[Any],
    tool_calls: Sequence[Mapping[str, object]],
    *,
    reason: str,
) -> ModelResponse[Any]:
    rejected_ids = {
        str(call.get("id", "research-todo-validation")) for call in tool_calls
    }
    preserved = [
        item
        for item in response.result
        if not (
            isinstance(item, ToolMessage) and item.tool_call_id in rejected_ids
        )
    ]
    errors = [
        ToolMessage(
            content=f"Research TODO update rejected: {reason}.",
            tool_call_id=str(call.get("id", "research-todo-validation")),
            name=str(call.get("name", "write_todos")),
            status="error",
        )
        for call in tool_calls
    ]
    return ModelResponse(
        result=[*preserved, *errors],
        structured_response=None,
    )


BUSINESS_TOOL_NAMES = frozenset(
    {
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
        "read_artifact_slice",
        "search_artifact",
    }
)
STATUS_SCHEMA_VERSION = "paperpilot-agent-status-v1"
STATUS_SOURCE = "paperpilot_harness"


@dataclass(frozen=True)
class ResearchTodoSnapshot:
    id: str
    content: str
    status: TodoStatus


@dataclass(frozen=True)
class ResearchLedgerSnapshot:
    candidate_ids: tuple[str, ...]
    prepared_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResearchToolEvent:
    name: str
    outcome: Literal["success", "error"]
    occurred_at_utc: datetime
    error_type: str | None = None


@dataclass(frozen=True)
class ResearchAlert:
    code: Literal["budget_low", "repeated_tool_call", "no_progress"]
    action: str
    tool_name: str | None = None
    repeat_count: int | None = None


@dataclass(frozen=True)
class ResearchStatusSnapshot:
    attempt: int
    sequence: int
    mode: Literal["unplanned", "direct", "planned"]
    todos: tuple[ResearchTodoSnapshot, ...]
    model_calls_used: int
    model_call_limit: int
    research_tool_calls_used: int
    research_tool_call_limit: int
    ledger: ResearchLedgerSnapshot
    generated_at_utc: datetime
    elapsed_since_last_model_response_ms: int | None
    last_event: ResearchToolEvent | None
    alerts: tuple[ResearchAlert, ...]


@dataclass(frozen=True)
class ResearchToolObservationToken:
    name: str
    fingerprint: str
    progress_signature_before: str | None


def business_tool_fingerprint(name: str, arguments: Mapping[str, object]) -> str:
    payload = name + json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def research_progress_signature(
    ledger: ResearchLedgerSnapshot,
    todos: Sequence[Mapping[str, object]],
) -> str:
    payload = {
        "candidate_ids": sorted(ledger.candidate_ids),
        "prepared_ids": sorted(ledger.prepared_ids),
        "evidence_ids": sorted(ledger.evidence_ids),
        "todos": sorted(
            [
                {
                    "id": str(todo.get("id", "")),
                    "status": str(todo.get("status", "")),
                }
                for todo in todos
            ],
            key=lambda item: item["id"],
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


class ResearchExecutionTracker:
    """Track bounded, derived metadata for one structured Research attempt."""

    def __init__(
        self,
        *,
        attempt: int,
        ledger_snapshot: Callable[[], ResearchLedgerSnapshot],
        wall_clock: Callable[[], datetime],
        monotonic_clock: Callable[[], float],
    ) -> None:
        if attempt < 1:
            raise ValueError("attempt must be positive")
        self._attempt = attempt
        self._ledger_snapshot = ledger_snapshot
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._lock = Lock()
        self._sequence = 0
        self._fingerprint_counts: dict[str, int] = {}
        self._latest_repeat: tuple[str, int] | None = None
        self._last_event: ResearchToolEvent | None = None
        self._last_model_response_monotonic: float | None = None
        self._previous_progress_signature: str | None = None
        self._no_progress_streak = 0
        self._business_tool_call_count = 0

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    def record_model_response(self) -> None:
        with self._lock:
            self._last_model_response_monotonic = self._monotonic_clock()

    def record_business_tool_start(
        self,
        name: str,
        arguments: Mapping[str, object],
    ) -> ResearchToolObservationToken:
        if name not in BUSINESS_TOOL_NAMES:
            raise ValueError(f"unsupported business tool: {name}")
        fingerprint = business_tool_fingerprint(name, arguments)
        with self._lock:
            count = self._fingerprint_counts.get(fingerprint, 0) + 1
            self._fingerprint_counts[fingerprint] = count
            if count >= 2:
                self._latest_repeat = (name, count)
            return ResearchToolObservationToken(
                name=name,
                fingerprint=fingerprint,
                progress_signature_before=self._previous_progress_signature,
            )

    def record_business_tool_finish(
        self,
        token: ResearchToolObservationToken,
        todos: Sequence[ResearchTodo],
        outcome: Literal["success", "error"],
        error_type: str | None,
    ) -> None:
        if outcome not in {"success", "error"}:
            raise ValueError("invalid tool outcome")
        with self._lock:
            self._business_tool_call_count += 1
            try:
                after_signature = research_progress_signature(self._ledger_snapshot(), todos)
            except Exception:
                after_signature = None
            if token.progress_signature_before is not None and after_signature is not None:
                if token.progress_signature_before == after_signature:
                    self._no_progress_streak += 1
                else:
                    self._no_progress_streak = 0
                self._previous_progress_signature = after_signature
            event_time = self._wall_clock()
            self._last_event = ResearchToolEvent(
                name=token.name,
                outcome=outcome,
                occurred_at_utc=event_time,
                error_type=error_type if outcome == "error" else None,
            )

    def next_snapshot(
        self,
        todos: Sequence[ResearchTodo],
        model_calls_used: int,
        model_call_limit: int,
        research_tool_calls_used: int,
        research_tool_call_limit: int,
    ) -> ResearchStatusSnapshot:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            ledger = self._ledger_snapshot()
            now = self._wall_clock()
            elapsed: int | None = None
            if self._last_model_response_monotonic is not None:
                elapsed = max(
                    0,
                    int(
                        (self._monotonic_clock() - self._last_model_response_monotonic) * 1000
                    ),
                )
            todo_snapshots = tuple(
                ResearchTodoSnapshot(
                    id=str(todo["id"]),
                    content=str(todo["content"]),
                    status=todo["status"],  # type: ignore[arg-type]
                )
                for todo in todos
            )
            self._previous_progress_signature = research_progress_signature(ledger, todos)
            mode: Literal["unplanned", "direct", "planned"]
            if todo_snapshots:
                mode = "planned"
            elif self._business_tool_call_count:
                mode = "direct"
            else:
                mode = "unplanned"
            alerts: list[ResearchAlert] = []
            if (
                model_call_limit - model_calls_used <= 1
                or research_tool_call_limit - research_tool_calls_used <= 1
            ):
                alerts.append(
                    ResearchAlert(
                        code="budget_low",
                        action="finish_required_work_or_return_limitations",
                    )
                )
            if self._latest_repeat is not None:
                tool_name, repeat_count = self._latest_repeat
                alerts.append(
                    ResearchAlert(
                        code="repeated_tool_call",
                        action="change_query_or_evidence_target_or_stop_branch",
                        tool_name=tool_name,
                        repeat_count=repeat_count,
                    )
                )
            if self._no_progress_streak >= 2:
                alerts.append(
                    ResearchAlert(
                        code="no_progress",
                        action="select_uncovered_required_point_and_change_action",
                    )
                )
            return ResearchStatusSnapshot(
                attempt=self._attempt,
                sequence=sequence,
                mode=mode,
                todos=todo_snapshots,
                model_calls_used=model_calls_used,
                model_call_limit=model_call_limit,
                research_tool_calls_used=research_tool_calls_used,
                research_tool_call_limit=research_tool_call_limit,
                ledger=ledger,
                generated_at_utc=now,
                elapsed_since_last_model_response_ms=elapsed,
                last_event=self._last_event,
                alerts=tuple(alerts),
            )


_ALERT_ACTIONS = {
    "budget_low": "finish_required_work_or_return_limitations",
    "repeated_tool_call": "change_query_or_evidence_target_or_stop_branch",
    "no_progress": "select_uncovered_required_point_and_change_action",
}


def _utc_millis(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    utc_value = value.astimezone(timezone.utc)
    return utc_value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def render_research_status(snapshot: ResearchStatusSnapshot) -> str:
    """Render one immutable snapshot into deterministic, bounded XML."""
    from xml.etree.ElementTree import Element, SubElement, tostring

    root = Element(
        "agent_status_bar",
        {
            "source": STATUS_SOURCE,
            "schema_version": STATUS_SCHEMA_VERSION,
            "attempt": str(snapshot.attempt),
            "sequence": str(snapshot.sequence),
        },
    )
    task_progress = SubElement(root, "task_progress", {"mode": snapshot.mode})
    for todo in snapshot.todos:
        item = SubElement(
            task_progress,
            "todo",
            {"id": todo.id, "status": todo.status},
        )
        item.text = todo.content

    execution_state = SubElement(root, "execution_state")
    SubElement(execution_state, "stage").text = "research"
    SubElement(
        execution_state,
        "model_calls",
        {
            "used": str(snapshot.model_calls_used),
            "limit": str(snapshot.model_call_limit),
            "remaining": str(max(0, snapshot.model_call_limit - snapshot.model_calls_used)),
        },
    )
    SubElement(
        execution_state,
        "research_tool_calls",
        {
            "used": str(snapshot.research_tool_calls_used),
            "limit": str(snapshot.research_tool_call_limit),
            "remaining": str(
                max(0, snapshot.research_tool_call_limit - snapshot.research_tool_calls_used)
            ),
        },
    )
    SubElement(execution_state, "candidate_papers", {"count": str(len(snapshot.ledger.candidate_ids))})
    SubElement(execution_state, "prepared_papers", {"count": str(len(snapshot.ledger.prepared_ids))})
    SubElement(execution_state, "evidence_items", {"count": str(len(snapshot.ledger.evidence_ids))})

    side_channel = SubElement(root, "side_channel")
    SubElement(side_channel, "generated_at_utc").text = _utc_millis(snapshot.generated_at_utc)
    if snapshot.elapsed_since_last_model_response_ms is None:
        SubElement(side_channel, "elapsed_since_last_model_response_ms", {"available": "false"})
    else:
        SubElement(side_channel, "elapsed_since_last_model_response_ms").text = str(
            max(0, snapshot.elapsed_since_last_model_response_ms)
        )
    if snapshot.last_event is None:
        SubElement(side_channel, "last_event", {"available": "false"})
    else:
        event_attributes = {
            "type": "tool_" + "result",
            "name": snapshot.last_event.name,
            "outcome": snapshot.last_event.outcome,
        }
        if snapshot.last_event.error_type is not None:
            event_attributes["error_type"] = snapshot.last_event.error_type
        event_attributes["occurred_at_utc"] = _utc_millis(snapshot.last_event.occurred_at_utc)
        SubElement(side_channel, "last_event", event_attributes)

    alerts_element = SubElement(root, "alerts")
    for alert in sorted(
        snapshot.alerts,
        key=lambda item: {"budget_low": 0, "repeated_tool_call": 1, "no_progress": 2}[item.code],
    ):
        attributes = {
            "code": alert.code,
            "severity": "warning",
            "action": _ALERT_ACTIONS[alert.code],
        }
        if alert.tool_name is not None:
            attributes["tool_name"] = alert.tool_name
        if alert.repeat_count is not None:
            attributes["repeat_count"] = str(alert.repeat_count)
        SubElement(alerts_element, "alert", attributes)
    return tostring(root, encoding="unicode", short_empty_elements=True)


class ResearchToolBudgetMiddleware(AgentMiddleware[ResearchAgentState, Any, Any]):
    """Reserve only Research's three business-tool calls."""

    def __init__(self, *, run_limit: int) -> None:
        super().__init__()
        if run_limit < 1:
            raise ValueError("run_limit must be positive")
        self.run_limit = run_limit
        self._used = 0
        self._lock = Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        response = handler(request)
        ai_message = next(
            (item for item in reversed(response.result) if isinstance(item, AIMessage)),
            None,
        )
        if ai_message is None:
            return response
        completed_tool_call_ids = {
            item.tool_call_id
            for item in response.result
            if isinstance(item, ToolMessage) and item.tool_call_id is not None
        }
        pending_calls = [
            call
            for call in ai_message.tool_calls
            if call.get("name") in BUSINESS_TOOL_NAMES
            and call.get("id") not in completed_tool_call_ids
        ]
        requested = len(pending_calls)
        if requested == 0:
            return response
        with self._lock:
            projected = self._used + requested
            if projected > self.run_limit:
                raise ToolCallLimitExceededError(
                    thread_count=0,
                    run_count=projected,
                    thread_limit=None,
                    run_limit=self.run_limit,
                    tool_name="research_business_tools",
                )
            self._used = projected
        return response


STATUS_UNAVAILABLE_XML = (
    '<agent_status_bar source="paperpilot_harness" '
    'schema_version="paperpilot-agent-status-v1" status="unavailable">'
    "<alerts><alert code=\"status_unavailable\" severity=\"warning\" "
    'action="continue_with_visible_messages_and_existing_limits" /></alerts>'
    "</agent_status_bar>"
)


class ResearchStatusMiddleware(AgentMiddleware[ResearchAgentState, Any, Any]):
    """Inject and observe one best-effort status bar per logical model call."""

    def __init__(
        self,
        *,
        tracker: ResearchExecutionTracker,
        model_call_limit: int,
        tool_budget: ResearchToolBudgetMiddleware,
        renderer: Callable[[ResearchStatusSnapshot], str] = render_research_status,
    ) -> None:
        super().__init__()
        self.tracker = tracker
        self.model_call_limit = model_call_limit
        self.tool_budget = tool_budget
        self.renderer = renderer

    def _status_message(
        self,
        request: ModelRequest[Any],
    ) -> Any | None:
        sequence: int
        try:
            raw_todos = request.state.get("todos", [])
            todos = raw_todos if isinstance(raw_todos, list) else []
            model_calls_used = request.state.get("run_model_call_count", 0)
            snapshot = self.tracker.next_snapshot(
                todos,
                int(model_calls_used),
                self.model_call_limit,
                self.tool_budget.used,
                self.tool_budget.run_limit,
            )
            sequence = snapshot.sequence
            content = self.renderer(snapshot)
            if not isinstance(content, str):
                raise TypeError("renderer must return text")
        except Exception as exc:
            sequence = self.tracker.sequence
            _LOGGER.warning(
                "research status rendering unavailable",
                extra={
                    "event": "research.status_render_failed",
                    "attempt": self.tracker.attempt,
                    "sequence": sequence,
                    "error_type": type(exc).__name__,
                },
            )
            content = STATUS_UNAVAILABLE_XML

        try:
            from langchain.messages import HumanMessage

            return HumanMessage(
                content=content,
                id=f"paperpilot-status-attempt-{self.tracker.attempt}-sequence-{sequence}",
                additional_kwargs={"paperpilot_source": "agent_status_bar"},
            )
        except Exception as exc:
            _LOGGER.warning(
                "research status message unavailable",
                extra={
                    "event": "research.status_message_failed",
                    "attempt": self.tracker.attempt,
                    "sequence": sequence,
                    "error_type": type(exc).__name__,
                },
            )
            return None

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        status_message = self._status_message(request)
        status_request = request
        if status_message is not None:
            status_request = request.override(messages=[*request.messages, status_message])
        response = handler(status_request)
        try:
            self.tracker.record_model_response()
        except Exception as exc:
            _LOGGER.warning(
                "research status response observation unavailable",
                extra={
                    "event": "research.status_response_observation_failed",
                    "attempt": self.tracker.attempt,
                    "sequence": self.tracker.sequence,
                    "error_type": type(exc).__name__,
                },
            )
        if status_message is None:
            return response
        return ModelResponse(
            result=[status_message, *response.result],
            structured_response=response.structured_response,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call.get("name")
        if name not in BUSINESS_TOOL_NAMES:
            return handler(request)
        arguments = request.tool_call.get("args", {})
        if not isinstance(arguments, Mapping):
            arguments = {}
        try:
            token = self.tracker.record_business_tool_start(name, arguments)
        except Exception as exc:
            _LOGGER.warning(
                "research status tool-start observation unavailable",
                extra={
                    "event": "research.status_tool_start_failed",
                    "attempt": self.tracker.attempt,
                    "sequence": self.tracker.sequence,
                    "error_type": type(exc).__name__,
                },
            )
            return handler(request)

        try:
            result = handler(request)
        except Exception as exc:
            try:
                todos = request.state.get("todos", [])
                self.tracker.record_business_tool_finish(
                    token,
                    todos if isinstance(todos, list) else [],
                    "error",
                    type(exc).__name__,
                )
            except Exception as observation_error:
                _LOGGER.warning(
                    "research status tool-error observation unavailable",
                    extra={
                        "event": "research.status_tool_finish_failed",
                        "attempt": self.tracker.attempt,
                        "sequence": self.tracker.sequence,
                        "error_type": type(observation_error).__name__,
                    },
                )
            raise

        try:
            todos = request.state.get("todos", [])
            self.tracker.record_business_tool_finish(
                token,
                todos if isinstance(todos, list) else [],
                "error" if isinstance(result, ToolMessage) and result.status == "error" else "success",
                None,
            )
        except Exception as exc:
            _LOGGER.warning(
                "research status tool observation unavailable",
                extra={
                    "event": "research.status_tool_finish_failed",
                    "attempt": self.tracker.attempt,
                    "sequence": self.tracker.sequence,
                    "error_type": type(exc).__name__,
                },
            )
        return result
