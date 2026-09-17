"""Tests for the Research Agent's attempt-local status contracts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from xml.etree import ElementTree

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelResponse
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import ToolRuntime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command
from pydantic import PrivateAttr

from paperpilot.deep_reading.research_status import (
    ResearchAlert,
    ResearchExecutionTracker,
    ResearchLedgerSnapshot,
    ResearchStatusSnapshot,
    ResearchStatusMiddleware,
    ResearchToolBudgetMiddleware,
    ResearchTodoSnapshot,
    ResearchToolEvent,
    ResearchTodoMiddleware,
    ResearchTodoValidationError,
    STATUS_UNAVAILABLE_XML,
    business_tool_fingerprint,
    render_research_status,
    research_progress_signature,
    validate_research_todo_update,
)


def _runtime(state: Mapping[str, Any], tool_call_id: str = "todo-call-1") -> ToolRuntime:
    return ToolRuntime(
        state=dict(state),
        context=None,
        config={},
        stream_writer=lambda _chunk: None,
        tool_call_id=tool_call_id,
        store=None,
    )


def _write_todos(middleware: ResearchTodoMiddleware) -> Any:
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "write_todos"
    return middleware.tools[0].func


def _model_request(
    *,
    state: Mapping[str, Any] | None = None,
) -> ModelRequest[Any]:
    return ModelRequest(
        model=object(),
        messages=[],
        state=dict(state or {"messages": []}),
    )


def _model_response(
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    structured_response: object | None = None,
) -> ModelResponse[Any]:
    return ModelResponse(
        result=[AIMessage(content="", tool_calls=tool_calls or [])],
        structured_response=structured_response,
    )


def test_todo_update_normalizes_content_and_advances_high_water_mark() -> None:
    todos, high_water_mark = validate_research_todo_update(
        current=[],
        proposed=[
            {"id": "todo_1", "content": "  检索方法定义  ", "status": "in_progress"},
            {"id": "todo_2", "content": "核对实验结果", "status": "pending"},
        ],
        high_water_mark=0,
    )

    assert todos == [
        {"id": "todo_1", "content": "检索方法定义", "status": "in_progress"},
        {"id": "todo_2", "content": "核对实验结果", "status": "pending"},
    ]
    assert high_water_mark == 2


def test_todo_update_completes_current_item_and_starts_next_atomically() -> None:
    current = [
        {"id": "todo_1", "content": "检索方法定义", "status": "in_progress"},
        {"id": "todo_2", "content": "核对实验结果", "status": "pending"},
    ]

    todos, high_water_mark = validate_research_todo_update(
        current=current,
        proposed=[
            {"id": "todo_1", "content": "检索方法定义", "status": "completed"},
            {"id": "todo_2", "content": "核对实验结果", "status": "in_progress"},
        ],
        high_water_mark=2,
    )

    assert [item["status"] for item in todos] == ["completed", "in_progress"]
    assert high_water_mark == 2


@pytest.mark.parametrize(
    "proposed",
    [
        [],
        [{"id": "todo_0", "content": "invalid", "status": "in_progress"}],
        [{"id": "todo_01", "content": "invalid", "status": "in_progress"}],
        [
            {"id": "todo_1", "content": "same", "status": "in_progress"},
            {"id": "todo_2", "content": " same ", "status": "pending"},
        ],
        [
            {"id": "todo_1", "content": "one", "status": "in_progress"},
            {"id": "todo_2", "content": "two", "status": "in_progress"},
        ],
        [{"id": "todo_1", "content": "", "status": "in_progress"}],
        [{"id": "todo_1", "content": "x" * 161, "status": "in_progress"}],
        [{"id": "todo_1", "content": "invalid", "status": "unknown"}],
        [{"id": "todo_1", "content": "invalid", "status": []}],
        [{"id": "todo_1", "content": "invalid", "status": {}}],
        [{"id": "todo_1", "content": "invalid", "status": "in_progress", "extra": 1}],
    ],
)
def test_todo_update_rejects_invalid_plan_shapes(
    proposed: list[dict[str, object]],
) -> None:
    with pytest.raises(ResearchTodoValidationError):
        validate_research_todo_update(
            current=[],
            proposed=proposed,
            high_water_mark=0,
        )


@pytest.mark.parametrize(
    ("current", "proposed"),
    [
        (
            [{"id": "todo_1", "content": "done", "status": "completed"}],
            [],
        ),
        (
            [{"id": "todo_1", "content": "active", "status": "in_progress"}],
            [],
        ),
        (
            [{"id": "todo_1", "content": "active", "status": "in_progress"}],
            [{"id": "todo_1", "content": "renamed", "status": "in_progress"}],
        ),
        (
            [{"id": "todo_1", "content": "pending", "status": "pending"}],
            [{"id": "todo_1", "content": "pending", "status": "completed"}],
        ),
    ],
)
def test_todo_update_rejects_illegal_existing_item_transitions(
    current: list[dict[str, str]],
    proposed: list[dict[str, str]],
) -> None:
    with pytest.raises(ResearchTodoValidationError):
        validate_research_todo_update(
            current=current,
            proposed=proposed,
            high_water_mark=1,
        )


def test_todo_update_rejects_reusing_deleted_pending_id() -> None:
    current = [{"id": "todo_1", "content": "old", "status": "pending"}]
    todos, high_water_mark = validate_research_todo_update(
        current=current,
        proposed=[{"id": "todo_2", "content": "new", "status": "in_progress"}],
        high_water_mark=1,
    )
    assert todos[0]["id"] == "todo_2"
    assert high_water_mark == 2

    with pytest.raises(ResearchTodoValidationError):
        validate_research_todo_update(
            current=todos,
            proposed=[{"id": "todo_1", "content": "reused", "status": "pending"}],
            high_water_mark=high_water_mark,
        )


def test_write_todos_tool_returns_command_without_echoing_todo_text() -> None:
    middleware = ResearchTodoMiddleware()
    result = _write_todos(middleware)(
        runtime=_runtime({"messages": []}),
        todos=[{"id": "todo_1", "content": "检索方法", "status": "in_progress"}],
    )

    assert isinstance(result, Command)
    assert result.update["todos"] == [
        {"id": "todo_1", "content": "检索方法", "status": "in_progress"}
    ]
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.content == "Research TODO list updated."
    assert "检索方法" not in str(message.content)


def test_write_todos_invalid_update_returns_error_without_state_update() -> None:
    middleware = ResearchTodoMiddleware()
    write_todos = _write_todos(middleware)

    first = write_todos(
        runtime=_runtime({"messages": []}, "todo-call-1"),
        todos=[
            {"id": "todo_1", "content": "第一项", "status": "in_progress"},
            {"id": "todo_2", "content": "第二项", "status": "pending"},
        ],
    )
    assert isinstance(first, Command)

    invalid = write_todos(
        runtime=_runtime(first.update, "todo-call-2"),
        todos=[
            {"id": "todo_1", "content": "第一项", "status": "completed"},
            {"id": "todo_2", "content": "第二项", "status": "pending"},
        ],
    )
    assert isinstance(invalid, ToolMessage)
    assert invalid.status == "error"
    assert invalid.content == "Research TODO update rejected: in_progress_required."


def test_write_todos_cannot_reuse_deleted_pending_id() -> None:
    middleware = ResearchTodoMiddleware()
    write_todos = _write_todos(middleware)
    first = write_todos(
        runtime=_runtime({"messages": []}, "todo-call-1"),
        todos=[
            {"id": "todo_1", "content": "当前项", "status": "in_progress"},
            {"id": "todo_2", "content": "可删除项", "status": "pending"},
        ],
    )
    assert isinstance(first, Command)
    second = write_todos(
        runtime=_runtime(first.update, "todo-call-2"),
        todos=[{"id": "todo_1", "content": "当前项", "status": "in_progress"}],
    )
    assert isinstance(second, Command)
    reused = write_todos(
        runtime=_runtime(second.update, "todo-call-3"),
        todos=[
            {"id": "todo_1", "content": "当前项", "status": "in_progress"},
            {"id": "todo_2", "content": "复用", "status": "pending"},
        ],
    )
    assert isinstance(reused, ToolMessage)
    assert reused.status == "error"
    assert reused.content == "Research TODO update rejected: id_not_monotonic."


def test_todo_middleware_allows_one_write_todos_call() -> None:
    middleware = ResearchTodoMiddleware()
    response = _model_response(
        tool_calls=[
            {
                "name": "write_todos",
                "args": {
                    "todos": [
                        {"id": "todo_1", "content": "计划", "status": "in_progress"}
                    ]
                },
                "id": "call-1",
                "type": "tool_call",
            }
        ]
    )

    result = middleware.wrap_model_call(
        _model_request(),
        lambda _request: response,
    )

    assert result is response


def test_todo_middleware_rejects_parallel_write_todos_and_business_tools() -> None:
    middleware = ResearchTodoMiddleware()
    response = _model_response(
        tool_calls=[
            {
                "name": "write_todos",
                "args": {"todos": []},
                "id": "call-1",
                "type": "tool_call",
            },
            {
                "name": "search_related_papers",
                "args": {"query": "x", "limit": 3},
                "id": "call-2",
                "type": "tool_call",
            },
        ]
    )

    result = middleware.wrap_model_call(
        _model_request(),
        lambda _request: response,
    )

    assert result.structured_response is None
    assert len(result.result) == 3
    assert all(isinstance(item, ToolMessage) for item in result.result[1:])
    assert all(item.status == "error" for item in result.result[1:])


def test_todo_middleware_rejects_write_todos_with_structured_result() -> None:
    middleware = ResearchTodoMiddleware()
    todo_call = {
        "name": "write_todos",
        "args": {
            "todos": [
                {"id": "todo_1", "content": "计划", "status": "in_progress"}
            ]
        },
        "id": "todo-call",
        "type": "tool_call",
    }
    final_call = {
        "name": "AgentResearchDecision",
        "args": {"decision": "done"},
        "id": "final-call",
        "type": "tool_call",
    }
    response = ModelResponse(
        result=[
            AIMessage(content="", tool_calls=[todo_call, final_call]),
            ToolMessage(
                content="Returning structured response",
                tool_call_id="final-call",
                name="AgentResearchDecision",
            ),
        ],
        structured_response={"decision": "done"},
    )

    result = middleware.wrap_model_call(_model_request(), lambda _request: response)

    assert result.structured_response is None
    tool_messages = [item for item in result.result if isinstance(item, ToolMessage)]
    assert [
        (item.tool_call_id, item.name, item.status)
        for item in tool_messages
    ] == [
        ("todo-call", "write_todos", "error"),
        ("final-call", "AgentResearchDecision", "error"),
    ]


def test_todo_middleware_rejects_structured_result_until_all_todos_completed() -> None:
    middleware = ResearchTodoMiddleware()
    request = _model_request(
        state={
            "messages": [],
            "todos": [
                {"id": "todo_1", "content": "计划", "status": "in_progress"}
            ],
        }
    )
    final_call = {
        "name": "AgentResearchDecision",
        "args": {"decision": "not yet"},
        "id": "final-call",
        "type": "tool_call",
    }
    response = ModelResponse(
        result=[
            AIMessage(content="", tool_calls=[final_call]),
            ToolMessage(
                content="Returning structured response",
                tool_call_id="final-call",
                name="AgentResearchDecision",
            ),
        ],
        structured_response={"decision": "not yet"},
    )

    result = middleware.wrap_model_call(request, lambda _request: response)

    assert result.structured_response is None
    tool_messages = [item for item in result.result if isinstance(item, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "final-call"
    assert tool_messages[0].name == "AgentResearchDecision"
    assert tool_messages[0].status == "error"
    assert tool_messages[0].content == "Research TODO update rejected: incomplete_plan."


def test_todo_middleware_allows_structured_result_without_open_todos() -> None:
    middleware = ResearchTodoMiddleware()
    response = _model_response(structured_response={"decision": "done"})
    result = middleware.wrap_model_call(
        _model_request(state={"messages": [], "todos": []}),
        lambda _request: response,
    )
    assert result is response


def _ledger(
    *,
    candidates: tuple[str, ...] = (),
    prepared: tuple[str, ...] = (),
    evidence: tuple[str, ...] = (),
) -> ResearchLedgerSnapshot:
    return ResearchLedgerSnapshot(
        candidate_ids=candidates,
        prepared_ids=prepared,
        evidence_ids=evidence,
    )


def _todo_snapshot(
    *,
    item_id: str = "todo_1",
    content: str = "检索方法",
    status: str = "in_progress",
) -> ResearchTodoSnapshot:
    return ResearchTodoSnapshot(id=item_id, content=content, status=status)  # type: ignore[arg-type]


def _status_snapshot(
    *,
    mode: str = "planned",
    todos: tuple[ResearchTodoSnapshot, ...] = (),
    alerts: tuple[ResearchAlert, ...] = (),
    last_event: ResearchToolEvent | None = None,
    elapsed: int | None = None,
) -> ResearchStatusSnapshot:
    return ResearchStatusSnapshot(
        attempt=1,
        sequence=1,
        mode=mode,  # type: ignore[arg-type]
        todos=todos,
        model_calls_used=2,
        model_call_limit=6,
        research_tool_calls_used=3,
        research_tool_call_limit=6,
        ledger=_ledger(candidates=("paper-1",), prepared=("paper-1",), evidence=("ev-1",)),
        generated_at_utc=datetime(2026, 8, 23, 10, 20, 31, 482000, tzinfo=timezone.utc),
        elapsed_since_last_model_response_ms=elapsed,
        last_event=last_event,
        alerts=alerts,
    )


def test_business_tool_fingerprint_is_key_order_independent() -> None:
    first = business_tool_fingerprint(
        "search_related_papers", {"query": "x", "limit": 3}
    )
    second = business_tool_fingerprint(
        "search_related_papers", {"limit": 3, "query": "x"}
    )
    changed = business_tool_fingerprint(
        "search_related_papers", {"query": "y", "limit": 3}
    )

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_progress_signature_is_stable_for_order_and_changes_for_progress() -> None:
    first = research_progress_signature(
        _ledger(
            candidates=("paper-2", "paper-1"),
            prepared=("paper-1",),
            evidence=("ev-2", "ev-1"),
        ),
        [
            {"id": "todo_2", "content": "B", "status": "pending"},
            {"id": "todo_1", "content": "A", "status": "in_progress"},
        ],
    )
    same = research_progress_signature(
        _ledger(
            candidates=("paper-1", "paper-2"),
            prepared=("paper-1",),
            evidence=("ev-1", "ev-2"),
        ),
        [
            {"id": "todo_1", "content": "A", "status": "in_progress"},
            {"id": "todo_2", "content": "B", "status": "pending"},
        ],
    )
    changed = research_progress_signature(
        _ledger(candidates=("paper-1", "paper-2"), prepared=("paper-1",)),
        [
            {"id": "todo_1", "content": "A", "status": "completed"},
            {"id": "todo_2", "content": "B", "status": "pending"},
        ],
    )

    assert first == same
    assert first != changed


def test_tracker_starts_sequence_and_records_elapsed_model_time() -> None:
    wall_time = datetime(2026, 8, 23, 10, 20, tzinfo=timezone.utc)
    monotonic = iter([10.0, 11.25])
    tracker = ResearchExecutionTracker(
        attempt=1,
        ledger_snapshot=lambda: _ledger(),
        wall_clock=lambda: wall_time,
        monotonic_clock=lambda: next(monotonic),
    )

    first = tracker.next_snapshot([], 0, 6, 0, 6)
    assert first.sequence == 1
    assert first.elapsed_since_last_model_response_ms is None
    assert first.last_event is None

    tracker.record_model_response()
    second = tracker.next_snapshot([], 1, 6, 0, 6)
    assert second.sequence == 2
    assert second.elapsed_since_last_model_response_ms == 1250


def test_tracker_emits_repeated_and_no_progress_alerts_and_resets_on_progress() -> None:
    tracker = ResearchExecutionTracker(
        attempt=1,
        ledger_snapshot=lambda: _ledger(),
        wall_clock=lambda: datetime(2026, 8, 23, 10, 20, tzinfo=timezone.utc),
        monotonic_clock=lambda: 1.0,
    )
    todos = [{"id": "todo_1", "content": "A", "status": "in_progress"}]

    tracker.next_snapshot(todos, 0, 6, 0, 6)
    first_token = tracker.record_business_tool_start("prepare_paper", {"id": "p-1"})
    tracker.record_business_tool_finish(first_token, todos, "success", None)
    tracker.next_snapshot(todos, 1, 6, 1, 6)
    second_token = tracker.record_business_tool_start("prepare_paper", {"id": "p-1"})
    tracker.record_business_tool_finish(second_token, todos, "success", None)
    repeated_snapshot = tracker.next_snapshot(todos, 2, 6, 2, 6)

    assert [alert.code for alert in repeated_snapshot.alerts] == [
        "repeated_tool_call",
        "no_progress",
    ]
    assert repeated_snapshot.alerts[0].tool_name == "prepare_paper"
    assert repeated_snapshot.alerts[0].repeat_count == 2

    ledger = {"candidates": ("p-1",)}
    progress_tracker = ResearchExecutionTracker(
        attempt=1,
        ledger_snapshot=lambda: _ledger(candidates=ledger["candidates"]),
        wall_clock=lambda: datetime(2026, 8, 23, 10, 20, tzinfo=timezone.utc),
        monotonic_clock=lambda: 1.0,
    )
    progress_tracker.next_snapshot(todos, 0, 6, 0, 6)
    progress_token = progress_tracker.record_business_tool_start("search_related_papers", {})
    ledger["candidates"] = ("p-1", "p-2")
    progress_tracker.record_business_tool_finish(progress_token, todos, "success", None)
    progress_snapshot = progress_tracker.next_snapshot(todos, 1, 6, 1, 6)
    assert all(alert.code != "no_progress" for alert in progress_snapshot.alerts)


def test_tracker_emits_budget_low_and_attempt_state_isolated() -> None:
    ledger = {"candidates": ("p-1",)}

    def snapshot() -> ResearchLedgerSnapshot:
        return _ledger(candidates=ledger["candidates"])

    tracker = ResearchExecutionTracker(
        attempt=1,
        ledger_snapshot=snapshot,
        wall_clock=lambda: datetime(2026, 8, 23, 10, 20, tzinfo=timezone.utc),
        monotonic_clock=lambda: 1.0,
    )
    first = tracker.next_snapshot([], 5, 6, 5, 6)
    assert [alert.code for alert in first.alerts] == ["budget_low"]

    tracker.next_snapshot([], 0, 6, 0, 6)
    token = tracker.record_business_tool_start("prepare_paper", {"id": "p-1"})
    tracker.record_business_tool_finish(token, [], "success", None)

    fresh = ResearchExecutionTracker(
        attempt=2,
        ledger_snapshot=snapshot,
        wall_clock=lambda: datetime(2026, 8, 23, 10, 20, tzinfo=timezone.utc),
        monotonic_clock=lambda: 1.0,
    )
    second = fresh.next_snapshot([], 0, 6, 0, 6)
    assert second.attempt == 2
    assert second.sequence == 1
    assert second.mode == "unplanned"
    assert second.ledger.candidate_ids == ("p-1",)
    assert second.last_event is None
    assert second.alerts == ()


def test_renderer_escapes_todo_text_and_uses_fixed_root_contract() -> None:
    xml = render_research_status(
        _status_snapshot(todos=(_todo_snapshot(content='method <A> & "B"'),))
    )
    root = ElementTree.fromstring(xml)

    assert root.tag == "agent_status_bar"
    assert root.attrib == {
        "source": "paperpilot_harness",
        "schema_version": "paperpilot-agent-status-v1",
        "attempt": "1",
        "sequence": "1",
    }
    assert root.findtext("task_progress/todo") == 'method <A> & "B"'
    assert "&lt;A&gt;" in xml
    assert "&amp;" in xml


@pytest.mark.parametrize("mode", ["unplanned", "direct", "planned"])
def test_renderer_is_deterministic_and_supports_all_task_modes(mode: str) -> None:
    snapshot = _status_snapshot(mode=mode)
    assert render_research_status(snapshot) == render_research_status(snapshot)
    assert ElementTree.fromstring(render_research_status(snapshot)).find(
        "task_progress"
    ).attrib["mode"] == mode


def test_renderer_uses_unavailable_side_channel_and_fixed_alert_order() -> None:
    event = ResearchToolEvent(
        name="retrieve_paper_evidence",
        outcome="error",
        occurred_at_utc=datetime(2026, 8, 23, 10, 20, 31, 470000, tzinfo=timezone.utc),
        error_type="ResearchContractError",
    )
    snapshot = _status_snapshot(
        elapsed=None,
        last_event=event,
        alerts=(
            ResearchAlert(
                code="no_progress",
                action="select_uncovered_required_point_and_change_action",
            ),
            ResearchAlert(
                code="budget_low",
                action="finish_required_work_or_return_limitations",
            ),
            ResearchAlert(
                code="repeated_tool_call",
                action="change_query_or_evidence_target_or_stop_branch",
                tool_name="prepare_paper",
                repeat_count=2,
            ),
        ),
    )
    xml = render_research_status(snapshot)
    root = ElementTree.fromstring(xml)

    assert root.find("side_channel/elapsed_since_last_model_response_ms").attrib == {
        "available": "false"
    }
    rendered_event = root.find("side_channel/last_event")
    assert rendered_event.attrib == {
        "type": "tool_result",
        "name": "retrieve_paper_evidence",
        "outcome": "error",
        "error_type": "ResearchContractError",
        "occurred_at_utc": "2026-08-23T10:20:31.470Z",
    }
    assert [item.attrib["code"] for item in root.findall("alerts/alert")] == [
        "budget_low",
        "repeated_tool_call",
        "no_progress",
    ]
    assert 'query="' not in xml
    assert "tool_args" not in xml
    assert "question_text" not in xml
    assert "paper-1" not in xml


def _tracker_for_middleware() -> ResearchExecutionTracker:
    return ResearchExecutionTracker(
        attempt=1,
        ledger_snapshot=lambda: _ledger(),
        wall_clock=lambda: datetime(2026, 8, 23, 10, 20, tzinfo=timezone.utc),
        monotonic_clock=lambda: 1.0,
    )


def _business_call(name: str, call_id: str = "call-1") -> dict[str, Any]:
    return {
        "name": name,
        "args": {"value": "x"},
        "id": call_id,
        "type": "tool_call",
    }


def test_business_budget_reserves_only_pending_business_calls() -> None:
    budget = ResearchToolBudgetMiddleware(run_limit=6)

    budget.wrap_model_call(
        _model_request(),
        lambda _request: _model_response(
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": []},
                    "id": "todo-call",
                    "type": "tool_call",
                }
            ]
        ),
    )
    assert budget.used == 0

    budget.wrap_model_call(
        _model_request(),
        lambda _request: _model_response(
            tool_calls=[
                _business_call("search_related_papers", "search-1"),
                _business_call("prepare_paper", "prepare-1"),
                _business_call("retrieve_paper_evidence", "retrieve-1"),
            ]
        ),
    )
    assert budget.used == 3

    budget.wrap_model_call(
        _model_request(),
        lambda _request: ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[_business_call("prepare_paper", "prepare-2")],
                ),
                ToolMessage(content="error", tool_call_id="prepare-2", status="error"),
            ]
        ),
    )
    assert budget.used == 3


def test_business_budget_rejects_an_over_limit_batch_atomically() -> None:
    budget = ResearchToolBudgetMiddleware(run_limit=1)
    response = _model_response(
        tool_calls=[
            _business_call("prepare_paper", "prepare-1"),
            _business_call("retrieve_paper_evidence", "retrieve-1"),
        ]
    )

    with pytest.raises(Exception, match="research_business_tools"):
        budget.wrap_model_call(_model_request(), lambda _request: response)

    assert budget.used == 0


def test_status_wrapper_appends_tail_status_and_preserves_previous_prefix() -> None:
    budget = ResearchToolBudgetMiddleware(run_limit=6)
    status = ResearchStatusMiddleware(
        tracker=_tracker_for_middleware(),
        model_call_limit=6,
        tool_budget=budget,
    )
    provider_messages: list[list[BaseMessage]] = []

    def provider(request: ModelRequest[Any]) -> ModelResponse[Any]:
        provider_messages.append(list(request.messages))
        return _model_response()

    first_request = _model_request(state={"messages": [], "run_model_call_count": 0})
    first = status.wrap_model_call(first_request, provider)
    first_provider_messages = provider_messages[0]
    assert isinstance(first_provider_messages[-1], HumanMessage)
    assert first_provider_messages[-1].additional_kwargs == {
        "paperpilot_source": "agent_status_bar"
    }
    assert first_provider_messages[-1].id == "paperpilot-status-attempt-1-sequence-1"
    assert first.result[0] is first_provider_messages[-1]

    second_request = _model_request(
        state={"messages": [], "run_model_call_count": 1},
    ).override(
        messages=[
            *first_provider_messages,
            *first.result[1:],
            ToolMessage(content="tool result", tool_call_id="tool-1"),
        ]
    )
    status.wrap_model_call(second_request, provider)
    second_provider_messages = provider_messages[1]
    assert second_provider_messages[: len(first_provider_messages)] == first_provider_messages
    assert second_provider_messages[-1].id == "paperpilot-status-attempt-1-sequence-2"


def test_status_wrapper_provider_retry_reuses_one_status_message() -> None:
    budget = ResearchToolBudgetMiddleware(run_limit=6)
    tracker = _tracker_for_middleware()
    status = ResearchStatusMiddleware(tracker=tracker, model_call_limit=6, tool_budget=budget)
    retry = ModelRetryMiddleware(max_retries=1, on_failure="error", initial_delay=0, jitter=False)
    provider_messages: list[list[BaseMessage]] = []
    attempts = 0

    def provider(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal attempts
        attempts += 1
        provider_messages.append(list(request.messages))
        if attempts == 1:
            raise ConnectionError("provider reset")
        return _model_response()

    result = status.wrap_model_call(
        _model_request(),
        lambda request: retry.wrap_model_call(request, provider),
    )

    assert result.result[0] is provider_messages[0][-1]
    assert provider_messages[0] == provider_messages[1]
    assert tracker.sequence == 1


def test_status_wrapper_render_failure_uses_fixed_unavailable_message() -> None:
    budget = ResearchToolBudgetMiddleware(run_limit=6)
    status = ResearchStatusMiddleware(
        tracker=_tracker_for_middleware(),
        model_call_limit=6,
        tool_budget=budget,
        renderer=lambda _snapshot: (_ for _ in ()).throw(RuntimeError("secret renderer error")),
    )
    captured: list[ModelRequest[Any]] = []

    result = status.wrap_model_call(
        _model_request(),
        lambda request: (captured.append(request) or _model_response()),
    )

    assert captured[0].messages[-1].content == STATUS_UNAVAILABLE_XML
    assert result.result[0] is captured[0].messages[-1]
    assert budget.used == 0


def test_status_wrapper_observes_business_tool_without_changing_result() -> None:
    budget = ResearchToolBudgetMiddleware(run_limit=6)
    tracker = _tracker_for_middleware()
    status = ResearchStatusMiddleware(tracker=tracker, model_call_limit=6, tool_budget=budget)
    request = ToolCallRequest(
        tool_call=_business_call("prepare_paper"),
        tool=None,
        state={"messages": []},
        runtime=_runtime({"messages": []}),
    )
    result = ToolMessage(content="ok", tool_call_id="call-1")

    assert status.wrap_tool_call(request, lambda _request: result) is result


def test_status_tool_observation_failure_preserves_business_result(monkeypatch) -> None:
    budget = ResearchToolBudgetMiddleware(run_limit=6)
    tracker = _tracker_for_middleware()
    monkeypatch.setattr(
        tracker,
        "record_business_tool_start",
        lambda _name, _arguments: (_ for _ in ()).throw(RuntimeError("tracker failed")),
    )
    status = ResearchStatusMiddleware(tracker=tracker, model_call_limit=6, tool_budget=budget)
    request = ToolCallRequest(
        tool_call=_business_call("prepare_paper"),
        tool=None,
        state={"messages": []},
        runtime=_runtime({"messages": []}),
    )
    business_result = ToolMessage(content="ok", tool_call_id="call-1")

    assert status.wrap_tool_call(request, lambda _request: business_result) is business_result


def test_status_observation_failure_preserves_provider_error(monkeypatch) -> None:
    budget = ResearchToolBudgetMiddleware(run_limit=6)
    tracker = _tracker_for_middleware()
    monkeypatch.setattr(
        tracker,
        "next_snapshot",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("tracker failed")),
    )
    status = ResearchStatusMiddleware(tracker=tracker, model_call_limit=6, tool_budget=budget)
    provider_error = ConnectionError("provider reset")

    with pytest.raises(ConnectionError) as exc_info:
        status.wrap_model_call(
            _model_request(),
            lambda _request: (_ for _ in ()).throw(provider_error),
        )

    assert exc_info.value is provider_error


class _SixCallModel(BaseChatModel):
    _invoke_count: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "six-call-status-test-model"

    @property
    def invoke_count(self) -> int:
        return self._invoke_count

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        del tools, tool_choice, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        self._invoke_count += 1
        if self._invoke_count <= 5:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo_status_value",
                        "args": {"value": str(self._invoke_count)},
                        "id": f"echo-{self._invoke_count}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_wrappers_allow_six_real_agent_model_calls_with_recursion_limit_24() -> None:
    @tool
    def echo_status_value(value: str) -> str:
        """Echo one scripted status-test value."""
        return value

    model = _SixCallModel()
    budget = ResearchToolBudgetMiddleware(run_limit=6)
    agent = create_agent(
        model=model,
        tools=[echo_status_value],
        middleware=[
            ModelCallLimitMiddleware(run_limit=6, exit_behavior="error"),
            budget,
            ResearchTodoMiddleware(),
            ResearchStatusMiddleware(
                tracker=_tracker_for_middleware(),
                model_call_limit=6,
                tool_budget=budget,
            ),
            ModelRetryMiddleware(max_retries=0, on_failure="error"),
        ],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="run six calls")]},
        config={"recursion_limit": 24},
    )

    assert result["messages"][-1].content == "done"
    assert model.invoke_count == 6
