"""Serializable one-boundary agent driver tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import paperpilot.agent.driver as driver_module
from paperpilot.agent.driver import (
    AgentLoopDriver,
    BudgetExceededError,
    ContextOverflowError,
    DriverAction,
    DriverState,
)
from paperpilot.agent.errors import ToolExecutionReplayRequired
from paperpilot.agent.policy import RunPolicy
from paperpilot.agent.tool_executor import ToolInvocationResult
from paperpilot.builtin_tools.research_todo import TodoStore, research_todo_tool
from paperpilot.core.adapter import ParsedResponse, Tool, ToolCall, ToolResult
from paperpilot.core.context_manager import ContextManager


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


class FakeClient:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.call_count = 0

    def call(self, messages, tools, *, system):
        self.call_count += 1
        return self.responses.pop(0)

    def append_assistant_turn(self, messages, response):
        content = response.raw.content if response.raw is not None else response.text
        messages.append({"role": "assistant", "content": content})

    def append_tool_results(self, messages, results):
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in results
            ],
        })


class RecordingExecutor:
    def __init__(self, tools=()):
        self.tools = {tool.name: tool for tool in tools}
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        call = kwargs["tool_call"]
        tool = self.tools.get(call.name)
        content = tool.handler(call.arguments) if tool is not None else f"result:{call.id}"
        result = ToolResult(id=call.id, content=str(content))
        return ToolInvocationResult(
            execution_id=f"execution:{call.id}",
            tool_result=result,
            result_preview=result.content[:1000],
            duration_ms=1,
            classification="read_only",
            sanitized_arguments=dict(call.arguments),
        )


def response(*, text=None, tool_calls=(), usage=None, blocks=None):
    raw = None if blocks is None else SimpleNamespace(content=list(blocks))
    return ParsedResponse(
        text=text,
        tool_calls=list(tool_calls),
        usage=usage or {"total_tokens": 1},
        raw=raw,
    )


def make_driver(
    *,
    client=None,
    tools=(),
    executor=None,
    context_manager=None,
    policy=None,
    resources=None,
):
    client = client or FakeClient()
    tool_list = list(tools)
    return AgentLoopDriver(
        client=client,
        tools=tool_list,
        system="system",
        context_manager=context_manager or ContextManager(),
        policy=policy or RunPolicy.for_depth("standard"),
        tool_executor=executor or RecordingExecutor(tool_list),
        resources=resources,
    )


def pending_state(*calls):
    return DriverState(
        messages=[{"role": "user", "content": "question"}],
        pending_tool_calls=list(calls),
    )


def execute_next(driver, state, sequence=1):
    return driver.execute(
        driver.plan_next(state),
        state,
        run_id="run-1",
        step_id=f"step-{sequence}",
        owner_id="worker-1",
    )


def test_driver_state_round_trip_preserves_pending_tool_batch():
    state = DriverState(
        messages=[{"role": "user", "content": "question"}],
        turn_count=1,
        tokens_used=12,
        downloaded_documents=[],
        todo_items=[{"content": "Read methods", "status": "in_progress"}],
        pending_tool_calls=[ToolCall(id="one", name="search", arguments={})],
        pending_tool_results=[ToolResult(id="done", content="result")],
        final_text=None,
    )

    encoded = state.to_dict()

    assert DriverState.from_dict(encoded) == state
    assert state.runtime_payload() == {
        key: value for key, value in encoded.items() if key != "messages"
    }
    json.dumps(encoded)


def test_driver_state_codec_preserves_anthropic_content_blocks():
    state = DriverState(messages=[{
        "role": "assistant",
        "content": [
            TextBlock(text="thinking"),
            ToolUseBlock(id="call-1", name="search", input={"query": "q"}),
        ],
    }])

    restored = DriverState.from_dict(state.to_dict())

    assert restored.messages[0]["content"] == [
        {"type": "text", "text": "thinking"},
        {
            "type": "tool_use",
            "id": "call-1",
            "name": "search",
            "input": {"query": "q"},
        },
    ]


def test_driver_action_is_json_serializable():
    action = DriverAction(kind="tool", input_data={"tool_call": {"id": "one"}})

    assert DriverAction.from_dict(action.to_dict()) == action
    json.dumps(action.to_dict())


def test_plan_next_prioritizes_finalize_then_pending_tool():
    driver = make_driver()
    completed = DriverState(messages=[], final_text="done")
    pending = pending_state(ToolCall(id="one", name="paper_deep_read", arguments={}))

    assert driver.plan_next(completed).kind == "finalize"
    assert driver.plan_next(pending).kind == "subagent"


@pytest.mark.parametrize(
    "state",
    (
        DriverState(messages=[], turn_count=4),
        DriverState(messages=[], tokens_used=20_000),
    ),
)
def test_plan_next_rejects_exhausted_budgets_as_non_retryable(state):
    driver = make_driver(policy=RunPolicy.for_depth("quick"))

    with pytest.raises(BudgetExceededError) as raised:
        driver.plan_next(state)

    assert raised.value.retryable is False


def test_driver_executes_one_llm_call_and_preserves_assistant_blocks():
    call = ToolCall(id="search-1", name="search", arguments={"query": "q"})
    blocks = [
        TextBlock(text="I will search"),
        ToolUseBlock(id=call.id, name=call.name, input=call.arguments),
    ]
    client = FakeClient([response(tool_calls=[call], usage={
        "input_tokens": 7,
        "output_tokens": 5,
    }, blocks=blocks)])
    driver = make_driver(client=client)

    result = execute_next(driver, DriverState(messages=[{
        "role": "user",
        "content": "question",
    }]))

    assert client.call_count == 1
    assert result.state.turn_count == 1
    assert result.state.tokens_used == 12
    assert result.state.pending_tool_calls == [call]
    assert result.state.messages[-1]["content"] == blocks
    assert result.final_text is None


def test_driver_llm_without_tools_sets_final_text_then_finalize_is_noop():
    client = FakeClient([response(text="answer", usage={"total_tokens": 3})])
    driver = make_driver(client=client)

    llm_result = execute_next(driver, DriverState(messages=[]))
    final_result = execute_next(driver, llm_result.state, sequence=2)

    assert client.call_count == 1
    assert llm_result.state.final_text == "answer"
    assert final_result.final_text == "answer"
    assert final_result.output_data == {"final_text": "answer"}


def test_driver_executes_one_tool_and_appends_batch_after_last_result():
    executor = RecordingExecutor()
    driver = make_driver(executor=executor)
    state = pending_state(
        ToolCall(id="one", name="search", arguments={}),
        ToolCall(id="two", name="search", arguments={}),
    )

    first = execute_next(driver, state)
    second = execute_next(driver, first.state, sequence=2)

    assert first.state.messages == state.messages
    assert len(first.state.pending_tool_results) == 1
    assert second.state.pending_tool_calls == []
    assert second.state.pending_tool_results == []
    assert second.state.messages[-1]["role"] == "user"
    assert len(second.state.messages[-1]["content"]) == 2
    assert [call["tool_call"].id for call in executor.calls] == ["one", "two"]
    assert executor.calls[0]["run_id"] == "run-1"
    assert executor.calls[0]["step_id"] == "step-1"
    assert executor.calls[0]["owner_id"] == "worker-1"
    assert second.tool_completion is not None
    json.dumps(second.to_dict())


def test_driver_repairs_build_index_without_mutating_checkpoint_input():
    executor = RecordingExecutor()
    driver = make_driver(executor=executor)
    call = ToolCall(id="build", name="mcp__colbert__build_index", arguments={})
    document = {"paper_id": "1706.03762", "text": "full text"}
    state = DriverState(
        messages=[],
        downloaded_documents=[document],
        pending_tool_calls=[call],
    )

    execute_next(driver, state)

    assert call.arguments == {}
    assert executor.calls[0]["tool_call"].arguments == {"documents": [document]}


def test_driver_checkpoints_downloaded_document_and_todo_items():
    todo_store = TodoStore()
    todo_tool = research_todo_tool(todo_store)
    download_tool = Tool(
        name="mcp__arxiv__download_paper",
        description="download",
        input_schema={},
        handler=lambda args: json.dumps({
            "paper_id": args["paper_id"],
            "text": "full text",
        }),
    )
    tools = [todo_tool, download_tool]
    executor = RecordingExecutor(tools)
    driver = make_driver(
        tools=tools,
        executor=executor,
        resources=SimpleNamespace(todo_store=todo_store),
    )
    items = [{"content": "Read methods", "status": "in_progress"}]
    state = pending_state(
        ToolCall(id="todo", name="research_todo", arguments={"todos": items}),
        ToolCall(
            id="download",
            name="mcp__arxiv__download_paper",
            arguments={"paper_id": "1706.03762"},
        ),
    )

    todo_result = execute_next(driver, state)
    download_result = execute_next(driver, todo_result.state, sequence=2)

    assert todo_result.state.todo_items == items
    assert download_result.state.downloaded_documents == [{
        "paper_id": "1706.03762",
        "text": "full text",
    }]


def test_driver_propagates_completed_tool_replay_without_changing_state():
    class ReplayExecutor:
        def execute(self, **kwargs):
            raise ToolExecutionReplayRequired(object())

    driver = make_driver(executor=ReplayExecutor())
    state = pending_state(ToolCall(id="one", name="search", arguments={}))

    with pytest.raises(ToolExecutionReplayRequired):
        execute_next(driver, state)

    assert len(state.pending_tool_calls) == 1
    assert state.pending_tool_results == []


def test_driver_compacts_once_and_returns_checkpointable_messages(monkeypatch):
    calls = []
    original = driver_module.compact_messages

    def counted_compact(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(driver_module, "compact_messages", counted_compact)
    messages = [{"role": "user", "content": "question"}]
    messages.extend(
        {"role": "assistant", "content": "x" * 80 + str(index)}
        for index in range(10)
    )
    client = FakeClient([response(text="SUMMARY")])
    driver = make_driver(
        client=client,
        context_manager=ContextManager(
            window_tokens=500,
            expected_output_tokens=1,
            soft_ratio=0.3,
            hard_ratio=0.8,
            critical_ratio=0.99,
        ),
    )
    state = DriverState(messages=messages)

    assert driver.plan_next(state).kind == "compact"
    result = execute_next(driver, state)

    assert len(calls) == 1
    assert client.call_count == 1
    assert "<context_summary>" in result.state.messages[1]["content"]


def test_driver_raises_non_retryable_overflow_when_compaction_cannot_help():
    client = FakeClient()
    driver = make_driver(
        client=client,
        context_manager=ContextManager(
            window_tokens=60,
            expected_output_tokens=1,
            soft_ratio=0.5,
            hard_ratio=0.8,
            critical_ratio=0.9,
        ),
    )
    state = DriverState(messages=[{"role": "user", "content": "x" * 500}])

    with pytest.raises(ContextOverflowError) as raised:
        execute_next(driver, state)

    assert raised.value.retryable is False
    assert client.call_count == 0
