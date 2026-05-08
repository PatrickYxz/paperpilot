"""Tests for paper_deep_read built-in subagent tool."""
from __future__ import annotations

import threading
import time
from typing import Callable

import pytest

from paperpilot.builtin_tools.subagent import (
    MAX_PAPERS,
    PAPER_DEEP_READ_NUDGE,
    SUBAGENT_MAX_ITER,
    SUBAGENT_SYSTEM,
    THREAD_POOL_SIZE,
    _extract_last_text,
    _filter_subagent_tools,
    _run_one,
    paper_deep_read_tool,
)
from paperpilot.core.adapter import ParsedResponse, Tool, ToolCall


class FakeClient:
    def __init__(self, responder: Callable[[list[dict]], ParsedResponse]):
        self._responder = responder
        self.call_count = 0

    def call(self, messages, tools, *, system):
        self.call_count += 1
        return self._responder(messages)

    def append_assistant_turn(self, messages, response):
        messages.append({"role": "assistant", "content": response.text or ""})

    def append_tool_results(self, messages, results):
        messages.append({
            "role": "user",
            "content": [
                {
                    "tool_use_id": result.id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in results
            ],
        })


def _text_response(text: str) -> ParsedResponse:
    return ParsedResponse(text=text, tool_calls=[], usage={"total_tokens": 50}, raw=None)


def _tool_call_response(name: str, args: dict | None = None) -> ParsedResponse:
    return ParsedResponse(
        text=None,
        tool_calls=[ToolCall(id="tc-1", name=name, arguments=args or {})],
        usage={"total_tokens": 50},
        raw=None,
    )


def _stub_tool(name: str, return_value: str = "ok") -> Tool:
    return Tool(
        name=name,
        description="stub",
        input_schema={},
        handler=lambda args: return_value,
    )


def test_filter_subagent_tools_keeps_only_deep_read_tools():
    tools = [
        _stub_tool("mcp__arxiv__search_papers"),
        _stub_tool("mcp__arxiv__download_paper"),
        _stub_tool("mcp__colbert__build_index"),
        _stub_tool("mcp__colbert__search"),
        _stub_tool("mcp__graph__build_graph"),
        _stub_tool("paper_deep_read"),
    ]
    kept = _filter_subagent_tools(tools)
    assert [tool.name for tool in kept] == [
        "mcp__arxiv__download_paper",
        "mcp__colbert__build_index",
        "mcp__colbert__search",
    ]


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    type = "tool_use"


def test_extract_last_text_from_anthropic_blocks():
    messages = [
        {"role": "assistant", "content": [_TextBlock("A"), _ToolUseBlock()]},
        {"role": "user", "content": "tool result"},
        {"role": "assistant", "content": [_TextBlock("B"), _TextBlock("C")]},
    ]
    assert _extract_last_text(messages) == "B\nC"


def test_extract_last_text_from_string_content():
    messages = [{"role": "assistant", "content": "summary text"}]
    assert _extract_last_text(messages) == "summary text"


def test_extract_last_text_returns_none_when_missing():
    assert _extract_last_text([{"role": "user", "content": "hi"}]) is None


def test_handler_rejects_zero_papers():
    tool = paper_deep_read_tool(
        client_factory=lambda: FakeClient(lambda m: _text_response("x")),
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    with pytest.raises(ValueError, match="1..8"):
        tool.handler({"paper_ids": [], "user_query": "q"})


def test_handler_rejects_too_many_papers():
    tool = paper_deep_read_tool(
        client_factory=lambda: FakeClient(lambda m: _text_response("x")),
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    with pytest.raises(ValueError, match=f"got {MAX_PAPERS + 1}"):
        tool.handler({
            "paper_ids": [f"p{i}" for i in range(MAX_PAPERS + 1)],
            "user_query": "q",
        })


def test_run_one_returns_ok_status_on_success():
    summary = "## Core Method\nFoo\n## Key Findings\nBar"
    client = FakeClient(lambda m: _text_response(summary))
    result = _run_one(
        "1706.03762",
        "attention",
        client_factory=lambda: client,
        tools=[],
        on_event=lambda k, v: None,
    )
    assert result == {
        "paper_id": "1706.03762",
        "summary": summary,
        "status": "ok",
    }


def test_run_one_returns_max_iter_when_guardrail_stops():
    client = FakeClient(lambda m: _tool_call_response("noop"))
    result = _run_one(
        "1706.03762",
        "q",
        client_factory=lambda: client,
        tools=[_stub_tool("noop")],
        on_event=lambda k, v: None,
        max_iter=2,
    )
    assert result["paper_id"] == "1706.03762"
    assert result["status"] == "max_iter_reached"


def test_run_one_returns_error_status_on_exception():
    def bad(messages):
        raise RuntimeError("boom")

    result = _run_one(
        "1706.03762",
        "q",
        client_factory=lambda: FakeClient(bad),
        tools=[],
        on_event=lambda k, v: None,
    )
    assert result["paper_id"] == "1706.03762"
    assert result["summary"] == ""
    assert result["status"].startswith("error: RuntimeError: boom")


def test_handler_aggregates_papers():
    tool = paper_deep_read_tool(
        client_factory=lambda: FakeClient(lambda m: _text_response("done")),
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    out = tool.handler({"paper_ids": ["A", "B", "C"], "user_query": "q"})
    assert out.startswith("## Paper Deep Read Results (3 papers)")
    for paper_id in ("A", "B", "C"):
        assert f"### {paper_id} (status: ok)" in out


def test_handler_one_paper_failed_others_ok():
    def factory():
        def responder(messages):
            if " B." in messages[0]["content"]:
                raise RuntimeError("paper B explodes")
            return _text_response("ok done")

        return FakeClient(responder)

    tool = paper_deep_read_tool(
        client_factory=factory,
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    out = tool.handler({"paper_ids": ["A", "B", "C"], "user_query": "q"})
    assert "### A (status: ok)" in out
    assert "### B (status: error: RuntimeError: paper B explodes)" in out
    assert "### C (status: ok)" in out


def test_handler_preserves_input_order():
    def factory():
        def responder(messages):
            if " A." in messages[0]["content"]:
                time.sleep(0.03)
                return _text_response("summary A")
            if " C." in messages[0]["content"]:
                time.sleep(0.01)
                return _text_response("summary C")
            return _text_response("summary B")

        return FakeClient(responder)

    tool = paper_deep_read_tool(
        client_factory=factory,
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    out = tool.handler({"paper_ids": ["A", "B", "C"], "user_query": "q"})
    assert out.index("### A") < out.index("### B") < out.index("### C")


def test_event_aggregation_prefixes_paper_id():
    received: list[tuple[str, dict]] = []
    received_lock = threading.Lock()

    def main_emit(kind: str, payload: dict) -> None:
        with received_lock:
            received.append((kind, dict(payload)))

    tool = paper_deep_read_tool(
        client_factory=lambda: FakeClient(lambda m: _text_response("done")),
        mcp_tools=[],
        on_event=main_emit,
    )
    tool.handler({"paper_ids": ["A", "B"], "user_query": "q"})

    turn_payloads = [payload for kind, payload in received if kind == "turn"]
    assert len(turn_payloads) == 2
    assert {payload["subagent_paper_id"] for payload in turn_payloads} == {"A", "B"}

    start_payloads = [
        payload for kind, payload in received if kind == "subagent_start"
    ]
    done_payloads = [
        payload for kind, payload in received if kind == "subagent_done"
    ]
    assert {payload["subagent_paper_id"] for payload in start_payloads} == {"A", "B"}
    assert {payload["subagent_paper_id"] for payload in done_payloads} == {"A", "B"}
    assert {payload["status"] for payload in done_payloads} == {"ok"}


def test_tool_metadata_and_worker_count():
    tool = paper_deep_read_tool(
        client_factory=lambda: FakeClient(lambda m: _text_response("x")),
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    assert tool.name == "paper_deep_read"
    assert tool.input_schema["additionalProperties"] is False
    assert tool.input_schema["required"] == ["paper_ids", "user_query"]
    assert tool.input_schema["properties"]["paper_ids"]["maxItems"] == MAX_PAPERS
    assert MAX_PAPERS == 8
    assert SUBAGENT_MAX_ITER == 8
    assert THREAD_POOL_SIZE == 3
    assert "paper_deep_read" in PAPER_DEEP_READ_NUDGE
    assert "paper_ids" not in SUBAGENT_SYSTEM
    assert "paper_id=" in SUBAGENT_SYSTEM
    assert "mcp__colbert__search(query" in SUBAGENT_SYSTEM


def test_subagent_system_requires_three_targeted_searches_and_short_answer():
    assert "at least 3 times" in SUBAGENT_SYSTEM
    assert "Search 1: the user's direct question" in SUBAGENT_SYSTEM
    assert "Search 2: key terms" in SUBAGENT_SYSTEM
    assert "Search 3: likely evidence locations" in SUBAGENT_SYSTEM
    assert "## Short Answer" in SUBAGENT_SYSTEM
    assert "## Evidence" in SUBAGENT_SYSTEM
    assert "exact atomic fact" in SUBAGENT_SYSTEM
