"""agent_loop orchestration tests."""
from __future__ import annotations

import json

from paperpilot.core.adapter import ParsedResponse, Tool, ToolCall
from paperpilot.core.context_manager import ContextManager
from paperpilot.core.guardrail import Guardrail
from paperpilot.core.loop import agent_loop


class FakeClient:
    def __init__(self):
        self.calls = [
            ParsedResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="download-1",
                        name="mcp__arxiv__download_paper",
                        arguments={"arxiv_id": "1706.03762"},
                    )
                ],
                usage={"total_tokens": 1},
                raw=None,
            ),
            ParsedResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="build-1",
                        name="mcp__colbert__build_index",
                        arguments={},
                    )
                ],
                usage={"total_tokens": 1},
                raw=None,
            ),
            ParsedResponse(
                text="done",
                tool_calls=[],
                usage={"total_tokens": 1},
                raw=None,
            ),
        ]

    def call(self, messages, tools, *, system):
        return self.calls.pop(0)

    def append_assistant_turn(self, messages, response):
        messages.append({"role": "assistant", "content": response.text})

    def append_tool_results(self, messages, results):
        messages.append({
            "role": "user",
            "content": [
                {
                    "tool_use_id": r.id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in results
            ],
        })


class CompactingFakeClient:
    def __init__(self):
        self.calls = [
            ParsedResponse(
                text="SUMMARY",
                tool_calls=[],
                usage={"total_tokens": 1},
                raw=None,
            ),
            ParsedResponse(
                text="done",
                tool_calls=[],
                usage={"total_tokens": 1},
                raw=None,
            ),
        ]
        self.call_count = 0

    def call(self, messages, tools, *, system):
        self.call_count += 1
        return self.calls.pop(0)

    def append_assistant_turn(self, messages, response):
        messages.append({"role": "assistant", "content": response.text})

    def append_tool_results(self, messages, results):
        messages.append({"role": "user", "content": []})


def test_agent_loop_repairs_empty_build_index_args_from_download_result():
    build_args_seen = []
    events = []

    def download_handler(args):
        return json.dumps({"paper_id": args["arxiv_id"], "text": "full text"})

    def build_handler(args):
        build_args_seen.append(args)
        return {"indexed_count": len(args["documents"])}

    tools = [
        Tool(
            name="mcp__arxiv__download_paper",
            description="download",
            input_schema={},
            handler=download_handler,
        ),
        Tool(
            name="mcp__colbert__build_index",
            description="build",
            input_schema={},
            handler=build_handler,
        ),
    ]

    agent_loop(
        [{"role": "user", "content": "test"}],
        system="",
        tools=tools,
        client=FakeClient(),
        guardrail=Guardrail(max_iterations=5),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    assert build_args_seen == [
        {"documents": [{"paper_id": "1706.03762", "text": "full text"}]}
    ]
    assert any(kind == "tool_arg_repair" for kind, _ in events)


def test_agent_loop_emits_full_tool_result_content():
    class OneToolClient(FakeClient):
        def __init__(self):
            self.calls = [
                ParsedResponse(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            id="search-1",
                            name="mcp__colbert__search",
                            arguments={"query": "q"},
                        )
                    ],
                    usage={"total_tokens": 1},
                    raw=None,
                ),
                ParsedResponse(
                    text="done",
                    tool_calls=[],
                    usage={"total_tokens": 1},
                    raw=None,
                ),
            ]

    long_result = "chunk:" + ("x" * 1000)
    events = []
    tools = [
        Tool(
            name="mcp__colbert__search",
            description="search",
            input_schema={},
            handler=lambda args: long_result,
        )
    ]

    agent_loop(
        [{"role": "user", "content": "test"}],
        system="",
        tools=tools,
        client=OneToolClient(),
        guardrail=Guardrail(max_iterations=5),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    tool_results = [
        payload for kind, payload in events if kind == "tool_result"
    ]
    assert tool_results[0]["content"] == long_result


def test_agent_loop_auto_compacts_before_llm_call():
    messages = [{"role": "user", "content": "original"}]
    for i in range(10):
        messages.append({"role": "assistant", "content": "x" * 80 + str(i)})
    events = []

    agent_loop(
        messages,
        system="",
        tools=[],
        client=CompactingFakeClient(),
        guardrail=Guardrail(max_iterations=3),
        context_manager=ContextManager(
            window_tokens=120,
            expected_output_tokens=1,
            soft_ratio=0.5,
            hard_ratio=0.8,
            critical_ratio=0.99,
        ),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    assert any(kind == "auto_compact" for kind, _ in events)
    assert "<context_summary>" in messages[1]["content"]


def test_agent_loop_stops_when_context_still_critical_after_noop_compact():
    class NeverCalledClient(CompactingFakeClient):
        def call(self, messages, tools, *, system):
            raise AssertionError("client.call should not run")

    messages = [{"role": "user", "content": "x" * 500}]
    events = []

    agent_loop(
        messages,
        system="",
        tools=[],
        client=NeverCalledClient(),
        guardrail=Guardrail(max_iterations=3),
        context_manager=ContextManager(
            window_tokens=60,
            expected_output_tokens=1,
            soft_ratio=0.5,
            hard_ratio=0.8,
            critical_ratio=0.9,
        ),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    assert any(kind == "context_overflow" for kind, _ in events)
