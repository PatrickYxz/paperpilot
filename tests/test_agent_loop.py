"""agent_loop orchestration tests."""
from __future__ import annotations

import json

from paperpilot.core.adapter import ParsedResponse, Tool, ToolCall
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
