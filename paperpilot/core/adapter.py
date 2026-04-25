"""LLM 客户端:用 Anthropic SDK 打 DeepSeek 的 /anthropic 兼容端点。

DeepSeek 提供 Anthropic /v1/messages 兼容端点,tool_use block、
stop_reason、input_schema 都和 Claude 完全一致,直接用 anthropic SDK 即可。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from anthropic import Anthropic


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    id: str
    content: str
    is_error: bool = False


@dataclass
class ParsedResponse:
    text: str | None
    tool_calls: list[ToolCall]
    usage: dict
    raw: Any


class LLMClient:
    def __init__(self, model: str | None = None):
        self.client = Anthropic(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/anthropic",
        )
        self.model = model or os.environ.get("DEFAULT_MODEL", "deepseek-chat")

    def call(
        self, messages: list[dict], tools: list[Tool], *, system: str
    ) -> ParsedResponse:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            tools=[_schema(t) for t in tools] if tools else [],
            max_tokens=4096,
        )
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(
                    id=block.id, name=block.name, arguments=dict(block.input),
                ))
        return ParsedResponse(
            text="\n".join(texts) or None,
            tool_calls=calls,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            },
            raw=resp,
        )

    def append_assistant_turn(
        self, messages: list[dict], response: ParsedResponse
    ) -> None:
        # 复用原始 content block list,保证 tool_use_id 往返无损
        messages.append({"role": "assistant", "content": response.raw.content})

    def append_tool_results(
        self, messages: list[dict], results: list[ToolResult]
    ) -> None:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in results
            ],
        })


def _schema(t: Tool) -> dict:
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema,
    }