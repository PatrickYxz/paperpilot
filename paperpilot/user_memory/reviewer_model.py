"""Reviewer model wiring for the memory-card approval stage.

The reviewer is configured by PAPERPILOT_MEMORY_REVIEWER_MODEL: unset or
"off" disables the stage; "qwen-*" routes to DashScope (different model
family from the DeepSeek proposer, the preferred pairing); anything else
is a DeepSeek model name (e.g. "deepseek-flash" — same family, cheaper,
still a second prompt with a risk-control bias).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Sequence

from dashscope import Generation
from pydantic import BaseModel

_LOGGER = logging.getLogger("paperpilot.user_memory")

_REVIEWER_MODEL_ENV = "PAPERPILOT_MEMORY_REVIEWER_MODEL"


class _StructuredDashScope:
    def __init__(self, parent: "DashScopeChatModel", schema: type[BaseModel]):
        self._parent = parent
        self._schema = schema

    def invoke(self, messages: Sequence[object]) -> BaseModel:
        payload = [
            {"role": _role_of(message), "content": _content_of(message)}
            for message in messages
        ]
        schema_json = json.dumps(
            self._schema.model_json_schema(), ensure_ascii=False
        )
        payload[-1]["content"] += (
            "\n\nRespond with a single JSON object matching this schema "
            f"exactly, no extra text:\n{schema_json}"
        )
        response = Generation.call(
            model=self._parent.model_name,
            api_key=self._parent.api_key,
            messages=payload,
            result_format="message",
            response_format={"type": "json_object"},
            temperature=0,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"dashscope reviewer call failed: {response.code} {response.message}"
            )
        raw = response.output.choices[0].message.content
        return self._schema.model_validate(json.loads(raw))


class DashScopeChatModel:
    def __init__(
        self,
        model_name: str = "qwen-plus",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")

    def with_structured_output(self, schema: type[BaseModel]):
        return _StructuredDashScope(self, schema)


class DeepSeekReviewerModel:
    """JSON-mode structured wrapper for DeepSeek reviewer models.

    deepseek-flash is a thinking model and rejects the tool_choice forcing
    that with_structured_output uses, so structure comes from an explicit
    schema instruction plus JSON response format instead.
    """

    def __init__(self, model_name: str) -> None:
        from langchain_deepseek import ChatDeepSeek

        self._chat = ChatDeepSeek(model=model_name, temperature=0).bind(
            response_format={"type": "json_object"}
        )

    def with_structured_output(self, schema: type[BaseModel]):
        return _JsonStructuredChat(self._chat, schema)


class _JsonStructuredChat:
    def __init__(self, chat: object, schema: type[BaseModel]) -> None:
        self._chat = chat
        self._schema = schema

    def invoke(self, messages: Sequence[object]) -> BaseModel:
        payload = [
            {"role": _role_of(message), "content": _content_of(message)}
            for message in messages
        ]
        schema_json = json.dumps(
            self._schema.model_json_schema(), ensure_ascii=False
        )
        payload[-1]["content"] += (
            "\n\nRespond with a single JSON object matching this schema "
            f"exactly, no extra text:\n{schema_json}"
        )
        result = self._chat.invoke(payload)
        return self._schema.model_validate(json.loads(result.content))


def build_reviewer_model_from_env():
    """Reviewer per env; unset/"off" disables the stage entirely."""
    name = os.environ.get(_REVIEWER_MODEL_ENV, "").strip()
    if not name or name.lower() in {"off", "disabled"}:
        return None
    if name.startswith("qwen"):
        return DashScopeChatModel(model_name=name)
    return DeepSeekReviewerModel(model_name=name)


def _role_of(message: object) -> str:
    role = getattr(message, "type", None)
    if role == "system":
        return "system"
    if role in {"ai", "assistant"}:
        return "assistant"
    return "user"


def _content_of(message: object) -> str:
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise TypeError("reviewer messages must carry string content")
    return content
