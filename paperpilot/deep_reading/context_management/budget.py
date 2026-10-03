"""Stable, conservative token budgeting without network tokenizer calls."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from math import ceil
from typing import Any


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _message_payload(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        raw = dict(message)
        message_type = raw.get("type") or raw.get("role") or "message"
        role = {
            "human": "user",
            "ai": "assistant",
            "tool": "tool",
            "system": "system",
        }.get(str(message_type), str(message_type))
        return {
            "role": role,
            "name": raw.get("name"),
            "content": _jsonable(raw.get("content", "")),
            "tool_call_id": raw.get("tool_call_id"),
            "tool_calls": _jsonable(raw.get("tool_calls", [])),
            "additional_kwargs": _jsonable(raw.get("additional_kwargs", {})),
        }
    message_type = getattr(message, "type", message.__class__.__name__)
    role = {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
        "system": "system",
    }.get(str(message_type), str(message_type))
    return {
        "role": role,
        "name": getattr(message, "name", None),
        "content": _jsonable(getattr(message, "content", "")),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "tool_calls": _jsonable(getattr(message, "tool_calls", [])),
        "additional_kwargs": _jsonable(
            getattr(message, "additional_kwargs", {})
        ),
    }


def normalize_messages(messages: Sequence[Any], *, tool_schemas=()) -> str:
    """Return canonical JSON containing every billable message component."""
    payload = {
        "wrapper": "paperpilot-context-messages-v1",
        "messages": [_message_payload(message) for message in messages],
        "tool_schemas": _jsonable(list(tool_schemas)),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_tool_schemas(tool_schemas: Sequence[Any]) -> str:
    return json.dumps(
        {"tool_schemas": _jsonable(list(tool_schemas))},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fallback_tokens(text: str) -> int:
    if not text:
        return 0
    return max(ceil(len(text.encode("utf-8")) / 3), ceil(len(text) / 2))


class ModelAwareTokenCounter:
    """Use a provider counter when available and a conservative local fallback."""

    def __init__(self, model: Any | None = None) -> None:
        self._model = model

    def count_messages(self, messages: Sequence[Any], *, tool_schemas=()) -> int:
        if self._model is not None:
            counter = getattr(self._model, "get_num_tokens_from_messages", None)
            if callable(counter):
                try:
                    value = counter(list(messages))
                    if isinstance(value, int) and value >= 0:
                        if tool_schemas:
                            value += _fallback_tokens(
                                _normalize_tool_schemas(tool_schemas)
                            )
                        return value
                except Exception:
                    pass
        return _fallback_tokens(
            normalize_messages(messages, tool_schemas=tool_schemas)
        )

    def count_text(self, text: str) -> int:
        return _fallback_tokens(text)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        if self.count_text(text) <= max_tokens:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count_text(text[:middle]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        return text[:low]


def usable_input_budget(
    model_context_window_tokens: int,
    configured_max_output_tokens: int,
    context_safety_margin_ratio: float,
) -> int:
    if not 0 <= context_safety_margin_ratio < 1:
        raise ValueError("context_safety_margin_ratio must be between 0 and 1")
    budget = (
        model_context_window_tokens
        - configured_max_output_tokens
        - ceil(model_context_window_tokens * context_safety_margin_ratio)
    )
    if budget <= 0:
        raise ValueError("context configuration must leave a positive input budget")
    return budget


def reclaim_threshold(
    usable_budget: int,
    *,
    minimum_tokens: int = 8_000,
    minimum_ratio: float = 0.10,
) -> int:
    if usable_budget < 1 or minimum_tokens < 0 or not 0 <= minimum_ratio < 1:
        raise ValueError("invalid reclaim threshold bounds")
    return max(minimum_tokens, ceil(usable_budget * minimum_ratio))
