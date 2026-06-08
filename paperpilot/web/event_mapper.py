"""Map PaperPilot agent events into bounded Web task events."""
from __future__ import annotations

import json
from typing import Any

MAX_EVENT_TEXT_CHARS = 800
MAX_ARGUMENT_PREVIEW_CHARS = 600
MAX_PAYLOAD_PREVIEW_CHARS = 1000


def map_paperpilot_event(kind: str, payload: dict | None) -> dict:
    """Convert one PaperPilot callback event to TaskStore.add_event kwargs."""
    event_payload = payload or {}
    if kind == "tool_call":
        name = str(event_payload.get("name", "unknown_tool"))
        return {
            "type": "progress",
            "stage": "tool_call",
            "message": f"Calling tool: {name}",
            "payload": {
                "source_kind": kind,
                "tool_name": name,
                "arguments_preview": _preview(
                    event_payload.get("arguments", {}),
                    MAX_ARGUMENT_PREVIEW_CHARS,
                ),
            },
        }
    if kind == "tool_result":
        name = str(event_payload.get("name", "unknown_tool"))
        content_preview, truncated = _preview_with_flag(
            event_payload.get("content", ""),
            MAX_EVENT_TEXT_CHARS,
        )
        return {
            "type": "progress",
            "stage": "tool_result",
            "message": f"Tool result: {name}",
            "payload": {
                "source_kind": kind,
                "tool_name": name,
                "content_preview": content_preview,
                "truncated": truncated,
            },
        }
    if kind == "turn":
        iteration = event_payload.get("iteration")
        return {
            "type": "progress",
            "stage": "agent_turn",
            "message": f"Agent turn {iteration}",
            "payload": {
                "source_kind": kind,
                "iteration": iteration,
                "tool_calls": _json_safe(event_payload.get("tool_calls", [])),
                "has_text": bool(event_payload.get("text")),
            },
        }
    if kind == "context_preflight":
        stage = str(event_payload.get("stage", "unknown"))
        return {
            "type": "progress",
            "stage": "context_preflight",
            "message": f"Context preflight: {stage}",
            "payload": {
                "source_kind": kind,
                "estimated_tokens": event_payload.get("estimated_tokens"),
                "window_tokens": event_payload.get("window_tokens"),
                "stage": stage,
            },
        }
    if kind == "auto_compact":
        result_preview, truncated = _preview_with_flag(
            event_payload.get("result", ""),
            MAX_EVENT_TEXT_CHARS,
        )
        return {
            "type": "progress",
            "stage": "auto_compact",
            "message": "Context compacted.",
            "payload": {
                "source_kind": kind,
                "result_preview": result_preview,
                "truncated": truncated,
            },
        }
    if kind in {"guardrail_stop", "context_overflow"}:
        reason = str(event_payload.get("reason") or kind)
        return {
            "type": "failed",
            "stage": "guardrail",
            "message": f"Guardrail stopped: {_truncate(reason, MAX_EVENT_TEXT_CHARS)}",
            "payload": {
                "source_kind": kind,
                "payload_preview": _preview(event_payload, MAX_PAYLOAD_PREVIEW_CHARS),
            },
        }

    return {
        "type": "progress",
        "stage": "agent_event",
        "message": f"Agent event: {kind}",
        "payload": {
            "source_kind": kind,
            "payload_preview": _preview(event_payload, MAX_PAYLOAD_PREVIEW_CHARS),
        },
    }


def _preview_with_flag(value: Any, max_chars: int) -> tuple[str, bool]:
    rendered = _render(value)
    truncated = len(rendered) > max_chars
    return _truncate(rendered, max_chars), truncated


def _preview(value: Any, max_chars: int) -> str:
    return _truncate(_render(value), max_chars)


def _render(value: Any) -> str:
    safe = _json_safe(value)
    if isinstance(safe, str):
        return safe
    return json.dumps(safe, ensure_ascii=False, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    return str(value)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."

