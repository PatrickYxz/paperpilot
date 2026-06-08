"""Encode/decode PaperPilot messages for JSON session storage."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any


def encode_messages(messages: list[dict]) -> list[dict]:
    """Convert runtime messages into JSON-serializable messages."""
    return [encode_message(message) for message in messages]


def decode_messages(payload: list[dict]) -> list[dict]:
    """Convert stored messages back into Anthropic-compatible dict messages."""
    return [decode_message(message) for message in payload]


def encode_message(message: dict) -> dict:
    role = message.get("role")
    if not isinstance(role, str):
        raise ValueError("message.role must be a string")
    return {
        "role": role,
        "content": _encode_content(message.get("content")),
    }


def decode_message(payload: dict) -> dict:
    role = payload.get("role")
    if not isinstance(role, str):
        raise ValueError("stored message.role must be a string")
    return {
        "role": role,
        "content": _decode_content(payload.get("content")),
    }


def _encode_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_encode_block(block) for block in content]
    return _json_safe(content)


def _decode_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_decode_block(block) for block in content]
    return _json_safe(content)


def _encode_block(block: Any) -> dict:
    if isinstance(block, Mapping):
        return {str(key): _json_safe(value) for key, value in block.items()}

    block_type = getattr(block, "type", None)
    if block_type == "text":
        return {
            "type": "text",
            "text": str(getattr(block, "text", "")),
        }
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": str(getattr(block, "id", "")),
            "name": str(getattr(block, "name", "")),
            "input": _json_safe(getattr(block, "input", {})),
        }

    if is_dataclass(block):
        return {
            str(key): _json_safe(value)
            for key, value in asdict(block).items()
        }

    dumped = _model_dump(block)
    if isinstance(dumped, Mapping):
        return {str(key): _json_safe(value) for key, value in dumped.items()}

    return {
        "type": str(block_type or type(block).__name__),
        "value": _json_safe(block),
    }


def _decode_block(block: Any) -> dict:
    if not isinstance(block, Mapping):
        return {"type": type(block).__name__, "value": _json_safe(block)}
    return {str(key): _json_safe(value) for key, value in block.items()}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))

    dumped = _model_dump(value)
    if dumped is not value:
        return _json_safe(dumped)

    return str(value)


def _model_dump(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return dict_method()
    return value
