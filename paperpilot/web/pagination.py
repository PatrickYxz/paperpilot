"""Opaque cursor helpers for bounded Web task pagination."""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import datetime

_CURSOR_VERSION = 1


@dataclass(frozen=True)
class TaskCursor:
    created_at: str
    task_id: str


class InvalidTaskCursor(ValueError):
    """Raised when a task cursor is invalid for the current request."""


def encode_task_cursor(
    *,
    user_id: str,
    status: str | None,
    created_at: str,
    task_id: str,
) -> str:
    payload = {
        "v": _CURSOR_VERSION,
        "u": user_id,
        "s": status,
        "t": created_at,
        "i": task_id,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_task_cursor(
    value: str,
    *,
    expected_user_id: str,
    expected_status: str | None,
) -> TaskCursor:
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(raw.decode())
        if not isinstance(payload, dict) or set(payload) != {"v", "u", "s", "t", "i"}:
            raise ValueError
        created_at = payload["t"]
        task_id = payload["i"]
        if (
            payload["v"] != _CURSOR_VERSION
            or payload["u"] != expected_user_id
            or payload["s"] != expected_status
            or not isinstance(created_at, str)
            or not isinstance(task_id, str)
            or not created_at
            or not task_id
        ):
            raise ValueError
        datetime.fromisoformat(created_at)
    except (
        binascii.Error,
        json.JSONDecodeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise InvalidTaskCursor("invalid task cursor") from exc
    return TaskCursor(created_at=created_at, task_id=task_id)
