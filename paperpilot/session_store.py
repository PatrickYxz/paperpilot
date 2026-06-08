"""File-backed storage for recoverable PaperPilot conversation sessions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paperpilot.message_codec import decode_messages, encode_messages

SESSION_STORE_VERSION = 1
DEFAULT_SESSION_ROOT = Path("data/sessions")
_SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class SessionInfo:
    name: str
    updated_at: str
    message_count: int


class SessionStore:
    """Persist conversation messages as JSON files."""

    def __init__(self, root: Path = DEFAULT_SESSION_ROOT) -> None:
        self.root = root

    def exists(self, name: str) -> bool:
        return self._path_for(name).exists()

    def load(self, name: str) -> list[dict]:
        path = self._path_for(name)
        if not path.exists():
            return []
        payload = self._read_payload(path)
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError(f"session '{name}' messages must be a list")
        return decode_messages(messages)

    def save(self, name: str, messages: list[dict]) -> None:
        path = self._path_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = _utc_now()
        created_at = now
        if path.exists():
            payload = self._read_payload(path)
            existing_created_at = payload.get("created_at")
            if isinstance(existing_created_at, str):
                created_at = existing_created_at

        encoded = encode_messages(messages)
        payload = {
            "version": SESSION_STORE_VERSION,
            "session_name": name,
            "created_at": created_at,
            "updated_at": now,
            "messages": encoded,
            "metadata": {
                "message_count": len(encoded),
            },
        }
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def reset(self, name: str) -> None:
        self.save(name, [])

    def list_sessions(self) -> list[SessionInfo]:
        if not self.root.exists():
            return []
        sessions: list[SessionInfo] = []
        for path in sorted(self.root.glob("*.json")):
            payload = self._read_payload(path)
            metadata = payload.get("metadata")
            messages = payload.get("messages")
            message_count = 0
            if isinstance(metadata, dict):
                message_count = int(metadata.get("message_count", 0))
            elif isinstance(messages, list):
                message_count = len(messages)
            sessions.append(SessionInfo(
                name=str(payload.get("session_name") or path.stem),
                updated_at=str(payload.get("updated_at") or ""),
                message_count=message_count,
            ))
        return sessions

    def _path_for(self, name: str) -> Path:
        _validate_session_name(name)
        path = self.root / f"{name}.json"
        root_resolved = self.root.resolve()
        path_resolved = path.resolve()
        if root_resolved != path_resolved.parent:
            raise ValueError(f"invalid session name: {name!r}")
        return path

    def _read_payload(self, path: Path) -> dict:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid session file: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"session file must contain an object: {path}")
        version = payload.get("version")
        if version != SESSION_STORE_VERSION:
            raise ValueError(f"unsupported session version: {version!r}")
        return payload


def _validate_session_name(name: str) -> None:
    if not name or name in {".", ".."} or not _SESSION_NAME_RE.match(name):
        raise ValueError(f"invalid session name: {name!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

